"""Reproducible visual lexicon policies with exact occurrence-to-CLIP-BPE alignment."""

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import regex
import torch
from torch import Tensor

from enhanced_irra.config import Masking
from enhanced_irra.data import ImageRecord
from enhanced_irra.vendor.simple_tokenizer import SimpleTokenizer, basic_clean, whitespace_clean

VOCAB_SIZE: int = 49408
MASK_ID: int = 49405
SOT_ID: int = 49406
EOT_ID: int = 49407
CONTEXT: int = 77
ADJECTIVES: frozenset[str] = frozenset("black white red blue green yellow pink purple orange grey gray brown beige tan navy maroon silver gold golden turquoise khaki dark light bright pale striped checked checkered plaid patterned floral spotted dotted plain long short sleeveless tight loose baggy leather denim wool woolen knitted puffy padded hooded bald curly straight blond blonde brunette round large small tall slim".split())
NOUNS: frozenset[str] = frozenset("jacket jackets coat coats shirt shirts tshirt t-shirt sweater sweaters sweatshirt hoodie hoodies cardigan vest suit uniform dress dresses skirt skirts jeans trousers pants shorts leggings shoes shoe sneakers sneaker boots boot sandals sandal slippers heels backpack backpacks bag bags handbag handbags purse purses suitcase luggage cap caps hat hats beanie scarf scarves gloves glove glasses sunglasses watch belt necklace headphones umbrella phone hair beard moustache mustache headwear footwear socks sock mask masks".split())
STOPWORDS: frozenset[str] = frozenset("a an the and or but of in on at to from with without for into inside is are was were be been being he she him her his hers their they them it its this that these those as by who which while has have had wearing wears wear carrying carries carry walking walks walk looking looks look man woman person people male female pair there some very also".split())


class TokenizerInterface(Protocol):
    encoder: dict[str, int]

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


@dataclass(frozen=True)
class WordSpan:
    text: str
    positions: tuple[int, ...]
    visual: bool
    stopword: bool


@dataclass(frozen=True)
class EncodedCaption:
    text: str
    token_ids: tuple[int, ...]
    words: tuple[WordSpan, ...]
    phrases: tuple[tuple[int, ...], ...]
    truncated: bool


@dataclass(frozen=True)
class MaskedCaption:
    clean: Tensor
    corrupted: Tensor
    targets: Tensor
    selected: tuple[int, ...]
    visual_selected: int
    stopword_selected: int
    eligible_tokens: int
    phrase_count: int
    selected_phrase_count: int


class TokenizerConnector:
    """Typed interface to the exact upstream IRRA tokenizer (MIT, vendored)."""

    def __init__(self, bpe_path: Path) -> None:
        self.tokenizer: TokenizerInterface = cast(TokenizerInterface, SimpleTokenizer(str(bpe_path)))
        if self.tokenizer.encoder["<|mask|>"] != MASK_ID:
            raise ValueError("Tokenizer is incompatible with the official IRRA vocabulary")

    def encode(self, text: str) -> EncodedCaption:
        """Align complete lexical occurrences; truncated words/phrases are never candidates."""
        normalized: str = cast(str, whitespace_clean(basic_clean(text))).lower()
        # Use the upstream lexical regex, including contractions and punctuation runs.
        pieces: list[str] = cast(list[str], SimpleTokenizerPattern.findall(normalized))
        ids: list[int] = []
        words: list[WordSpan] = []
        for piece in pieces:
            subwords: list[int] = self.tokenizer.encode(piece)
            start: int = 1 + len(ids)
            ids.extend(subwords)
            if start + len(subwords) <= CONTEXT - 1:
                words.append(WordSpan(piece, tuple(range(start, start + len(subwords))), piece in ADJECTIVES or piece in NOUNS, piece in STOPWORDS or not any(c.isalnum() for c in piece)))
        if ids != self.tokenizer.encode(normalized):
            raise ValueError(f"Word-to-BPE alignment failed for caption: {text!r}")
        phrases: list[tuple[int, ...]] = []
        index: int = 0
        while index < len(words):
            end: int = index
            while end < len(words) and words[end].text in ADJECTIVES:
                end += 1
            if end > index and end < len(words) and words[end].text in NOUNS:
                phrases.append(tuple(p for word in words[index:end + 1] for p in word.positions))
                index = end + 1
            else:
                index += 1
        content: list[int] = ids[:CONTEXT - 2]
        tokens: tuple[int, ...] = tuple([SOT_ID] + content + [EOT_ID] + [0] * (CONTEXT - 2 - len(content)))
        return EncodedCaption(text, tokens, tuple(words), tuple(phrases), len(ids) > CONTEXT - 2)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))


# Compile once using the exact upstream regex; no NLP model download is required.
SimpleTokenizerPattern: regex.Pattern[str] = regex.compile(r"""<\|startoftext\|>|<\|mask\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", regex.IGNORECASE)


def tokenizer_connector() -> TokenizerConnector:
    return TokenizerConnector(Path(__file__).parent / "vendor" / "data" / "bpe_simple_vocab_16e6.txt.gz")


def select_groups(groups: tuple[tuple[int, ...], ...], target: int, rng: random.Random) -> tuple[int, ...]:
    """Shuffle candidate spans, select whole spans to reach the explicit token budget.

    Indivisible phrases can overshoot the budget; the effective rate is recorded.
    An empty candidate set stays empty, with zero MLM contribution.
    """
    shuffled: list[tuple[int, ...]] = rng.sample(list(groups), k=len(groups))
    selected: set[int] = set()
    for group in shuffled:
        if len(selected) >= target:
            break
        selected.update(group)
    return tuple(sorted(selected))


def mask_caption(caption: EncodedCaption, strategy: Masking, probability: float, seed: int) -> MaskedCaption:
    """Clone clean IDs, sample targets, then use IRRA/BERT 80/10/10 corruption.

    Random matches upstream eligibility (0 < id < 49405), with forced first token
    when no position is drawn. Semantic policies select whole words or phrases
    toward ceil(p × content BPE count); no random-policy substitution occurs.
    """
    if not 0 < probability < 1:
        raise ValueError("Mask probability must be in (0, 1)")
    rng: random.Random = random.Random(seed)
    eligible: tuple[int, ...] = tuple(i for i, token in enumerate(caption.token_ids) if 0 < token < MASK_ID)
    if not eligible:
        raise ValueError(f"No maskable CLIP tokens in {caption.text!r}")
    random_replacements: dict[int, int] = {}
    if strategy == "random":
        random_selected: list[int] = []
        for position in eligible:
            draw: float = rng.random()
            if draw < probability:
                random_selected.append(position)
                corruption_draw: float = draw / probability
                if corruption_draw < 0.8:
                    random_replacements[position] = MASK_ID
                elif corruption_draw < 0.9:
                    random_replacements[position] = rng.randrange(1, MASK_ID)
                else:
                    random_replacements[position] = caption.token_ids[position]
        selected: tuple[int, ...] = tuple(random_selected)
        if not selected:
            selected = (eligible[0],)
            random_replacements[eligible[0]] = MASK_ID
    elif strategy == "attribute":
        groups: tuple[tuple[int, ...], ...] = tuple(w.positions for w in caption.words if w.visual)
        selected = select_groups(groups, math.ceil(probability * len(eligible)), rng)
    elif strategy == "phrase":
        selected = select_groups(caption.phrases, math.ceil(probability * len(eligible)), rng)
    else:
        raise ValueError(f"Unknown masking policy: {strategy}")
    clean: Tensor = torch.tensor(caption.token_ids, dtype=torch.long)
    corrupted: Tensor = clean.clone()
    targets: Tensor = torch.full_like(clean, -100)
    for position in selected:
        targets[position] = clean[position]
        if strategy == "random":
            corrupted[position] = random_replacements[position]
        else:
            corruption_draw = rng.random()
            if corruption_draw < 0.8:
                corrupted[position] = MASK_ID
            elif corruption_draw < 0.9:
                corrupted[position] = rng.randrange(1, MASK_ID)
    visual: set[int] = {p for w in caption.words if w.visual for p in w.positions}
    stopwords: set[int] = {p for w in caption.words if w.stopword for p in w.positions}
    selected_set: set[int] = set(selected)
    return MaskedCaption(clean, corrupted, targets, selected, len(visual & selected_set), len(stopwords & selected_set), len(eligible), len(caption.phrases), sum(set(p).issubset(selected_set) for p in caption.phrases))


def frequency_weights(records: Sequence[ImageRecord], tokenizer: TokenizerConnector, power: float) -> Tensor:
    """KPDM-inspired (1-f)^power for top-25 train-only BPE IDs; others weight 1.

    Here f is relative corpus occurrence frequency, explicitly resolving an
    underspecified paper detail. Floor 0.05 bounds relative gradient amplification
    by 20×; no inverse-frequency explosion or validation/test text is used.
    """
    if not records or any(r.split != "train" for r in records):
        raise ValueError("Frequency fitting requires nonempty training records only")
    counts: Counter[int] = Counter(token for r in records for c in r.captions for token in tokenizer.encode(c).token_ids if 0 < token < MASK_ID)
    total: int = sum(counts.values())
    weights: Tensor = torch.ones(VOCAB_SIZE, dtype=torch.float32)
    for token, count in counts.most_common(25):
        weights[token] = max(0.05, (1 - count / total) ** power)
    return weights
