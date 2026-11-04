# Attribution

The CLIP implementation, tokenizer, and BPE vocabulary derive from IRRA and OpenAI CLIP. Their original MIT notices are preserved in [IRRA LICENSE](src/enhanced_irra/vendor/LICENSE) and [CLIP LICENSE](src/enhanced_irra/vendor/CLIP_LICENSE).

The architecture illustration is original artwork based on this project's implementation. The README screenshots document the project's research interface with RSTPReid examples; they are not a hosted application or evidence of real-world identity. RSTPReid is built from MSMT17. Dataset and model archives are excluded from this public repository.

Research sources, pinned upstream commits, and implementation adaptations are recorded in [research/sources.json](research/sources.json).

## Paper figure excerpts

The screenshots under `assets/papers/` are cropped excerpts from the original figures and remain the work of their respective authors:

- `irra-relation-reasoning.png`: Jiang and Ye, IRRA, CVPR 2023, Figure 2, PDF page 4. https://arxiv.org/pdf/2303.12501
- `kpdm-frequency-weighting.png`: You et al., KPDM, AAAI 2026, Figure 3(b), PDF page 4. https://ojs.aaai.org/index.php/AAAI/article/download/38199/42161
- `image-aware-masking.png`: Bitton et al., Data Efficient Masked Language Modeling for Vision and Language, EMNLP 2021, Figure 1, PDF page 1. https://aclanthology.org/2021.findings-emnlp.259.pdf

They illustrate the specific ideas discussed in the README; paper figures are separate from this project's original architecture illustration and measured results. Source URLs, PDF checksums, page numbers, and crop coordinates are recorded in `research/paper-figure-sources.json`.
