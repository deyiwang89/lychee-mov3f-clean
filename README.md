# Lychee-MV3F revision release candidate (2026-09-29)

This package contains the reproducibility code for the revision release. The
publication target is `deyiwang89/lychee-mov3f-clean`.
See [REPRODUCE.md](REPRODUCE.md) for the current executable entrypoints and
[DATA_ACCESS.md](DATA_ACCESS.md) for the separate data/model assets.

**Precision clarification:** Fig. 1 and the graphical abstract now distinguish
offline FP32 from the two tested K230 artifacts. See
[the five-row table and evidence mapping](docs/precision_and_deployment.md).
Model assets preserve 26 PTH, 11 KModel and 4 ONNX files. Historical photos
remain unchanged; dataset attribution/publication gates still apply.

The release restores the missing training implementation, model registry,
loader utilities and seed-specific manifests. It retains the frozen
602/151/3012 image-level protocol and seeds 42, 43 and 44. The 1409-image
supplement is descriptive, not independent external validation.

Validated locally with Python 3.9, PyTorch 2.1.0+cu121, torchvision 0.16.0,
NumPy 1.26.4 and Pillow 9.3.0: 25 checkpoints loaded strictly, CPU forward
passes returned finite ten-class outputs, and all 1409 original/crop input
tensors were identical. No retraining or new board measurements were run.

The current Fig. 8 uses six fixed crops with the seed-43 PyTorch hybrid and
backbone-gradient heatmaps, not quantized K230 predictions. The command is
`scripts/generate_fig8_unified.py`; its CPU predictions were checked against
the archived figure records. Other legacy figure scripts remain for provenance
and must not be substituted for this command.

The data and checkpoint ZIP files belong in release assets, not Git history.
The 23 photographs are a separate asset. Six selected crops and photographic
figures in this tree remain subject to the data-permission and attribution
terms documented in `DATA_ACCESS.md`.

Code retains MIT terms and upstream attribution in `LICENSE-UPSTREAM`/`NOTICE`.
Images are not licensed under the code MIT grant.
