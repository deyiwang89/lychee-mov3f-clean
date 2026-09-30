# Data access and provenance

The primary raw archive remains an external input. The 1409-image classification
archives are separate local release assets; see ../DATA_ACCESS.md for status and
permission gates. The primary protocol uses 3765 unique image files, stratified
20% development and 80% held-out image-level test partitions, seeds 42/43/44,
602 training and 151 validation images per seed. These are not independent
orchard or physical-leaf populations.

The 23 self-acquired photographs are now a separate `real_scene_23.zip` asset.
They have no per-leaf expert truth and must not be used to calculate external
accuracy. The six selected Fig. 8 crops are in `fig8_selected/`. Their pixels are
preserved and metadata cleaned; all image publishing remains subject to the
documented permission and attribution gate.

For the real-scene run, each source photograph is processed at 0.4 and 0.6 times its shorter edge with a stride of 0.5 times the window edge. Crops are resized to 224 x 224. Candidate filtering uses confidence >= 0.80, IoU suppression at 0.30, and at most five candidates per image. These settings are recorded in `results/real_scene_sliding_window/manifest_sha256.json`.
