# Data and model access

Target repository: https://github.com/deyiwang89/lychee-mov3f-clean

The code is maintained in this repository. The supplementary images and model files are distributed as release assets rather than Git objects.

Prepared separate assets:

1. `supplementary_1409_original.zip`: full-view images in seven class folders,
   metadata removed without changing decoded pixels, labels and provenance CSVs.
2. `supplementary_1409_crops.zip`: lossless PNG union-box crops with 10% margin,
   identical formal evaluation inputs after preprocessing.
3. `model_weights.zip`: 26 PTH, 11 distinct classification KModels, 4 ONNX
   exports/intermediates and evaluation evidence. See `docs/precision_and_deployment.md`
   and the model manifest for stage, configuration and verification status.
4. `real_scene_23.zip`: qualitative self-acquired canopy photographs, without
   expert per-leaf labels. Not an accuracy benchmark.

The original supplementary train/valid/test labels are archive metadata, not
a newly created evaluation split. The 951 representatives are specified by
CSV rather than extra copies or random sampling. Grouping does not establish
physical-leaf independence or independent external validation.

The primary 3765-image deduplicated protocol is reconstructed using its cited
source archive and the checked-in manifests. The primary raw archive is not
redistributed as part of this package.

The owner confirmed public-release permission. CC BY 4.0 is the intended data
license, subject to completing source attribution and checking all original
permissions for compatibility. `source_permissions.csv` identifies outstanding
evidence. Do not interpret code MIT licensing as a license to third-party images.
This gate also covers the photographic panels and six selected crops in the code
tree. Model redistribution terms also require explicit owner confirmation.

Release tag: `v2026.09.30`. Release assets and their SHA-256 values are listed at
https://github.com/deyiwang89/lychee-mov3f-clean/releases/tag/v2026.09.30 and in
`release_manifest.csv` attached to that release. Do not interpret the release as
an independent external validation dataset.
