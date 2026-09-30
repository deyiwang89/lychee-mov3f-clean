# Revision Release: 2026-09-30

Published release package. Target repository:
https://github.com/deyiwang89/lychee-mov3f-clean
Release assets: https://github.com/deyiwang89/lychee-mov3f-clean/releases/tag/v2026.09.30

## Included

- Restored training code, the required model constructors and loader utilities.
- Preserved seed-42/43/44 training, validation and test manifests.
- Added classification-package evaluation for full-view originals and lossless crops.
- Updated the fixed-six-image Fig. 8 generation entrypoint and provenance output.
- Included archived measurements, predictions and configuration mappings.
- Retained upstream license and attribution separately from dataset permissions.

## Separate Assets

- `supplementary_1409_original.zip`: 1409 full-view JPEGs in seven class folders.
- `supplementary_1409_crops.zip`: 1409 lossless PNG crops using the recorded union-box margin.
- Both packages include the same 951-representative selection and explicit ten-class model indices.
- `model_weights.zip`: 26 PTH, 11 KModels, 4 ONNX files and linked evaluation evidence. Historical claims and diagnostic records are separate from full-test results.
- `real_scene_23.zip`: 23 canopy photographs, not expert-annotated leaf-level test data.

Weights and full datasets should be Release attachments, not committed Git objects.
Published version tag: `v2026.09.30`.

## Checks Performed

All 25 checkpoints loaded strictly and produced finite ten-class CPU outputs.
Both data readers produced exactly equal tensors for all 1409 samples.
Main command-line help, training-matrix dry-run and selected figure generation
were checked. No training run, full accuracy rerun or new K230 measurement was made.

## Publication Gates

The owner confirmed permission to publish the images. Item-level attribution
and compatibility with the intended CC BY 4.0 grant remain to be documented.
This also covers selected crop images and photographic figures inside this tree.
Code MIT terms do not license the images.

Fig. 1 and the graphical abstract have been updated after author approval of the precision-stage clarification. The release package is separate from the journal submission files.

Historical metadata paths are explanatory identifiers, not a guarantee that
every historical input is redistributed. See `docs/archived_records.md`.

## Precision Clarification

| Model / stage | Seed | Calibration images | Correct / test images | Accuracy (%) |
|---|---:|---:|---:|---:|
| PC FP32 | 42 | N/A | 2982 / 3012 | 99.00 |
| PC FP32 | 43 | N/A | 2978 / 3012 | 98.87 |
| PC FP32 | 44 | N/A | 2955 / 3012 | 98.11 |
| K230 PTQ | 43 | 100 | 1799 / 3012 | 59.73 |
| K230 PTQ | 43 | 602 | 1711 / 3012 | 56.81 |

FP32 denotes 32-bit floating point. The offline runs use their respective image-level test splits (mean +/- sample SD: 98.66 +/- 0.48%). Both K230 artifacts were converted from seed 43 using NoClip calibration, uint8 (unsigned 8-bit integer) weights, and int16 (signed 16-bit integer) activations. Calibration counts are not test-set sizes. Board inputs were center-cropped and JPEG-reencoded from the seed-43 test images, whereas the offline evaluation used the original images. Runtime calls averaged 63.14 ms and 62.98 ms for the 100- and 602-image calibrations, respectively, excluding file loading/decoding, camera acquisition, and display.

The legacy FP32 strict-load check is additional to the earlier 25-checkpoint checks. No complete accuracy rerun or new board test was performed.
