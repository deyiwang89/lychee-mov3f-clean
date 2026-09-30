# Precision and execution-stage records

| Model / stage | Seed | Calibration images | Correct / test images | Accuracy (%) |
|---|---:|---:|---:|---:|
| PC FP32 | 42 | N/A | 2982 / 3012 | 99.00 |
| PC FP32 | 43 | N/A | 2978 / 3012 | 98.87 |
| PC FP32 | 44 | N/A | 2955 / 3012 | 98.11 |
| K230 PTQ | 43 | 100 | 1799 / 3012 | 59.73 |
| K230 PTQ | 43 | 602 | 1711 / 3012 | 56.81 |

FP32 denotes 32-bit floating point. The offline runs use their respective image-level test splits (mean +/- sample SD: 98.66 +/- 0.48%). Both K230 artifacts were converted from seed 43 using NoClip calibration, uint8 (unsigned 8-bit integer) weights, and int16 (signed 16-bit integer) activations. Calibration counts are not test-set sizes. Board inputs were center-cropped and JPEG-reencoded from the seed-43 test images, whereas the offline evaluation used the original images. Runtime calls averaged 63.14 ms and 62.98 ms for the 100- and 602-image calibrations, respectively, excluding file loading/decoding, camera acquisition, and display.

## Traceability

`precision_stage_comparison.csv` links each result to its checkpoint SHA-256.
`model_manifest.csv` distinguishes full-test accuracy, diagnostic agreement,
strict checkpoint loading and unverified historical claims.

The model asset preserves 26 PTH files, 11 distinct classification KModels and
4 ONNX exports/intermediates. The original 25 checkpoints and six KModels remain.
No model is retrained or overwritten. Original filenames containing `w8_a16`
are retained for identification; they denote the archived compiler settings,
not a new algorithm or a claim about every operator's arithmetic.

The legacy floating-point CSV and confusion matrix both contain 3312 correct
predictions among 3392 images (97.64%). Its checkpoint passed strict loading;
the full accuracy run was not repeated. The legacy 2.34 MiB quantized model is
preserved, but the original 96.95% claim lacks a located evaluation record.
It is not relabeled with either of the newer seed-43 device accuracies.

The 19/20 diagnostic record is agreement with PC predictions on selected crops,
not 95% classification accuracy on the full dataset. The failed automatic
mixed-quantization intermediate is an ONNX file, not a successful KModel.

`evidence_index.csv` in the model asset records original and packaged hashes.
Packaged text replaces machine-local path prefixes with `PROJECT`; numerical
records are unchanged. Original files remain intact. Public evidence paths
are identifiers, not promises that primary raw images are redistributed.
The JPEG-preparation metadata documents the limited first-100-image PC check.

Fig. 1 and the graphical abstract separately label offline and K230 results.
Original photos are unchanged. The photographed display is a historical
interface illustration, not a new end-to-end frame-rate benchmark.

No new training, full accuracy rerun, calibration, conversion or board test
was performed during this clarification. No remote upload has occurred.
