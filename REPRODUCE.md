# Reproduction entrypoints

Run commands from this repository root. The primary image archive is an external
input from the dataset cited in the paper. Restore its `datasets/train/<class>/`
and `datasets/test/<class>/` paths; the checked-in `data/manifests/seed_*/`
files preserve the actual experimental membership. Do not regenerate splits
merely because the original folder names look like a conventional split.

## Environment

Validated environment: Python 3.9, torch 2.1.0+cu121, torchvision 0.16.0+cu121,
numpy 1.26.4, Pillow 9.3.0. Install `requirements.txt` in a separate environment.
CUDA is optional for the package's CPU checks; training used the archived CUDA
configuration. `nncase`, ONNX tools and CanMV firmware are separate deployment
dependencies, not implied to be installed by the training requirements.

Download/extract the release-candidate assets beside the code directory:

```text
github_repo/
public_assets/
  model_weights/pytorch/best_weights/
  supplementary_1409_original/
  supplementary_1409_crops/
  real_scene_23/
```

No public asset URL is asserted before actual release. See DATA_ACCESS.md.

## Frozen-checkpoint evaluation

```powershell
python scripts/evaluate_model.py --help
python scripts/evaluate_classification_package.py --data-root ../public_assets/supplementary_1409_original --kind original --weights ../public_assets/model_weights/pytorch/best_weights/fractal_seed_43/best_epoch_weights.pth --output-dir outputs/supplement_original
python scripts/evaluate_classification_package.py --data-root ../public_assets/supplementary_1409_crops --kind crop --weights ../public_assets/model_weights/pytorch/best_weights/fractal_seed_43/best_epoch_weights.pth --output-dir outputs/supplement_crops
```

Default supplementary evaluation selects the fixed 951 representatives.
Add `--all` for 1409 rows. The model still predicts ten classes; seven-folder
alphabetical indices must not be substituted for the original output order.
The two package readers produce identical tensors. Cropped PNGs must not be
cropped a second time. Results written by these commands are new local outputs,
not replacements for the archived paper tables.

## Training and sensitivity (not run during packaging)

```powershell
python scripts/run_training_matrix.py --dry-run --output-root outputs/dry_run/weights --results-root outputs/dry_run/results
python scripts/run_training_matrix.py --seeds 42 43 44 --variants gap wide_gap gap_gmp fractal fractal_only
python scripts/run_sensitivity_matrix.py --seeds 42
python scripts/run_sensitivity_matrix.py --settings gated --seeds 43 44
```

The second through fourth commands perform long training jobs: run them only
intentionally. They require the authorized primary dataset and ImageNet
pretraining weights resolved by the model constructors. `--no-pretrained` is
for smoke tests, not reproduction of the paper's reported results.
The historical `gap_matched` checkpoint is not a substitute for `wide_gap`.

## Figures

```powershell
python scripts/generate_core_figures.py --core-summary results/primary_ablation/capacity_control_summary.json --output-dir outputs/figures
python scripts/generate_sensitivity_figure.py --sensitivity-summary results/primary_ablation/sensitivity_summary.json --robustness results/robustness/fractal_seed_43.json --output-dir outputs/figures
python scripts/generate_secondary_1409_figure.py --output outputs/figures/fig9.png
python scripts/generate_fig8_unified.py --weights ../public_assets/model_weights/pytorch/best_weights/fractal_seed_43/best_epoch_weights.pth --output-dir outputs/fig8 --device cpu
```

The final command is the current Fig. 8 workflow. It uses six fixed manifest
selections and writes predictions, gradient overlays and provenance together.
The manifest is not modified. It does not compute field accuracy or isolate
the hard-threshold branch's spatial contribution.

## K230

`k230/main.py` and `k230/libs/` preserve the operator-assisted runtime.
`k230/k230_full_test_jpeg_parts_*.py` preserve the timed test pathway.
These scripts run under CanMV/MicroPython on the board, not ordinary CPython.
Firmware revisions are in `k230/revision.txt`. Conversion entrypoints are
`scripts/export_seed43_onnx.py` and `scripts/convert_seed43_k230_*.py`.
Use their `--help` and explicit artifact/calibration paths; do not infer
calibration from a `.kmodel` filename alone.

Archived hardware results describe one setup. The 63 ms interval is a runtime
call, not camera-to-display throughput; 3.14 W is estimated system input;
39.1 degrees C is a maximum surface temperature. No board test was run while
assembling this release.


## Precision-stage mapping

See `docs/precision_and_deployment.md` and `docs/precision_stage_comparison.csv`
before choosing a checkpoint. The three offline FP32 results are not board
accuracies. Both full-test KModels originate from seed 43, with uint8 weights
and int16 activations. Use the model manifest SHA-256 to select the 100- or
602-image-calibrated artifact. The former reports 59.73%/63.14 ms; the latter
56.81%/62.98 ms. Calibration counts are not test counts.
Preserved legacy and diagnostic files have separate verification statuses.
