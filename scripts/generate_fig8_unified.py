"""Generate Fig. 8 from one frozen inference and Grad-CAM-style pass.

The six crop selections are fixed by ``selection_manifest.csv``.  This script
uses the same bilinear preprocessing for prediction and visualization, writes
the predictions used in the figure, and records file hashes and preprocessing
metadata.  The heatmap is a backbone-gradient visualization; it is not an
independent explanation of the hard-threshold fractal descriptor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from PIL import Image


SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets import get_model_from_name
from utils.utils import get_classes


WEIGHTS = None
MANIFEST = PROJECT_ROOT / "data/fig8_selected/selection_manifest.csv"
OUT_DIR = PROJECT_ROOT / "outputs/fig8_unified"
FIGURE = OUT_DIR / "fig_rebuttal_real_scene_sliding_window.png"

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = (224, 224)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess(path: Path) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).float()
    return image, tensor


def load_model(device: torch.device):
    classes, count = get_classes(str(PROJECT_ROOT / "model_data" / "cls_classes.txt"))
    model = get_model_from_name["mobilenetv3_fractal"](num_classes=count, pretrained=False)
    model.head.frac.box_sizes = (1, 2, 4)
    model.head.frac.threshold = 0.5
    checkpoint = torch.load(WEIGHTS, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    return model.to(device).eval(), classes


def cam_for(model, tensor: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    target = model.features[-1]

    def forward_hook(_module, _inputs, output):
        activations["x"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["x"] = grad_output[0]

    forward_handle = target.register_forward_hook(forward_hook)
    backward_handle = target.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor.unsqueeze(0).to(device))
        index = int(logits.argmax(dim=1).item())
        logits[0, index].backward()
        activation = activations["x"].detach()
        gradient = gradients["x"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=IMAGE_SIZE, mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
        probabilities = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
    finally:
        forward_handle.remove()
        backward_handle.remove()
    return cam, probabilities


def write_figure(records: list[dict[str, object]], figure_path: Path) -> None:
    figure = plt.figure(figsize=(9.2, 4.8), dpi=450, facecolor="white")
    outer = figure.add_gridspec(2, 3, left=0.025, right=0.975, bottom=0.075, top=0.90, wspace=0.08, hspace=0.14)
    for index, record in enumerate(records):
        row, column = divmod(index, 3)
        inner = outer[row, column].subgridspec(2, 2, height_ratios=(0.24, 1.0), hspace=0.02, wspace=0.025)
        title_axis = figure.add_subplot(inner[0, :])
        title_axis.axis("off")
        ax_original = figure.add_subplot(inner[1, 0])
        ax_heatmap = figure.add_subplot(inner[1, 1])
        ax_original.imshow(record["image"])
        ax_heatmap.imshow(record["overlay"])
        for axis in (ax_original, ax_heatmap):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        label = str(record["predicted_label"]).replace("_", " ")
        title = f"({chr(97 + index)}) {label}\np = {float(record['confidence']):.4f}"
        title_axis.text(0.5, 0.48, title, ha="center", va="center", fontsize=8.0, fontweight="bold")
        ax_original.set_title("original", fontsize=6.1, pad=1)
        ax_heatmap.set_title("gradient heatmap", fontsize=6.1, pad=1)
    figure.suptitle(
        "Qualitative predictions on selected real-scene crops (PyTorch seed 43)",
        fontsize=11,
        y=0.965,
    )
    figure.text(
        0.5,
        0.012,
        "Bilinear resize and ImageNet normalization; heatmaps visualize backbone gradients only.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    global WEIGHTS, MANIFEST, OUT_DIR, FIGURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    WEIGHTS, MANIFEST, OUT_DIR = args.weights, args.manifest, args.output_dir
    FIGURE = OUT_DIR / "fig_rebuttal_real_scene_sliding_window.png"
    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (OUT_DIR / "selection_manifest_before_unified.csv").exists():
        shutil.copy2(MANIFEST, OUT_DIR / "selection_manifest_before_unified.csv")
    model, classes = load_model(device)
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig", newline="")))
    records: list[dict[str, object]] = []
    for row in rows:
        path = Path(row["crop_path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        image, tensor = preprocess(path)
        cam, probabilities = cam_for(model, tensor, device)
        index = int(np.argmax(probabilities))
        heat = cm.jet(cam)[..., :3]
        original = np.asarray(image, dtype=np.float32) / 255.0
        overlay = np.clip(0.55 * original + 0.45 * heat, 0.0, 1.0)
        row.update(
            predicted_label=classes[index],
            confidence=f"{float(probabilities[index]):.12g}",
            top3_labels="|".join(classes[i] for i in np.argsort(probabilities)[::-1][:3]),
            top3_probabilities="|".join(f"{float(probabilities[i]):.6f}" for i in np.argsort(probabilities)[::-1][:3]),
            entropy=str(float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))),
        )
        records.append({"image": original, "overlay": overlay, "predicted_label": classes[index], "confidence": float(probabilities[index])})
        panel = np.concatenate([original, overlay], axis=1)
        Image.fromarray(np.uint8(np.clip(panel, 0, 1) * 255)).save(OUT_DIR / f"{path.stem}_original_heatmap.png")
    fields = list(rows[0].keys())
    with (OUT_DIR / "selection_manifest_evaluated.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    prediction_fields = ["crop_name", "crop_path", "predicted_label", "confidence", "top3_labels", "top3_probabilities", "entropy"]
    with (OUT_DIR / "fig8_predictions_unified.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in prediction_fields})
    write_figure(records, FIGURE)
    provenance = {
        "checkpoint": str(WEIGHTS),
        "checkpoint_sha256": sha256(WEIGHTS),
        "device": str(device),
        "model": "mobilenetv3_fractal",
        "fractal_box_sizes": [1, 2, 4],
        "fractal_threshold": 0.5,
        "preprocessing": {"resize": "224x224 bilinear", "normalization": "ImageNet mean/std", "color": "RGB"},
        "heatmap": "Grad-CAM-style backbone feature-gradient visualization; it does not isolate the hard-threshold fractal branch.",
        "selection": "Six preselected crops from the archived sliding-window crop set; no new images or labels.",
        "records": [
            {
                "crop_name": row["crop_name"],
                "crop_sha256": sha256(PROJECT_ROOT / row["crop_path"]),
                "predicted_label": row["predicted_label"],
                "confidence": float(row["confidence"]),
            }
            for row in rows
        ],
    }
    (OUT_DIR / "fig8_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# Fig. 8 unified generation\n\n"
        "The manifest selections, predictions, heatmaps, and labels in the figure were produced in one run with the frozen seed-43 PyTorch checkpoint. "
        "The six panels are qualitative examples without per-crop expert truth. The heatmap is a backbone-gradient visualization and cannot establish an independent spatial contribution from the hard-threshold fractal descriptor.\n",
        encoding="utf-8",
    )
    print(json.dumps({"figure": str(FIGURE), "predictions": str(OUT_DIR / "fig8_predictions_unified.csv"), "device": str(device)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
