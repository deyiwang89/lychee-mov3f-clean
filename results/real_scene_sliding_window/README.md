# Rebuilt Fig. 8: real-scene sliding-window diagnosis

This run uses frozen MobileNetV3-Fractal seed-43 on the archived original whole-scene photographs. It generates 40% and 60% short-edge square windows with a 50% window-size stride, then retains high-confidence (>= 0.80) windows after IoU NMS (0.30). The images do not have per-leaf expert ground truth; these results are qualitative diagnostic demonstrations and do not report an external-test accuracy. `fig8_selection.csv` records the four panels used in the manuscript figure.

The generated `crops/` cache is excluded from the public repository because it is reproducible from the 23 source photographs, the checked-in code, and the withheld seed-43 checkpoint. Run `scripts/run_real_scene_sliding_window.py` to recreate it locally.
