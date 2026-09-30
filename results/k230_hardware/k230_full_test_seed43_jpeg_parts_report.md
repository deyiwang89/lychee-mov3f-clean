# K230 Seed-43 Test Evaluation (Split JPEG Bundle)

- Model: `mobilenetv3_fractal_seed43_noclip_w8_a16.kmodel`
- Precision: 8-bit weights, 16-bit activations (NoClip PTQ)
- Completed: 3012/3012
- K230 accuracy: 1799/3012 (59.728%)
- PC/K230 Top-1 agreement: 1804/3012 (59.894%)
- Core latency: mean 63.14 ms, median 63.00 ms, range 62-66 ms

| True class | Samples | K230 correct | K230 accuracy | PC/K230 agreement |
|---|---:|---:|---:|---:|
| Anthrax_Leaf | 118 | 83 | 70.34% | 70.34% |
| Bituminous_Leaf | 110 | 35 | 31.82% | 32.73% |
| Curl_Leaf | 142 | 136 | 95.77% | 95.77% |
| Deficiency_Leaf | 98 | 89 | 90.82% | 87.76% |
| Dry_Leaf | 305 | 237 | 77.70% | 77.70% |
| Felt_Leaf | 306 | 3 | 0.98% | 2.29% |
| Fungal_Leaf_Spot | 459 | 0 | 0.00% | 0.22% |
| Healthy_Leaf | 502 | 360 | 71.71% | 72.31% |
| Leaf_Blight | 474 | 368 | 77.64% | 77.64% |
| Leaf_Gall | 498 | 488 | 97.99% | 97.79% |

Core latency excludes JPEG file writes, acquisition, display, and power measurement.
