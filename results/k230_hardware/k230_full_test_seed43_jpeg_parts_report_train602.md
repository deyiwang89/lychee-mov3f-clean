# K230 Seed-43 Test Evaluation (Split JPEG Bundle)

- Model: `mobilenetv3_fractal_seed43_noclip_train602_w8_a16.kmodel`
- Precision: 8-bit weights, 16-bit activations (NoClip PTQ)
- Completed: 3012/3012
- K230 accuracy: 1711/3012 (56.806%)
- PC/K230 Top-1 agreement: 1718/3012 (57.039%)
- Core latency: mean 62.98 ms, median 63.00 ms, range 62-66 ms

| True class | Samples | K230 correct | K230 accuracy | PC/K230 agreement |
|---|---:|---:|---:|---:|
| Anthrax_Leaf | 118 | 76 | 64.41% | 64.41% |
| Bituminous_Leaf | 110 | 31 | 28.18% | 29.09% |
| Curl_Leaf | 142 | 136 | 95.77% | 95.77% |
| Deficiency_Leaf | 98 | 83 | 84.69% | 81.63% |
| Dry_Leaf | 305 | 226 | 74.10% | 74.10% |
| Felt_Leaf | 306 | 5 | 1.63% | 2.94% |
| Fungal_Leaf_Spot | 459 | 0 | 0.00% | 0.00% |
| Healthy_Leaf | 502 | 306 | 60.96% | 61.75% |
| Leaf_Blight | 474 | 358 | 75.53% | 75.53% |
| Leaf_Gall | 498 | 490 | 98.39% | 98.59% |

Core latency excludes JPEG file writes, acquisition, display, and power measurement.
