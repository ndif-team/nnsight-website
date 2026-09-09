| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 92 | 86 | 89 | 86 | 90 | 86 |
| generate, 8 concurrent | 618 | 577 | 597 | 580 | 596 | 578 |
| capture 1 layer, every step | · | 79 | 89 | **84** | 89 | 75 |
| capture every layer, every step | · | **68** | **88** | 48 | 71 | 37 |
| capture 1 layer, 8 concurrent | · | 529 | 577 | **568** | 588 | 470 |
| additive steering, 1 layer | · | 78 | 89 | **85** | 89 | 49 |
| logit lens every step | · | **78** | **84** | 63 | 65 | 47 |
| linear probe every step | · | 78 | 89 | **84** | 89 | 49 |
| zero one attention head every step | · | 79 | 89 | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 79 | 89 | ✗ | ✗ | ✗ |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.75 s | 1.60 s | 1.48 s | · | · | **1.33 s** |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 1.09 s | 1.00 s | · | · | ✗ |
