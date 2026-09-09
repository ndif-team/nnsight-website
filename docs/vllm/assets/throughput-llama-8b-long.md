| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 90 | 84 | 87 | 85 | 87 | 85 |
| generate, 8 concurrent | 574 | 539 | 558 | 537 | 553 | 539 |
| capture 1 layer, every step | · | 77 | 87 | **84** | 87 | 76 |
| capture every layer, every step | · | **66** | **86** | 48 | 67 | 38 |
| capture 1 layer, 8 concurrent | · | 530 | 553 | 528 | 544 | 468 |
| additive steering, 1 layer | · | 75 | 86 | **84** | 87 | 50 |
| logit lens every step | · | **77** | **82** | 65 | 64 | 48 |
| linear probe every step | · | 75 | 86 | **83** | 87 | 49 |
| zero one attention head every step | · | 77 | 87 | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 76 | 87 | ✗ | ✗ | ✗ |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.76 s | 1.64 s | 1.54 s | · | · | **1.35 s** |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 1.12 s | 1.03 s | · | · | ✗ |
