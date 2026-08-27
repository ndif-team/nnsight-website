| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 91 | 67 | 88 | 69 | 88 | 68 |
| generate, 8 concurrent | 604 | 471 | 583 | **496** | 580 | **490** |
| capture 1 layer, every step | · | 60 | 87 | **67** | 87 | 60 |
| capture every layer, every step | · | **52** | **86** | 39 | 67 | 32 |
| capture 1 layer, 8 concurrent | · | 419 | 562 | **475** | 568 | 387 |
| additive steering, 1 layer | · | 59 | 87 | **67** | 87 | 42 |
| logit lens every step | · | **61** | **82** | 51 | 63 | 41 |
| linear probe every step | · | 60 | 87 | **68** | 87 | 42 |
| zero one attention head every step | · | 61 | 87 | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 60 | 87 | ✗ | ✗ | ✗ |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.78 s | 1.74 s | 1.59 s | · | · | **1.41 s** |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 1.19 s | 1.04 s | · | · | ✗ |
