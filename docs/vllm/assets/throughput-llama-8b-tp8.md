| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 313 | 69 | 286 | **72** | 289 | **74** |
| generate, 8 concurrent | 1,689 | 496 | 1,568 | **511** | 1,540 | **525** |
| capture 1 layer, every step | · | 64 | **284** | **68** | 251 | 65 |
| capture every layer, every step | · | **57** | **276** | 22 | 36 | 36 |
| capture 1 layer, 8 concurrent | · | 442 | **1,443** | 440 | 1,082 | 415 |
| additive steering, 1 layer | · | 64 | **278** | **70** | 251 | 57 |
| logit lens every step | · | **63** | **275** | 23 | 30 | 52 |
| linear probe every step | · | 64 | **276** | **68** | 245 | 56 |
| zero one attention head every step | · | 64 | ✗ | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 65 | 284 | ✗ | ✗ | ✗ |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.25 s | 1.21 s | 1.18 s | · | · | **1.02 s** |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 2.24 s | 2.21 s | · | · | ✗ |
