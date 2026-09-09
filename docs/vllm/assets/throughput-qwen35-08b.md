| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 420 | 47 | **251** | 48 | 241 | 48 |
| generate, 8 concurrent | 2,522 | 346 | **1,545** | 354 | 1,406 | **358** |
| capture 1 layer, every step | · | 45 | 245 | **49** | 239 | 35 |
| capture every layer, every step | · | 41 | **239** | 40 | 215 | 32 |
| capture 1 layer, 8 concurrent | · | 319 | 1,374 | **352** | 1,370 | 254 |
| additive steering, 1 layer | · | 44 | 241 | **49** | 239 | 32 |
| logit lens every step | · | **44** | **226** | 36 | 89 | 31 |
| linear probe every step | · | 44 | 238 | **48** | 238 | 32 |
| zero one attention head every step | · | 45 | ✗ | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 45 | 245 | ✗ | ✗ | ✗ |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.45 s | 1.30 s | 1.28 s | · | · | **0.81 s** |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 0.91 s | 0.82 s | · | · | ✗ |
