| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 61 | 30 | 59 | **32** | 59 | **32** |
| generate, 8 concurrent | 351 | 202 | 331 | **218** | 334 | **219** |
| capture 1 layer, every step | · | 28 | **58** | **31** | 55 | **29** |
| capture every layer, every step | · | **24** | **57** | 5 | 7 | 10 |
| capture 1 layer, 8 concurrent | · | 185 | **323** | 189 | 279 | 173 |
| additive steering, 1 layer | · | 27 | **58** | **31** | 56 | 21 |
| logit lens every step | · | **28** | **58** | 16 | 21 | 21 |
| linear probe every step | · | 28 | **58** | **30** | 56 | 22 |
| zero one attention head every step | · | 28 | ✗ | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 28 | 58 | ✗ | ✗ | ✗ |
| sweep: 1024 × 1 token, capture 1 layer, per request | 3.16 s | 5.51 s | 5.12 s | · | · | 5.70 s |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 6.24 s | 5.70 s | · | · | ✗ |
