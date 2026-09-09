| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 210 | 40 | **205** | 41 | 181 | **42** |
| generate, 8 concurrent | 671 | 303 | **1,091** | 309 | 874 | **312** |
| capture 1 layer, every step | · | 38 | **202** | **41** | 179 | 39 |
| capture every layer, every step | · | **35** | **200** | 34 | 148 | 33 |
| capture 1 layer, 8 concurrent | · | 280 | **1,023** | **304** | 863 | 279 |
| additive steering, 1 layer | · | 38 | **201** | **41** | 180 | 36 |
| logit lens every step | · | **38** | **188** | 34 | 99 | 35 |
| linear probe every step | · | 38 | **200** | **40** | 179 | 36 |
| zero one attention head every step | · | 38 | 202 | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 38 | 202 | ✗ | ✗ | ✗ |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.44 s | 1.36 s | 1.27 s | · | · | **0.82 s** |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 0.71 s | 0.66 s | · | · | ✗ |
