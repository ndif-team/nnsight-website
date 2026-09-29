| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens | vanilla vLLM 0.20.2 | TransformerLens (compile + CUDA graphs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 228 | 72 | 216 | 69 | 216 | **76** | 231 | ✗ |
| generate, 8 concurrent | 1,431 | 532 | 1,355 | 512 | 1,345 | **550** | 1,460 | ✗ |
| capture 1 layer, every step | · | 66 | **213** | **69** | 206 | **69** | · | ✗ |
| capture every layer, every step | · | **58** | **208** | 29 | 54 | 37 | · | ✗ |
| capture 1 layer, 8 concurrent | · | 463 | **1,256** | **482** | 1,163 | 430 | · | ✗ |
| additive steering, 1 layer | · | 66 | 210 | **71** | 210 | 58 | · | ✗ |
| logit lens every step | · | **66** | **205** | 33 | 48 | 55 | · | ✗ |
| linear probe every step | · | 66 | 209 | **69** | 206 | 57 | · | ✗ |
| zero one attention head every step | · | 66 | ✗ | ✗ | ✗ | ✗ | · | ✗ |
| override the sampled token every step | · | 67 | 213 | ✗ | ✗ | ✗ | · | ✗ |
| one forward over 512 tokens, capture 1 layer | 23 ms | 39 ms | 39 ms | **34 ms** | **34 ms** | **29 ms** | 19 ms | 101 ms |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.33 s | 1.20 s | 1.18 s | · | · | **1.00 s** | 0.33 s | 12.59 s |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 1.46 s | 1.35 s | · | · | ✗ | · | ✗ |
