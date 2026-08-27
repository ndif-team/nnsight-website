| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: |
| generate | 164 | 31 | 161 | 32 |
| generate, 8 concurrent | 873 | 232 | 871 | 237 |
| capture 1 layer, every step | · | 29 | 159 | **30** |
| capture every layer, every step | · | **27** | 157 | 26 |
| capture 1 layer, 8 concurrent | · | 213 | 810 | 215 |
| additive steering, 1 layer | · | **29** | 158 | 28 |
| logit lens every step | · | **29** | 152 | 27 |
| linear probe every step | · | **29** | 157 | 27 |
| zero one attention head every step | · | 29 | ✗ | ✗ |
| override the sampled token every step | · | 29 | 159 | ✗ |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.50 s | 1.53 s | 1.42 s | **0.92 s** |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 0.94 s | 0.71 s | ✗ |
