| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 37 | 30 | 36 | 29 | 36 | **32** |
| generate, 8 concurrent | 235 | 200 | 226 | 203 | 226 | **214** |
| capture 1 layer, every step | · | 27 | 35 | **29** | 35 | **28** |
| capture every layer, every step | · | **23** | **35** | 7 | 10 | 10 |
| capture 1 layer, 8 concurrent | · | 179 | **220** | **187** | 212 | 170 |
| additive steering, 1 layer | · | 27 | 35 | **29** | 35 | 19 |
| logit lens every step | · | **27** | **35** | 19 | 22 | 18 |
| linear probe every step | · | 27 | 35 | **29** | 35 | 19 |
| zero one attention head every step | · | 27 | 35 | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 27 | 35 | ✗ | ✗ | ✗ |
