| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 140 | 35 | 106 | **37** | 103 | **36** |
| generate, 8 concurrent | 857 | 263 | **648** | **274** | 627 | 270 |
| capture 1 layer, every step | · | 33 | 104 | **36** | 102 | 26 |
| capture every layer, every step | · | **31** | **103** | 28 | 87 | 22 |
| capture 1 layer, 8 concurrent | · | 242 | 604 | **263** | **623** | 193 |
| additive steering, 1 layer | · | 33 | 104 | **36** | 102 | 24 |
| logit lens every step | · | **33** | **97** | 28 | 60 | 23 |
| linear probe every step | · | 33 | 104 | **36** | 102 | 24 |
| zero one attention head every step | · | 33 | ✗ | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 33 | 104 | ✗ | ✗ | ✗ |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.96 s | 2.01 s | 2.01 s | · | · | **1.59 s** |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 1.65 s | 1.49 s | · | · | ✗ |
