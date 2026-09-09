| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 395 | 145 | 372 | **150** | 376 | **155** |
| generate, 8 concurrent | 2,548 | 1,090 | 2,421 | 1,102 | 2,405 | **1,148** |
| capture 1 layer, every step | · | 135 | 366 | **143** | 370 | 135 |
| capture every layer, every step | · | **119** | **361** | 102 | 295 | 96 |
| capture 1 layer, 8 concurrent | · | 940 | 2,224 | **1,032** | **2,339** | 882 |
| additive steering, 1 layer | · | 134 | 365 | **145** | 370 | 116 |
| logit lens every step | · | **134** | **325** | 90 | 145 | 109 |
| linear probe every step | · | 134 | 364 | **143** | 367 | 112 |
| zero one attention head every step | · | 135 | 367 | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 136 | 365 | ✗ | ✗ | ✗ |
