| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens | vanilla vLLM 0.20.2 | TransformerLens batched | TransformerLens (compile + CUDA graphs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 395 | 145 | 372 | **150** | 376 | **155** | 359 | ✗ | ✗ |
| generate, 8 concurrent | 2,548 | 1,090 | 2,421 | 1,102 | 2,405 | **1,148** | 2,353 | ✗ | ✗ |
| capture 1 layer, every step | · | 135 | 366 | **143** | 370 | 135 | · | ✗ | ✗ |
| capture every layer, every step | · | **119** | **361** | 102 | 295 | 96 | · | ✗ | ✗ |
| capture 1 layer, 8 concurrent | · | 940 | 2,224 | **1,032** | **2,339** | 882 | · | ✗ | ✗ |
| additive steering, 1 layer | · | 134 | 365 | **145** | 370 | 116 | · | ✗ | ✗ |
| logit lens every step | · | **134** | **325** | 90 | 145 | 109 | · | ✗ | ✗ |
| linear probe every step | · | 134 | 364 | **143** | 367 | 112 | · | ✗ | ✗ |
| zero one attention head every step | · | 135 | 367 | ✗ | ✗ | ✗ | · | ✗ | ✗ |
| override the sampled token every step | · | 136 | 365 | ✗ | ✗ | ✗ | · | ✗ | ✗ |
| one forward over 512 tokens, capture 1 layer | 11 ms | 19 ms | 18 ms | **14 ms** | **14 ms** | **15 ms** | 9 ms | 60 ms | 13 ms |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.22 s | 0.90 s | 1.07 s | · | · | **0.56 s** | 0.19 s | 3.73 s | 8.07 s |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 0.54 s | 0.48 s | · | · | ✗ | · | ✗ | ✗ |
