| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens | vanilla vLLM 0.20.2 | TransformerLens (compile + CUDA graphs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 148 | 72 | 142 | 71 | 142 | **75** | 149 | ✗ |
| generate, 8 concurrent | 933 | 525 | 898 | 520 | 893 | 537 | 941 | ✗ |
| capture 1 layer, every step | · | 67 | 140 | **71** | 139 | 67 | · | ✗ |
| capture every layer, every step | · | **59** | **138** | 36 | 65 | 35 | · | ✗ |
| capture 1 layer, 8 concurrent | · | 462 | 851 | **507** | 859 | 424 | · | ✗ |
| additive steering, 1 layer | · | 66 | 139 | **72** | 139 | 53 | · | ✗ |
| logit lens every step | · | **66** | **134** | 43 | 59 | 51 | · | ✗ |
| linear probe every step | · | 66 | 138 | **72** | 139 | 53 | · | ✗ |
| zero one attention head every step | · | 66 | ✗ | ✗ | ✗ | ✗ | · | ✗ |
| override the sampled token every step | · | 67 | 140 | ✗ | ✗ | ✗ | · | ✗ |
| one forward over 512 tokens, capture 1 layer | 30 ms | 45 ms | 46 ms | **37 ms** | **38 ms** | **36 ms** | 25 ms | 74 ms |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.49 s | 1.33 s | 1.42 s | · | · | **1.17 s** | 0.49 s | 16.93 s |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 1.05 s | 1.05 s | · | · | ✗ | · | ✗ |
