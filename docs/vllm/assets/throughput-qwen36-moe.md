| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate | 182 | 15 | **119** | **16** | 108 | **15** |
| generate, 8 concurrent | 732 | 112 | **545** | **115** | 491 | **117** |
| capture 1 layer, every step | · | 14 | **116** | **16** | 107 | 13 |
| capture every layer, every step | · | **14** | **114** | 13 | 74 | 12 |
| capture 1 layer, 8 concurrent | · | 105 | 487 | **116** | 492 | 95 |
| additive steering, 1 layer | · | 14 | **115** | **16** | 107 | 12 |
| logit lens every step | · | **14** | **111** | 12 | 38 | 12 |
| linear probe every step | · | 14 | **115** | **15** | 106 | 12 |
| zero one attention head every step | · | 14 | ✗ | ✗ | ✗ | ✗ |
| override the sampled token every step | · | 14 | 116 | ✗ | ✗ | ✗ |
| sweep: 1024 × 1 token, capture 1 layer, per request | 0.82 s | 2.27 s | 2.11 s | · | · | **1.62 s** |
| sweep: 1024 × 1 token, capture 1 layer, edit() once | · | 1.59 s | 1.52 s | · | · | ✗ |
