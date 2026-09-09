| workload | vanilla vLLM | nnsight eager | nnsight taps (CUDA graphs) | interp-engine vllm | interp-engine vllm-static | vLLM-Lens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generate, 32 concurrent | 1,523 | 1,434 | 1,469 | 1,430 | 1,456 | 1,455 |
| capture 1 layer, 32 concurrent | · | 1,325 | 1,366 | **1,371** | **1,408** | 1,100 |
