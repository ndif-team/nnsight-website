# Performance

There are two engines, and the choice between them is the main thing that sets speed.

- **The default engine** serves every location and runs at the speed of plain vLLM in eager
  mode.
- **A tapped engine** (`taps=`) serves the locations you declare and keeps CUDA-graph replay.

## Why the default engine is slower

vLLM gets much of its generation speed from replaying CUDA graphs. A replayed graph runs no
Python, so a block cannot run inside one. The default engine therefore sets
`enforce_eager=True`. That cost is vLLM's eager mode, not nnsight: with no block running, plain
`vllm.LLM(..., enforce_eager=True)` and `VLLM(...)` generate at the same rate.

An eager engine spends Python time on the CPU for every module call, so its speed depends on
the host's CPU load, and it does not rise as GPUs are added. A tapped engine waits on the GPU
and is steady.

## Measured

Llama-3.1-8B and Llama-3.1-70B in bf16 on A100-80GB GPUs, with a 512-token prompt, 128 new
tokens, greedy decoding, and prefix caching off. Each figure is the mean of nine timed runs.
These numbers were measured with vLLM 0.27.1.

### One request, one GPU

Llama-3.1-8B. The plain vLLM and tapped columns are tokens per second. The eager column is a
share of plain eager vLLM doing the same generation, because that ratio holds across hosts
and the rate does not.

| Workload | Plain vLLM | `VLLM(...)` | `VLLM(..., taps=)` |
| --- | ---: | ---: | ---: |
| Generate, no block | 92 tok/s | 1.00 | 89 tok/s |
| Generate inside an empty trace | | 0.93 | 89 |
| Read one layer, every step | | 0.92 | 89 |
| Read every layer, every step | | 0.79 | 88 |
| Steer at one layer | | 0.91 | 89 |
| Logit lens every step | | 0.91 | 84 |
| Zero one attention head every step | | 0.92 | 89 |
| Override the sampled token every step | | 0.92 | 89 |
| Read one layer, 8 concurrent requests | 618 tok/s | 0.92 | 577 tok/s |

A block costs the default engine between 7% and 21% of its speed. A tapped engine holds 84 to
89 tokens per second against plain vLLM's 92, whatever the block does.

### Tensor parallelism

Reading one layer on every step, as a share of plain vLLM with the same settings.

| | Plain vLLM | `VLLM(...)` | `VLLM(..., taps=)` |
| --- | ---: | ---: | ---: |
| 8B, 1 GPU | 92 tok/s | 85% | 96% |
| 8B, 2 GPUs | 148 | 45% | 95% |
| 8B, 4 GPUs | 229 | 29% | 93% |
| 8B, 8 GPUs | 313 | 20% | 91% |
| 70B, 4 GPUs | 37 | 75% | 97% |
| 70B, 8 GPUs | 61 | 46% | 95% |

The default engine's rate barely changes as GPUs are added, because the Python step between
modules runs on the CPU while the GPUs wait. Use the default engine on one GPU or in a
notebook, where serving every location is worth more than speed. Declare taps when you scale.

### On vLLM 0.33

A tapped engine on vLLM 0.33 uses piecewise CUDA graphs, which replay more slowly than the
full graphs plain vLLM uses. On Llama-3.2-1B with 32 concurrent requests, a tapped engine runs
at 77% of plain vLLM and the default engine at 39%. The tables above have not been measured
again on 0.33.

## What else affects speed

- **What you save.** A `[tokens, vocab]` tensor per layer is costly to return. Reduce in the
  block, for example with `topk`.
- **`model.edit()` for a sweep.** An [edit](editing.md) sends the block once, not once per
  request. Reading one layer over 1,024 prompts on the 8B model took 1.60 s traced, 1.09 s
  edited, and 0.75 s with plain vLLM. Past four GPUs a traced sweep is faster, because an
  installed block saves on every GPU's worker (8 GPUs: 1.2 s traced, 2.2 s edited).
- **Bind the module before a trace with many invokes.** Each invoke sends its block and what
  the block refers to. `model.model.layers[16]` inside the block refers to the model, which
  costs about 7 ms per invoke. `layer = model.model.layers[16]` before the trace and
  `layer.output` inside costs nothing. Over 1,024 invokes that is 8.6 s against 1.6 s.
- **The number of taps.** Each tap splits the graph.

For the same workloads on other libraries, see [Comparisons](comparisons.md#throughput).
