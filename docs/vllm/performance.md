# Performance

Two engines, one decision: serve every location eagerly, or declare the locations you need and
keep CUDA-graph replay.

## Why eager costs what it costs

vLLM's decode throughput comes largely from replaying CUDA graphs, and a replayed graph runs no
Python — so nothing can be served from inside one. The default `VLLM(...)` therefore builds the
engine with `enforce_eager=True`, where every module's forward runs as Python and any location is
reachable.

**That price is vLLM's eager mode, not nnsight.** Built back to back and asked to generate with no
trace running, plain `vllm.LLM(..., enforce_eager=True)` and a dispatched `VLLM(...)` measure the
same thing every time: 69.6 against 69.7 tok/s, 70.7 against 71.2, 52.1 against 54.1 on a loaded
host, and 39.3 against 39.0 at `tensor_parallel_size=2`. What a reachable location costs is what an
eager engine costs.

An eager engine spends a Python round trip per module call on the driver, so its throughput
follows whatever CPU the host has to spare. Llama-3.1-8B generates at 86 tok/s on a quiet
A100-80GB and at 54 on the same machine with other tenants running. A tapped engine waits on
the GPU instead and does not move: 89 tok/s on the quiet measurement, 89 on the busy one. So the
eager column below is a **ratio** to plain eager vLLM, which holds across hosts, while the
vanilla and taps columns are tokens per second, which reproduce.

## Graph taps

`taps=` names locations that are recorded *into* the graph — as breaks in vLLM's breakable CUDA
graphs — and served on every replay. Everything else is vanilla vLLM with graphs on.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM(
    "Qwen/Qwen3-8B",
    dispatch=True,
    taps=["model.layers.*.output", "model.layers.10.self_attn.o_proj.input"],
)
print(len(model.taps), model.taps[:2])
# 37 ('model.model.layers.0.output', 'model.model.layers.1.output')
```

`*` matches one path segment; the `model.` prefix is implied; a tap that names no module is
refused at construction. The same trace syntax, per-request scoping, `tracer.iter`, `model.logits`
and `model.samples` all work unchanged. What changes:

- **A tap can reach inside a forward.** `taps=["model.layers.10.self_attn.source.qkv_split_0.output"]`
  taps a [`.source` op](locations.md#inside-a-forward): the worker instruments that module's
  forward before recording the graphs, and the op is served on every replay — q after the
  projection split, k after rope, whatever the forward names. Verified on Qwen3-8B: reads equal
  the eager engine's exactly, and zeroing q through the tap changes the text identically. An op
  the forward does not have is refused at load with the list of ones it has.
- **Only taps are reachable.** A read of any other module location fails when the request ends
  with `'...' is not a tap on this engine`. Keep the set small: each tap splits the graph.
- **Edits land in place.** `x[:] += v` is exactly right. A replacement (`layer.output = t`) is
  copied back into the graph's memory and must keep its shape.
- **Clone what you keep.** The value served *is* the graph's memory, rewritten next step.
- **`torch.compile` is off** (breakable graphs keep replay, drop the compiled path — most of the
  win, not all). The switch is process-wide.
- **Hybrid and recurrent trunks replay graphs for decode only.** A gated-delta-net or Mamba
  layer runs a different computation for a prefill than for a decode step, so a full graph
  captured for one silently miscomputes the other (plain vLLM does this too, with compilation
  off). On a model vLLM reports as hybrid or attention-free (Qwen3.5, Qwen3.6, Mamba, ...), a
  tapped engine therefore pins `cudagraph_mode="FULL_DECODE_ONLY"`: prefill runs eagerly, decode
  keeps replay — which is where replay pays. Verified on Qwen3.5-0.8B: tapped generation matches
  eager exactly. Your own `compilation_config` overrides the pin.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True, taps=["model.layers.*.output"])
torch.manual_seed(0)
v = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
v = 60.0 * v / v.norm()

with model.trace("The capital of France is", temperature=0.0, max_tokens=16) as tracer:
    hs = list().save()
    for _ in tracer.iter[:16]:
        model.model.layers[10].output[0][:] += v                      # in place
        hs.append(sum(model.model.layers[20].output)[-1].clone())     # clone: graph memory
    out = tracer.result.save()

print(len(hs), hs[0].shape, repr(out.outputs[0].text))
# 16 torch.Size([4096]) ' a city called Paris, and the capital of Belgium is a city called Brussels.'
```

## Measured

Llama-3.1-8B and Llama-3.1-70B, bf16, A100-80GB, vLLM 0.27.1, 512-token prompt, 128 new tokens,
greedy, prefix caching off on every engine, mean of 9 timed runs.

**One request, one GPU (8B).** The vanilla and taps columns are tokens per second. The eager
column is a share of plain vLLM with `enforce_eager=True` doing the same generation, which is what
the eager engine matches on the first row.

| workload | vanilla vLLM | `VLLM(...)` eager | `VLLM(..., taps=)` |
| --- | ---: | ---: | ---: |
| generate | 92 tok/s | 1.00 | 89 tok/s |
| generate inside an empty trace | · | 0.93 | 89 |
| capture one layer, every step | · | 0.92 | 89 |
| capture every layer, every step | · | 0.79 | 88 |
| additive steering at one layer | · | 0.91 | 89 |
| logit lens every step, in the worker | · | 0.91 | 84 |
| zero one head every step | · | 0.92 | 89 |
| override the sampled token every step | · | 0.92 | 89 |
| 8 concurrent, capture one layer | 618 tok/s (plain) | 0.92 | 577 tok/s |

An eager engine with a block installed keeps 0.79 to 0.93 of what the same engine does with nothing
installed, so the block is cheap next to eager mode itself. A tapped engine holds 84 to 89 tok/s
against vanilla's 92 whatever the block does, including the every-layer capture that costs the
eager engine a fifth.

**Scaling with tensor parallelism** — capture one layer every step, as a share of vanilla vLLM at
the same settings.

| | vanilla | eager | taps |
| --- | ---: | ---: | ---: |
| 8B, 1 GPU | 92 | 85% | 89 (96%) |
| 8B, tp=2 | 148 | 45% | 140 (95%) |
| 8B, tp=4 | 229 | 29% | 213 (93%) |
| 8B, tp=8 | 313 | 20% | 284 (91%) |
| 70B, tp=4 | 37 | 75% | 35 (97%) |
| 70B, tp=8 | 61 | 46% | 58 (95%) |

Graph replay is what scales. An eager engine's rate barely moves as cards are added — the
per-module handoff runs on the driver while the GPUs wait — and that ceiling belongs to eager mode
rather than to the block: measured back to back at `tensor_parallel_size=2`, plain eager vLLM gives
39.3 tok/s and `VLLM(...)` 39.0. Where the eager engine is fine — one GPU, a notebook — it is the
simpler choice because every location is served; where the GPUs outnumber the driver's ability to
hand off, declare taps.

## What else moves the number

- **Prefix caching.** A traced request skips the cache (it must recompute its prompt); plain
  requests on the same engine use it. Benchmark with `enable_prefix_caching=False` everywhere or
  the plain rows are flattered.
- **What you send home.** A `[pos, vocab]` tensor per layer is expensive; a `topk` in the worker is
  not. Capturing every layer of 70B every step costs nothing measurable under taps because the
  clones stay on the worker until collection.
- **`model.edit()`** for sweeps: the block is serialized once instead of once per request.
  Capturing one layer over 1024 prompts on 8B: 1.60 s traced, 1.09 s edited, 0.75 s bare vLLM.
  Past four cards the balance flips — the installed block runs its saves on every rank — and
  the per-request trace is the faster sweep (tp=8: 1.2 s against 2.2 s).
- **Chunked prefill is off** by default; a long prompt waits for a step that fits rather than
  being split.
- **Bind the module outside a many-invoke trace.** Each invoke serializes its block plus the
  names it references. `model.model.layers[16]` *inside* the block references the model, and
  that costs about 7 ms per invoke; `layer = model.model.layers[16]` before the trace and
  `layer.output` inside references one envoy and costs nothing. Invisible in a single trace;
  over 1024 invokes it is 8.6 s against 1.6 s. Or install the block once with `model.edit()`.

The full grid, including interp-engine's backends, is on
[Compared with interp-engine](comparisons.md#throughput).
