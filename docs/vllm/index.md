# nnsight on vLLM

`VLLM("repo")` is an nnsight model whose forward pass is a [vLLM](https://github.com/vllm-project/vllm)
engine. You write the same `with model.trace(...)` block you would write against a HuggingFace
model; nnsight serializes it onto the request, runs it *inside* vLLM's worker interleaved with the
forward, and hands the saved values back on the finished output. Every module in the model is
reachable — `.input`, `.output`, `.source` ops, the pre-sampling `model.logits`, the drawn
`model.samples` — and an edit lands in the running model. You keep PagedAttention, continuous
batching, tensor parallelism and, with [taps](performance.md), CUDA-graph replay.

Every snippet in this section was run against `Qwen/Qwen3-8B` on an A100 with vLLM 0.27.1;
the outputs shown are what came back.

## Install

```bash
pip install "nnsight[vllm]"     # nnsight + vLLM, CUDA required
pip install "nnsight[serve]"    # also the nnsight-serve server (FastAPI, uvicorn)
```

## Read one activation

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    resid = model.model.layers[10].output[0].clone().save()   # [pos, d_model]
    logits = model.logits.save()                              # [1, vocab]

print(resid.shape, model.tokenizer.decode(logits.argmax(-1)))
# torch.Size([5, 4096])  Paris
```

Three things differ from a HuggingFace trace, and every page below leans on them:

- **No batch axis.** vLLM packs every in-flight request's tokens into one `[total_tokens, hidden]`
  slab; nnsight narrows your block to its own request's rows, so you see `[pos, d_model]`.
- **A decoder layer returns a tuple.** `layers[i].output` is `(hidden_states, residual)`, and the
  residual stream leaving the block is their **sum**. See [Locations](locations.md).
- **Clone what you keep.** vLLM reuses and overwrites activation buffers in place; `.clone()` before
  `.save()` or the value you get back may belong to a later layer.

## Steer one layer

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
torch.manual_seed(0)
vector = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
vector = vector / vector.norm()

with model.trace("The capital of France is", temperature=0.0, max_tokens=16) as tracer:
    for _ in tracer.iter[:16]:                              # every step, prompt included
        model.model.layers[10].output[0][:] += 60.0 * vector
    out = tracer.result.save()

print(out.outputs[0].text)
# a city called Paris, and the capital of Belgium is a city called Brussels.
```

## Where to go

| Page | For |
| --- | --- |
| [Locations](locations.md) | What can be asked for, and how it is named on a vLLM module tree |
| [Loading models](loading.md) | Eager, CUDA-graph and async engines; what nnsight forces and why |
| [Capabilities and limits](capabilities.md) | What refuses, what is missing, how errors surface |
| [Editing the engine](editing.md) | `model.edit()`: install a block once, choose edits per request with `edits=`, invokes, async, serve |
| [Capture](capture.md) | Read activations: one location, every layer, every step, many prompts |
| [Attention](attention.md) | q/k/v, per-head outputs, the attention pattern the paged kernel never forms |
| [Logit lens](logit-lens.md) | Send a residual through the unembed, inside the worker |
| [Steering](steering.md) | Add, ablate, project, cap, mask positions, write anywhere |
| [Activation patching](patching.md) | Clean into corrupt: layer × position sweeps and one head, batched as invokes |
| [Ablation](ablation.md) | Zero every MLP, every head, in one trace |
| [Direct logit attribution](attribution.md) | What each layer and head wrote toward the answer |
| [Conditional interventions](conditional.md) | Decide in the worker, every step: gate, replace, stop |
| [SAE features](sae.md) | Find features with the full SAE once; clamp them in the worker with two rows |
| [Generation](generation.md) | Sampling, per-step logits and samples, the finished request, `n > 1` |
| [Chat and tokens](chat.md) | Templates, token-id prompts, spans |
| [Async and servers](serving.md) | `mode="async"`, concurrency, `nnsight-serve`, GPU-less clients, engine-wide edits |
| [Tensor parallelism](tensor-parallel.md) | Sharded models read as whole tensors |
| [Examples](examples/causal-tracing.md) | Causal tracing, the Jacobian lens, a linear probe, concept directions — end to end |
| [Performance](performance.md) | Eager vs graph taps, measured |
| [Comparisons](comparisons.md) | The same jobs in interp-engine and vLLM-Lens, with one throughput grid |

The longer, narrative version of all this is the [vLLM Support](../features/16_vllm_support.ipynb)
notebook; the design is written up in
[NNsight × vLLM: Interpretability at Production Scale](../blog/posts/vllm-integration.md).
