# nnsight on vLLM

`VLLM("repo")` is an nnsight model whose forward pass is a [vLLM](https://github.com/vllm-project/vllm)
engine. You write the same `with model.trace(...)` block as for a HuggingFace model. nnsight sends
the block with the request, runs it inside vLLM's worker as the model computes, and returns what
you saved. You keep vLLM's continuous batching, tensor parallelism and, with
[taps](setup.md#cuda-graphs-with-taps), CUDA-graph replay.

Use it for throughput over many prompts, models that need several GPUs, streamed tokens, or a
served engine. For gradients, `.scan()` or image inputs, use `TransformersModel`. The block is
the same.

## Install

```bash
pip install "nnsight[vllm]"     # nnsight and vLLM; needs a CUDA GPU
pip install "nnsight[serve]"    # adds the nnsight-serve server
```

## Read an activation

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    hidden = model.model.layers[10].output[0].clone().save()   # [tokens, hidden]
    logits = model.logits.save()                               # [1, vocab]

print(hidden.shape, model.tokenizer.decode(logits.argmax(-1)))
# torch.Size([5, 4096])  Paris
```

## Change an activation

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
torch.manual_seed(0)
vector = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
vector = vector / vector.norm()

with model.trace("The capital of France is", temperature=0.0, max_tokens=16) as tracer:
    for _ in tracer.iter[:16]:                               # every step, the prompt included
        model.model.layers[10].output[0][:] += 60.0 * vector
    out = tracer.result.save()

print(out.outputs[0].text)
# a city called Paris, and the capital of Belgium is a city called Brussels.
```

## What differs from a HuggingFace trace

- **No batch axis.** A value is `[tokens, hidden]` for your request: every prompt token on the
  first step, then one row per generated token.
- **A decoder layer returns a pair.** `layers[i].output` is `(hidden, residual)`, and the residual
  stream after the layer is their sum.
- **Clone what you keep.** A value is a view of the engine's memory, which later layers overwrite.
  Call `.clone()` before `.save()`.
- **One prompt per invoke.** Several prompts are several invokes, which vLLM batches.

[Writing a block](blocks.md) covers each of these.

## Pages

| Page | What it covers |
| --- | --- |
| [Choosing a setup](setup.md) | Which engine to build for your use case, and the settings that go with it |
| [Writing a block](blocks.md) | What a block reads and writes: shapes, logits, generation steps, several prompts |
| [Engine-wide edits](editing.md) | `model.edit()`: install a block once and run it on every request |
| [Async and serving](serving.md) | `mode="async"`, `nnsight-serve` and clients without a GPU |
| [Tensor parallelism](tensor-parallel.md) | A model sharded across GPUs |
| [Limits and errors](limits.md) | What is refused, what nothing checks, and where an error message appears |
| [Examples](examples/logit-lens.md) | Logit lens, steering, activation patching, attention patterns, a probe, the Jacobian lens |
| [Performance](performance.md) | What the eager engine and the tapped engine cost, measured |
| [Comparisons](comparisons.md) | How interp-engine, vLLM-Lens and TransformerLens differ from this |

The snippets in this section were run on `Qwen/Qwen3-8B` with vLLM 0.33 on one A100, unless a
page says otherwise. The outputs shown are what came back.
