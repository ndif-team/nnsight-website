# Steering

A write is Python on the live tensor. There is no spec object: the operation is whatever you
write, at whatever location, under whatever condition. Edits to `.output` land in the running
forward; the next module reads the edited value.

Two rules that hold on every engine:

- **Edit in place** (`x[:] += v`, `x[:, i] = 0`). That works on the eager engine and is the only
  form that lands under [graph taps](performance.md). A replacement (`layer.output = new`) works
  eagerly and is copied back, shape-checked, under taps.
- **Put the edit under `tracer.iter` / `tracer.all()`** to keep it on while the model writes. A bare
  edit in the block fires once, on the prefill — which still changes the output, so it is easy to
  mistake for working.

## Without steering

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0, max_tokens=16) as tracer:
    out = tracer.result.save()

print(out.outputs[0].text)
# Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
```

## Additive

`h -> h + scale * vector`, at the residual stream leaving block 10. `layers[i].output` is
`(hidden, residual)` and the next block sums them, so adding to either element adds to the stream.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
torch.manual_seed(0)
vector = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
vector = vector / vector.norm()

with model.trace("The capital of France is", temperature=0.0, max_tokens=16) as tracer:
    for _ in tracer.iter[:16]:
        model.model.layers[10].output[0][:] += 60.0 * vector
    out = tracer.result.save()

print(out.outputs[0].text)
# a city called Paris, and the capital of Belgium is a city called Brussels.
```

A useful `scale` is a multiple of the residual's own norm at that layer; read it once with
[Capture](capture.md) and scale from there.

## Orthogonal decomposition

`h -> (I - P)h + coeff * P h`, where `P` projects onto `vector`. `coeff=0` ablates the direction;
`coeff=3` triples it.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
torch.manual_seed(0)
u = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
u = u / u.norm()
coeff = 0.0

with model.trace("The capital of France is", temperature=0.0, max_tokens=16) as tracer:
    for _ in tracer.iter[:16]:
        h = model.model.layers[10].output[0]
        proj = (h @ u).unsqueeze(-1) * u                 # P h
        h[:] = h - proj + coeff * proj
    out = tracer.result.save()

print(out.outputs[0].text)
# Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
```

Unchanged: a *random* direction carries nothing the model uses, so removing it costs nothing.
A direction that matters — a probe's, a difference of means — is the interesting case, and this
is the operation to test it with.

## Projection cap

Clamp the projection onto `vector` into `[lo, hi]`, leaving the orthogonal part alone.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
torch.manual_seed(0)
u = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
u = u / u.norm()

with model.trace("The capital of France is", temperature=0.0, max_tokens=16) as tracer:
    for _ in tracer.iter[:16]:
        h = model.model.layers[10].output[0]
        coef = (h @ u).unsqueeze(-1)
        h[:] += (coef.clamp(None, 2.0) - coef) * u
    out = tracer.result.save()

print(out.outputs[0].text)
# Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
```

## Many layers, many operations

Operations are lines of Python; order is program order.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
torch.manual_seed(0)
a, b = torch.randn(2, 4096, dtype=torch.bfloat16, device="cuda")
a, b = a / a.norm(), b / b.norm()

with model.trace("The capital of France is", temperature=0.0, max_tokens=16) as tracer:
    for _ in tracer.iter[:16]:
        model.model.layers[10].output[0][:] += 30.0 * a
        h = model.model.layers[20].output[0]
        h[:] -= (h @ b).unsqueeze(-1) * b                # ablate b at 20
        h[:] += 15.0 * a
    out = tracer.result.save()

print(out.outputs[0].text)
# a city that is also the name of a famous French wine. What is it
```

## Which positions

Rows are positions. On the prefill step the tensor holds the whole prompt; on each decode step it
holds one row, the running token. So `[-1]` is "the newest token" on every step, and an absolute
index only means something on the prefill.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
prompt = "The capital of France is"
ids = model.tokenizer(prompt)["input_ids"]
special = torch.tensor([i in model.tokenizer.all_special_ids for i in ids])
torch.manual_seed(0)
vector = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
vector = vector / vector.norm()

with model.trace(prompt, temperature=0.0, max_tokens=16) as tracer:
    h = model.model.layers[10].output[0]
    h[~special.to(h.device)] += 60.0 * vector       # prefill: content tokens only
    for _ in tracer.iter[1:16]:
        model.model.layers[10].output[0][:] += 60.0 * vector   # decode: the new token
    out = tracer.result.save()

print(out.outputs[0].text)
# a city called Paris, and the capital of Belgium is a city called Brussels.
```

## Somewhere other than the residual

Any location. Zero one attention head's output before `W_O` — the input of `o_proj` is
`[pos, n_heads * head_dim]`, head `h` is columns `h*128:(h+1)*128`:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
head = 7

with model.trace("The capital of France is", temperature=0.0, max_tokens=16) as tracer:
    for _ in tracer.iter[:16]:
        model.model.layers[10].self_attn.o_proj.input[:, head * 128:(head + 1) * 128] = 0
    out = tracer.result.save()

print(out.outputs[0].text)
# Paris. The capital of France is Paris. The capital of France is Paris.
```

The same slice works under tensor parallelism, whichever rank holds the head — see
[Tensor parallelism](tensor-parallel.md).

### An expert

On a mixture-of-experts model the individual experts are fused into one kernel, so the write goes
to the router. Run on `Qwen/Qwen1.5-MoE-A2.7B`:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen1.5-MoE-A2.7B", dispatch=True)
expert = 3

with model.trace("The capital of France is", temperature=0.0, max_tokens=8) as tracer:
    for _ in tracer.iter[:8]:
        model.model.layers[10].mlp.gate.output[0][:, expert] = float("-inf")
    out = tracer.result.save()

print(out.outputs[0].text)
# ______.
# Paris
# London
# Berlin
```

## The sampler

Two locations sit after the model: `model.logits` (what the sampler is about to draw from) and
`model.samples` (what it drew). Both are writable.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
paris = model.tokenizer.encode(" Paris")[0]

with model.trace("The capital of France is", temperature=0.0, max_tokens=6) as tracer:
    for step in tracer.iter[:6]:
        model.logits[:, paris] = float("-inf")           # never Paris
        if step == 3:
            model.samples[:] = model.tokenizer.encode(" Berlin")[0]   # force one token
    out = tracer.result.save()

print(out.outputs[0].text)
# a city in Berlin, Germany
```

An edited `samples` is the token the engine continues from, so the next step's prompt has it.

## Every request, not just yours

`model.edit()` installs a block on the engine, to run on every request it serves from then on —
including requests from an OpenAI-compatible client that has never heard of nnsight. See
[Async and servers](serving.md#edit-the-engine).
