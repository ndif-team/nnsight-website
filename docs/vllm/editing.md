# Engine-wide edits

A trace sends its block with one request. `model.edit()` installs a block on the engine, and
the block then runs on every request the engine serves. Use it when:

- **You run one block over many prompts.** The block is sent once, not once per prompt.
- **You do not send the requests.** The block runs for every client of the engine.

Keep tracing for a single experiment, and when you want the values back in your own variables.

## Install and read

Write the block like a trace body, without `invoke`. Each request gets its own copy, and what
the block saves comes back on that request's output as `.saves`.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.edit() as (tracer, edit):
    stream = sum(model.model.layers[10].output).save()

outputs = model.generate(["The capital of France is", "Tokyo is the capital of"],
                         temperature=0.0, max_tokens=4)              # plain requests
print([(o.outputs[0].text, tuple(o.saves["stream"].shape)) for o in outputs])
# [(' Paris. The capital', (5, 4096)), (' Japan. It is', (6, 4096))]

edit.clear()                                                         # or model.clear_edits()
```

A traced request gets the edit too. Its values are on `tracer.result.saves`.

## Follow generation

As in a trace, a block body runs on the first step. `tracer.all()` follows each request
through its generated tokens.

```python
import nnsight
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.edit() as (tracer, edit):
    norms = nnsight.save([])
    for _ in tracer.all():
        norms.append(round(sum(model.model.layers[10].output)[-1].norm().item()))

outputs = model.generate(["The colours of the French flag are"], temperature=0.0, max_tokens=5)
print(repr(outputs[0].outputs[0].text), outputs[0].saves["norms"])
# ' blue, white, and' [62, 62, 52, 70, 54]

edit.clear()
```

## Choose edits per request

Give an edit a name, and a request can choose which named edits run with `edits=[...]`.
Unnamed edits always run.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
torch.manual_seed(0)
vector = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
vector = 60.0 * vector / vector.norm()

with model.edit(name="steer") as (tracer, steer):
    for _ in tracer.all():
        model.model.layers[10].output[0][:] += vector

prompt = ["The capital of France is"]
for edits in (None, []):
    kwargs = {} if edits is None else {"edits": edits}
    out = model.generate(prompt, temperature=0.0, max_tokens=6, **kwargs)
    print(edits, repr(out[0].outputs[0].text))
# None ' a city called Paris, and'
# [] ' Paris. The capital of Italy'

model.clear_edits()
```

`edits=` works on `trace`, on one `invoke`, and on `generate`. A name that no edit is
installed under raises `ValueError`.

## Async engines and servers

- On a `mode="async"` engine, use `async with model.edit()` and `await edit.aclear()`.
- `model.edit(serve=url)` installs the block on an [`nnsight-serve`](serving.md#nnsight-serve)
  engine.

## What to know

- **Leave prefix caching off.** nnsight builds engines with it off. A cached token has no
  forward pass, so an installed block would see a short prompt and raise nothing.
- **`edit(inplace=False)` is refused.** There is one engine, and no copy to edit.
- **Saves on every rank.** Under tensor parallelism an installed block saves on every rank, so
  past four GPUs a traced sweep is faster ([Performance](performance.md#what-else-affects-speed)).
