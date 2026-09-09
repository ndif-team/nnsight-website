# Editing the engine

A trace carries its block on the one request it rides. `model.edit()` puts a block on the
**engine** instead: it is sent once, and every request the engine runs from then on gets its own
copy — requests you trace, plain `generate` calls, and requests from other clients of a served
engine that have never heard of nnsight. What each copy saves comes back on *that request's*
output. This page is the whole behaviour: installing, reading, following generation, choosing
edits per request, and how it composes with invokes, plain calls, the async engine and
`nnsight-serve`.

## Install and read

The block is written like a trace body with no `invoke` (it belongs to no request). Build the
engine with `enable_prefix_caching=False` — see [why](#prefix-caching-must-be-off).

```python
import nnsight
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True, enable_prefix_caching=False)

with model.edit() as (tracer, edit):
    out = model.model.layers[16].output
    resid = (out[0] + out[1])[-1].clone().save()          # residual stream, last position

outputs = model.generate(["The capital of France is", "Water boils at"],
                         temperature=0.0, max_tokens=3)      # plain requests, no trace
for o in outputs:
    print(repr(o.outputs[0].text), tuple(o.saves["resid"].shape), round(o.saves["resid"].norm().item(), 1))
# ' Paris. The' (4096,) 88.0
# ' 10' (4096,) 102.5
```

Each output carries its own value: the block ran once per request, in that request's scope. A
traced request gets the same copy; read the edit's values through `tracer.result.saves` — the
trace's own saves come back as your variables, the edit's ride the result:

```python
with model.trace("The capital of Peru is", temperature=0.0, max_tokens=3) as tracer:
    mine = model.logits.argmax(-1).item().save()
    result = tracer.result.save()

print(model.tokenizer.decode(mine), result.saves["resid"].shape)
print(sorted(result.saves), hasattr(result, "nnsight_saves"))
#  Lima torch.Size([4096])
# ['resid'] False
```

If a trace saves a name the edit also saves, `tracer.result.saves[name]` is the edit's and the
variable is the trace's; an output from a plain `generate` holds both on `output.saves` with the
trace's winning, and the trace's alone on `output.nnsight_saves`. Different names avoid the
question.

## Following generation

An edit sees a request's prefill unless it iterates: `tracer.all()` inside the block follows every
generated step, so a per-step readout accumulates on the request.

```python
edit.clear()

with model.edit() as (tracer, edit):
    norms = nnsight.save([])
    for step in tracer.all():
        norms.append(model.model.layers[16].output[1][-1].float().norm().item())

out = model.generate("The three primary colors are", temperature=0.0, max_tokens=5)[0]
print(repr(out.outputs[0].text), [round(n) for n in out.saves["norms"]])
# ' red, yellow, and' [78, 92, 75, 88, 80]
edit.clear()
```

`tracer.iter[:N]` works too. Anything the copy has not finished when its request ends — a read of
a location the request never reached — is unwound quietly, and what it did save is kept.

## Named edits: choosing per request

`model.edit(name="probe")` tags an edit. A request then says which installed edits it wants with
`edits=[...]`. The rule:

- no `edits=` → every installed edit runs (the default, above);
- `edits=["probe"]` → the edits installed under the names listed, **plus every unnamed edit**;
- `edits=[]` → the unnamed edits only.

```python
with model.edit(name="probe") as (tracer, probe):
    score = model.model.layers[20].output[1][-1].float().norm().item().save()

with model.edit(name="steer") as (tracer, steer):
    model.model.layers[8].output[0][:] += 60.0                  # crude, visible

with model.edit() as (tracer, always):                          # unnamed
    seen = nnsight.save(True)

prompt = "The capital of France is"
for edits in (None, ["probe"], ["steer"], []):
    kw = {} if edits is None else {"edits": edits}
    o = model.generate(prompt, temperature=0.0, max_tokens=4, **kw)[0]
    print(f"{str(edits):10s} ran={str(sorted(o.saves)):28s} text={o.outputs[0].text!r}")
# None       ran=['score', 'seen']            text='异地dis- -'
# ['probe']  ran=['score', 'seen']            text=' Paris. The capital'
# ['steer']  ran=['seen']                     text='异地dis- -'
# []         ran=['seen']                     text=' Paris. The capital'
```

A name is a tag, not a key: two edits installed under one name both run when it is asked for.
`edit.name` reads it back; `repr(probe)` shows it.

`edits=` goes wherever sampling settings go, and a request can only ask for what is installed:

```python
with model.trace(temperature=0.0, max_tokens=2, edits=["probe"]) as tracer:
    with tracer.invoke("The capital of Italy is"):               # the trace's choice
        first = tracer.result.save()
    with tracer.invoke("The capital of Spain is", edits=[]):     # this invoke's own wins
        second = tracer.result.save()
print(sorted(first.saves), sorted(second.saves))
# ['score', 'seen'] ['seen']

try:
    model.generate(prompt, max_tokens=1, edits=["nope"])
except ValueError as e:
    print(type(e).__name__, str(e)[:96])
# ValueError edits=['nope'] names ['nope'], but no edit is installed under that name (installed: ['probe', 's

try:
    model.generate(prompt, max_tokens=1, edits="probe")
except TypeError as e:
    print(type(e).__name__, str(e))
# TypeError edits= takes a list of edit names, not a string; write edits=['probe']
```

On a local engine an unknown name is refused at the call; on a served engine (whose edits were
installed over HTTP) it comes back as the request's error. `edits=` is not a sampling parameter —
it rides the request beside the block, so a served engine, the async engine and a plain call all
read it the same way.

## With invokes

Every invoke is its own request, so it gets its own copy of every edit it runs, and its own values:

```python
with model.trace(temperature=0.0, max_tokens=1, edits=["probe"]) as tracer:
    for p in ["Paris is in", "Tokyo is in", "Lima is in"]:
        with tracer.invoke(p):
            r = tracer.result.save()

print([round(x.saves["score"]) for x in r])
# [100, 106, 102]
model.clear_edits()
print([getattr(o, "saves", None) for o in model.generate(prompt, max_tokens=1)])
# [None]
```

After `clear()` (or `model.clear_edits()`, which clears every one) an output has **no** `.saves`
attribute at all — read it with `getattr(output, "saves", {})` when an edit may be gone.

## On the async engine

The install is a collective RPC, which on `mode="async"` can only be awaited from inside the
running loop: `async with model.edit()`, `await edit.aclear()`, `await model.aclear_edits()`. A
plain `with` raises there rather than silently not installing. Values arrive on the awaited
outputs, and `edits=` works unchanged.

```python
import asyncio

amodel = VLLM("Qwen/Qwen3-8B", dispatch=True, mode="async", enable_prefix_caching=False)

async def main():
    async with amodel.edit(name="probe") as (tracer, probe):
        score = amodel.model.layers[20].output[1][-1].float().norm().item().save()
    async with amodel.edit() as (tracer, always):
        seen = nnsight.save(True)

    outputs = await amodel.generate(["The capital of France is", "Water boils at"],
                                    temperature=0.0, max_tokens=2, edits=[])
    print([sorted(o.saves) for o in outputs])
    # [['seen'], ['seen']]
    outputs = await amodel.generate("The capital of France is", temperature=0.0, max_tokens=2,
                                    edits=["probe"])
    print([sorted(o.saves) for o in outputs])
    # [['score', 'seen']]

    with amodel.trace("The capital of Japan is", temperature=0.0, max_tokens=2, edits=["probe"]) as tracer:
        result = tracer.result.save()
    async for output in tracer.backend:
        if output.finished:
            print(output.outputs[0].text, sorted(output.saves))
    #  Tokyo. ['result', 'score', 'seen']
    await amodel.aclear_edits()

asyncio.run(main())
```

A streamed request's saves — the edit's included — are attached only to its **finished** output,
where `output.saves` holds the edits' values and the trace's own (`result` above) together.

## Over `nnsight-serve`

A GPU-less client installs an edit on a server's engine with `model.edit(serve=url, name=...)`,
and a served trace chooses with `edits=` exactly as above. The server is
`nnsight-serve Qwen/Qwen3-8B --port 6677 --enable-prefix-caching False` here.

```python
URL = "http://127.0.0.1:6677"
client = VLLM("Qwen/Qwen3-8B")                                  # meta tree, never dispatched

with client.edit(serve=URL, name="probe") as (tracer, probe):
    score = client.model.layers[20].output[1][-1].float().norm().item().save()
with client.edit(serve=URL) as (tracer, always):
    seen = nnsight.save(True)

for edits in (None, ["probe"], []):
    kw = {} if edits is None else {"edits": edits}
    with client.trace("The capital of France is", serve=URL, temperature=0.0, max_tokens=2, **kw) as tracer:
        result = tracer.result.save()
    print(f"{str(edits):10s} {sorted(result.saves)}")
# None       ['score', 'seen']
# ['probe']  ['score', 'seen']
# []         ['seen']

try:
    with client.trace("The capital of France is", serve=URL, max_tokens=1, edits=["nope"]) as tracer:
        result = tracer.result.save()
except RuntimeError as e:                                       # the request's error, re-raised
    print(type(e).__name__, str(e)[:72])
# RuntimeError ValueError: edits=['nope'] names ['nope'], but no edit is installed unde

probe.clear(); always.clear()
```

Every request the server handles is a trace (it has no OpenAI routes), so "another client's
request" means another client's trace: it runs the edits it asks for, or all of them if it asks
for none. Edits installed over HTTP are not in the client's `model._installed_edits`, which is
why the name check there is the worker's.

## Prefix caching must be off

A prefix-cached token is served from the KV cache without a forward pass, so no hook fires for it
and an installed block sees a short activation with no error. A trace asks for its own request to
be recomputed; an edit rides requests it did not create and cannot. Build with
`enable_prefix_caching=False` — `nnsight-serve ... --enable-prefix-caching False` — and editing an
engine that has it on warns.

## Errors and cost

An error inside an installed block is that request's: it comes back on the output's
`nnsight_error` (re-raised by a trace's collect) and the engine keeps serving. Nothing accumulates
on the worker — values are handed over as each request is collected.

An edit is the cheap shape for a sweep: the block is serialized once, and each request pays only
its own copy. A per-request trace pays a serialization per invoke, and any reference to `model`
inside its block — `model.logits`, `model.samples`, `tracer.result` — ships the model with every
invoke, ~9 ms each ([Performance](performance.md#what-else-moves-the-number)). Past four cards the
balance flips: an installed block runs its saves on every rank.
