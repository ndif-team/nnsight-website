# Async and servers

Three ways to put an engine behind more than one caller: the async engine in your own process, a
standalone `nnsight-serve` that GPU-less clients submit traces to, and an edit installed on the
engine that runs for every request it serves.

## The async engine

`mode="async"` builds vLLM's `AsyncLLM`. The block is written exactly as before; on exit the
request is submitted and `tracer.backend` is the stream.

```python
import asyncio
from nnsight.modeling.vllm import VLLM


async def main():
    model = VLLM("Qwen/Qwen3-8B", dispatch=True, mode="async")

    with model.trace("The capital of France is", temperature=0.0, max_tokens=8) as tracer:
        resid = sum(model.model.layers[10].output).save()

    async for output in tracer.backend:
        print(repr(output.outputs[0].text), output.finished)
    print(output.saves["resid"].shape)                 # saves ride the finished output
    # ' Paris' False
    # ' Paris.' False
    # ...
    # ' Paris. The capital of Italy is Rome' True
    # torch.Size([5, 4096])


asyncio.run(main())
```

Every yielded `RequestOutput` carries the cumulative text; only the last (`finished`) carries
`.saves`, keyed by the variable names you saved. `await tracer.backend` drains the stream and
returns that last output. Closing the stream early aborts the request. One prompt per async
trace — for many, fire many.

**One engine, one event loop.** `AsyncLLM` binds to the loop that built it, which is where it
keeps its output handler and its per-request futures. Build the model inside the coroutine you
will await it from — as `main()` does above — not at import time followed by two `asyncio.run()`
calls; the second loop never hears back from the engine. In a server, that means building it in
the app's startup (FastAPI's `lifespan`), on the loop that will serve requests.

## Concurrency

vLLM batches concurrent requests; each block sees only its own rows, and an edit in one request
never leaks into a neighbour.

```python
import asyncio
import torch
from nnsight.modeling.vllm import VLLM


async def one(model, prompt, steer):
    with model.trace(prompt, temperature=0.0, max_tokens=8) as tracer:
        for _ in tracer.iter[:8]:
            model.model.layers[10].output[0][:] += steer
    last = await tracer.backend
    return last.outputs[0].text


async def main():
    model = VLLM("Qwen/Qwen3-8B", dispatch=True, mode="async")
    torch.manual_seed(0)
    v = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    v = 60.0 * v / v.norm()
    prompts = ["The capital of France is", "The capital of Japan is", "The capital of Peru is"]
    texts = await asyncio.gather(*(one(model, p, s) for p, s in zip(prompts, [0 * v, v, 0 * v])))
    for p, t in zip(prompts, texts):
        print(f"{p!r:28} -> {t!r}")
    # 'The capital of France is'   -> ' Paris. The capital of Italy is Rome'
    # 'The capital of Japan is'    -> ' a city in the east, and the'
    # 'The capital of Peru is'     -> ' Lima, and the capital of the region'


asyncio.run(main())
```

## `nnsight-serve`

A single-model FastAPI server around a dispatched async engine. The CLI mirrors `vllm serve`;
trailing flags are vLLM engine args.

```bash
nnsight-serve Qwen/Qwen3-8B --port 6677 --gpu-memory-utilization 0.5
nnsight-serve Qwen/Qwen3-8B --port 6677 --api-key SECRET --tensor-parallel-size 2
```

It binds to `127.0.0.1` by default and executes client-supplied code, so only expose it on a
network you trust; `--api-key` requires the header on every request. `GET /health` reports
readiness. (An editable install may not register the console script — `python -m
nnsight.modeling.vllm.serve.cli` is the same thing.)

### A GPU-less client

The client builds only the meta tree, writes a trace, and adds `serve=`. Saved values are pushed
back into your variables exactly as they are locally.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B")                       # no GPU, never dispatched

with model.trace("The capital of France is", temperature=0.0, max_tokens=8,
                 serve="http://127.0.0.1:6677") as tracer:
    model.model.layers[10].output[0][:] = 0         # runs on the server
    resid = sum(model.model.layers[20].output).save()
    out = tracer.result.save()

print(resid.shape, repr(out.outputs[0].text))
# torch.Size([5, 4096]) ' a country, and the capital of Germany'
```

Save `tracer.result` if you want the generation — the server returns saved values only. A
build or runtime error comes back with its real type and traceback; a transport failure is a
`ConnectionError`. Pass `api_key="SECRET"` when the server requires one.

## Edit the engine

A trace carries its block on one request. `model.edit()` installs a block on the engine itself,
once, to run on **every** request from then on — traced or not, yours or another tenant's. Its
saves ride each request's own output.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True, enable_prefix_caching=False)

with model.edit() as (tracer, edit):
    resid = sum(model.model.layers[10].output).save()

outputs = model.generate(["The capital of France is", "The capital of Japan is"],
                         temperature=0.0, max_tokens=4)          # plain requests, no trace
print([tuple(o.saves["resid"].shape) for o in outputs])
# [(5, 4096), (5, 4096)]

with model.trace("The capital of Peru is", temperature=0.0, max_tokens=4) as tracer:
    out = tracer.result.save()
print(out.saves["resid"].shape)                                  # a traced request gets it too
# torch.Size([5, 4096])

edit.clear()                                                     # or model.clear_edits()
```

The block is written like a trace body with no `invoke`; `tracer.all()` inside it follows each
request through its generated tokens. On an async engine, `async with model.edit()` and
`await edit.aclear()`. Over the wire, `model.edit(serve=url)` installs it on a server's engine.

**Prefix caching must be off** on an engine you edit: a cached token is served without a forward,
so an installed block would see a short prompt and no error. A trace can ask for its own recompute;
an edit rides requests it did not create and cannot.

When to edit rather than trace: a sweep over many prompts (the block is serialized once, not per
request), or instrumenting traffic you do not write. When to trace: one-off experiments, and
whenever you want the values back as your own variables.

## Which to pick

| | Engine per process | `nnsight-serve` | NDIF (`remote=True`) |
| --- | --- | --- | --- |
| Who owns the GPU | you | one server, one model | a shared cluster |
| Clients | this process | any host, no GPU | any host, no GPU |
| Scheduling | vLLM's | vLLM's | queued, multi-tenant |
| Fit | notebooks, scripts | a team on one model | many models, many users |

`VLLM` is `Remotable`, so `trace(..., remote=True)` submits to NDIF like any other model.
