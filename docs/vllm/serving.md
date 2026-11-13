# Async and serving

There are three ways to put an engine behind more than one caller.

| | Engine in your process | `nnsight-serve` | NDIF (`remote=True`) |
| --- | --- | --- | --- |
| Who holds the GPU | You | One server, one model | A shared cluster |
| Clients | This process | Any machine, no GPU | Any machine, no GPU |
| Fits | Notebooks and scripts | A team on one model | Many models and users |

## The async engine

`mode="async"` builds vLLM's `AsyncLLM`. The block is the same. When the `with` block exits,
the request is submitted and `tracer.backend` is its stream.

```python
import asyncio
from nnsight.modeling.vllm import VLLM


async def main():
    model = VLLM("Qwen/Qwen3-8B", dispatch=True, mode="async")

    with model.trace("The capital of France is", temperature=0.0, max_tokens=8) as tracer:
        stream = sum(model.model.layers[10].output).save()

    async for output in tracer.backend:
        print(repr(output.outputs[0].text), output.finished)
    print(output.saves["stream"].shape)
    # ' Paris' False
    # ' Paris.' False
    # ...
    # ' Paris. The capital of Italy is Rome' True
    # torch.Size([5, 4096])


asyncio.run(main())
```

- **Saved values are on the finished output**, under the name you saved them as. The async
  engine does not put them back in your variables.
- `await tracer.backend` waits for the stream to end and returns the last output.
- An async trace takes one prompt. For several, start several traces.
- Build the model inside the coroutine that uses it. `AsyncLLM` belongs to the event loop that
  built it.

Concurrent traces are batched by vLLM. Each block sees only its own request's rows.

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

`nnsight-serve` is a server that holds one engine and runs traces sent to it.

```bash
nnsight-serve Qwen/Qwen3-8B --port 6677 --gpu-memory-utilization 0.5
```

`--host`, `--port` and `--api-key` belong to the server. Every other `--flag value` goes to
vLLM's engine arguments. A boolean needs a value (`--enable-lora True`), only long flags are
read (`--tensor-parallel-size 2`, not `-tp 2`), and `taps` cannot be given on the command line.

A client builds the model without `dispatch=True`, which needs no GPU, and adds `serve=` to a
trace. Saved values come back into your variables as they do locally.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B")                       # the module tree only

with model.trace("The capital of France is", temperature=0.0, max_tokens=8,
                 serve="http://127.0.0.1:6677") as tracer:
    model.model.layers[10].output[0][:] = 0         # runs on the server
    stream = sum(model.model.layers[20].output).save()
    out = tracer.result.save()

print(stream.shape, repr(out.outputs[0].text))
# torch.Size([5, 4096]) ' a country, and the capital of Germany'
```

- The server returns saved values only. Save `tracer.result` to get the generated text.
- The server runs code that clients send. It listens on `127.0.0.1` by default. Expose it only
  to a network you trust, and set `--api-key` (clients then pass `api_key=`).
- It is not an OpenAI-compatible server. It serves nnsight traces and
  [edits](editing.md) (`model.edit(serve=url)`).
- `GET /health` reports when the engine is ready.

## NDIF

`VLLM` models also run on NDIF: `model.trace(..., remote=True)`.
