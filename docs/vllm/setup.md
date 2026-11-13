# Choosing a setup

There is one class, `VLLM`. Its constructor arguments decide which engine you get.

| You want to | Build | Details |
| --- | --- | --- |
| Explore a model, reading and writing anywhere | `VLLM(repo, dispatch=True)` | [The default engine](#the-default-engine) |
| Keep vLLM's full speed, or scale across GPUs | `VLLM(repo, taps=[...])` | [CUDA graphs with taps](#cuda-graphs-with-taps) |
| Run a model that does not fit on one GPU | `VLLM(repo, tensor_parallel_size=N)` | [Tensor parallelism](tensor-parallel.md) |
| Stream tokens, or serve concurrent callers in your process | `VLLM(repo, mode="async")` | [Async and serving](serving.md) |
| Share one model with clients that have no GPU | `nnsight-serve` and `trace(..., serve=url)` | [Async and serving](serving.md#nnsight-serve) |
| Run one block over many prompts, or on traffic you do not send | `model.edit()` | [Engine-wide edits](editing.md) |
| Run with a LoRA adapter | `VLLM(repo, enable_lora=True)` | [LoRA adapters](#lora-adapters) |
| Take gradients, call `.scan()`, or pass images | `TransformersModel(repo)` | Not available on vLLM |

These combine: a tapped engine can be tensor-parallel and async.

## The default engine

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
print(model.dispatched, type(model.vllm_entrypoint).__name__)
# True LLM
```

Every module is reachable. The cost is speed: nnsight builds this engine with
`enforce_eager=True`, because a replayed CUDA graph runs no Python and so cannot serve a block.
It runs at the speed of plain vLLM in eager mode ([Performance](performance.md)).

`dispatch=True` builds the engine in the constructor. Without it, the constructor builds only a
copy of the module tree with no weights, and the engine is built on the first trace. The tree is
enough to look at the model, and to write a trace for a [server](serving.md#nnsight-serve).

Other keyword arguments go to vLLM unchanged:

<!-- norun -->
```python
model = VLLM(
    "Qwen/Qwen3-8B",
    dispatch=True,
    gpu_memory_utilization=0.5,     # share of the GPU vLLM may take; default 0.9
    max_model_len=4096,
    dtype="bfloat16",
)
```

## CUDA graphs with taps

`taps=` names the locations you need. nnsight records them into vLLM's CUDA graphs, and the
engine keeps graph replay.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM(
    "Qwen/Qwen3-8B",
    dispatch=True,
    taps=["model.layers.*.output", "model.layers.10.mlp.input"],
)
print(len(model.taps), model.taps[:2])
# 37 ('model.model.layers.0.output', 'model.model.layers.1.output')
```

`*` matches one path segment. A tap can also name an operation inside a module's forward
(`"model.layers.10.self_attn.source.qkv_split_0.output"`). The trace syntax is unchanged. What
changes:

- **Only taps are served.** A read of any other module fails when the request ends, with
  `'...' is not a tap on this engine`. `model.logits`, `model.samples` and `tracer.result` are
  always served.
- **Edit in place.** `x[:] += v` lands. A replacement (`layer.output = t`) is copied back and must
  keep its shape.
- **Clone what you keep.** A tapped value is the graph's own memory, rewritten on the next step.
- **Keep the set small.** Each tap splits the graph.
- **`torch.compile` is off** for the whole process, and the graphs are piecewise.

## LoRA adapters

Enable LoRA on the engine and pass a `LoRARequest` with the trace or with one invoke.

<!-- norun -->
```python
from vllm.lora.request import LoRARequest
from nnsight.modeling.vllm import VLLM

model = VLLM("meta-llama/Llama-3.2-1B", dispatch=True, enable_lora=True, max_lora_rank=8)
adapter = LoRARequest(lora_name="mine", lora_int_id=1, lora_path="/path/to/adapter")
model.vllm_entrypoint.llm_engine.add_lora(adapter)   # load it now
prompt = "The capital of France is"

with model.trace(temperature=0.0, max_tokens=1, lora_request=adapter) as tracer:
    with tracer.invoke(prompt):                       # uses the trace's adapter
        adapted = model.logits.save()
    with tracer.invoke(prompt, lora_request=None):    # the base model
        base = model.logits.save()
```

- A wrapped layer's `.output` includes the adapter's contribution.
- Raise `max_lora_rank` to your adapter's rank, and `max_loras` to the number of adapters in one
  batch.
- Load an adapter with `add_lora` first. An adapter that fails to load inside a request stops
  the engine (`EngineDeadError`), while `add_lora` raises an ordinary exception.

## What nnsight sets on the engine

| Setting | Value | Why |
| --- | --- | --- |
| `enforce_eager` | `True` unless you pass `taps` | A replayed graph runs no Python. Passing a value that contradicts `taps` is refused. |
| `enable_chunked_prefill` | `False` | A block must see its prompt in one step. If you turn it on, a request whose prompt is split returns an error. |
| `enable_prefix_caching` | `False` | A cached token has no forward pass, so nothing runs for it. Leave it off on an engine you [edit](editing.md). |
| CUDA graph mode, with `taps` | piecewise | A tap is a break in a piecewise graph. Asking for full graphs is refused. |
| Worker class | nnsight's | It runs the block. |

## In a script

A script that builds an engine needs a `__main__` guard. vLLM starts its worker with `spawn`,
which imports your file again. Without the guard, that import tries to build another engine
and the engine fails to start. Notebooks do not need it.

<!-- norun -->
```python
from nnsight.modeling.vllm import VLLM


def main():
    model = VLLM("Qwen/Qwen3-8B", dispatch=True)
    with model.trace("The capital of France is", temperature=0.0):
        logits = model.logits.save()
    print(model.tokenizer.decode(logits.argmax(-1)))


if __name__ == "__main__":
    main()
```

An engine lives until its process exits. There is no `shutdown()`.

## Versions

nnsight targets vLLM 0.33, and the `vllm` extra pins `vllm>=0.33,<0.34`. It subclasses vLLM's
model runner, which changes between minor releases, so another vLLM version is untested.
