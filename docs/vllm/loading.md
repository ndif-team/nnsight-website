# Loading models

One class, a HuggingFace repo id, and vLLM's own engine keywords. Which engine you get is decided
by two constructor arguments.

| | `VLLM(repo)` | `VLLM(repo, taps=[...])` | `VLLM(repo, mode="async")` |
| --- | --- | --- | --- |
| vLLM engine | `LLM`, `enforce_eager=True` | `LLM`, CUDA graphs on | `AsyncLLM` |
| Locations served | every one | the declared taps, plus `logits` / `samples` / `result` | as the sync engine |
| Speed | vanilla eager | ≈ vanilla vLLM ([measured](performance.md)) | as the sync engine |
| Result | saves land in your variables | same | stream `tracer.backend`; saves on the finished output |

## Eager (default)

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
print(model.dispatched, type(model.vllm_entrypoint).__name__)
# True LLM
```

Every module location is reachable. This is the engine every page in this section uses unless it
says otherwise.

## In a `.py` file, build the engine under a `__main__` guard {#main-guard}

Every snippet in this section is written flat, the way you would type it into a notebook. Saved to
a file and run with `python`, the same code needs one wrapper:

<!-- norun -->
```python
from nnsight.modeling.vllm import VLLM


def main():
    model = VLLM("Qwen/Qwen3-8B", dispatch=True)
    with model.trace("The capital of France is", temperature=0.0, max_tokens=1):
        logits = model.logits.save()
    print(model.tokenizer.decode(logits.argmax(-1)))


if __name__ == "__main__":
    main()
```

Dispatching the engine initialises CUDA in your process, so vLLM starts its workers with `spawn`
rather than `fork` and logs why:

```
WARNING [system_utils.py:157] We must use the `spawn` multiprocessing start method.
Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. ... Reasons: CUDA is initialized
```

Spawning re-imports the main module, so a `VLLM(..., dispatch=True)` sitting at the top level of
that module runs again in the child and the engine core dies:

```
RuntimeError: An attempt has been made to start a new process before the current process has
finished its bootstrapping phase.
RuntimeError: Engine core initialization failed.
```

This holds for one GPU and `mode="sync"` as much as for eight and `mode="async"`. Notebooks have
no main module to re-import, which is why the snippets run there as written; plain `vllm.LLM` at
module level works for the same reason, since nothing has touched CUDA before the fork.

## CUDA graphs, at declared taps

CUDA-graph replay runs no Python, so an ordinary hook never fires under it. `taps=` names the
locations that get recorded *into* the graph and served on every replay; everything else runs as
vanilla vLLM.

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

`*` matches one path segment. A tap that names no module is refused at construction, and a read of
any *other* location fails when the request ends, naming the location. Edits at a tap land in place
and a kept value must be cloned — see [Performance](performance.md) for the rules and the numbers.

A tap is written against the same path `model.get(...)` takes, so it carries whatever prefix the
checkpoint's tree has. On a vision-language checkpoint the decoder sits under the language model,
and `taps=["model.layers.*.output"]` names no module: write
`taps=["language_model.model.layers.*.output"]`. `print(model)` on the meta tree settles it before
the engine is built.

A tap naming no module is caught against the meta tree and raises where you wrote it. A `.source`
op is checked later, in the worker that instruments the forward, so a name that forward does not
have surfaces as `RuntimeError: Engine core initialization failed. See root cause above.` — the
message listing the ops it *does* have is in the `(EngineCore pid=...)` lines above it.

## Async

```python
import asyncio
from nnsight.modeling.vllm import VLLM


async def main():
    model = VLLM("Qwen/Qwen3-8B", dispatch=True, mode="async")
    with model.trace("The capital of France is", temperature=0.0, max_tokens=8) as tracer:
        logits = model.logits.save()
    last = await tracer.backend          # drains the stream; saves ride the finished output
    print(last.outputs[0].text, last.saves["logits"].shape)
    # Paris. The capital of Italy is Rome torch.Size([1, 151936])


asyncio.run(main())
```

Mode is fixed at construction. Streaming, concurrency and servers are on
[Async and servers](serving.md).

## A GPU-less client

Without `dispatch=True` the constructor builds only a meta-device copy of the module tree — enough
to write and serialize a trace, no GPU touched. That is the client half of
[`nnsight-serve`](serving.md#nnsight-serve).

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B")            # no weights, no GPU
print(model.dispatched, model.model.layers[10].mlp)
# False Qwen2MLP(
#   (gate_up_proj): MergedColumnParallelLinear(in_features=4096, output_features=24576, ...)
#   (down_proj): RowParallelLinear(in_features=12288, output_features=4096, ...)
#   (act_fn): SiluAndMul()
# )
```

Any trace on a non-dispatched model without `serve=`/`remote=` dispatches it first.

## Engine keywords

Everything else goes verbatim to `vllm.LLM` / `AsyncEngineArgs`.

<!-- norun -->
```python
from nnsight.modeling.vllm import VLLM

model = VLLM(
    "Qwen/Qwen3-8B",
    dispatch=True,
    tensor_parallel_size=2,             # shard across GPUs; see Tensor parallelism
    gpu_memory_utilization=0.5,         # KV-cache budget; default 0.9
    max_model_len=4096,
    dtype="bfloat16",
    enable_prefix_caching=False,        # required for model.edit(); see below
)
```

## What nnsight forces, and why

| Setting | Value | Why |
| --- | --- | --- |
| `enforce_eager` | `True` unless `taps` | A replayed graph runs no Python, so no location can be served from one. Passing it yourself is refused if it contradicts `taps`. |
| `enable_chunked_prefill` | `False` unless you pass it | A block must see its prompt whole. A prompt that does not fit one step's budget waits a step instead of being split. If you turn chunking on, a request whose prompt *does* get chunked comes back with an error rather than a slice of its activations. The `Chunked prefill is enabled with max_num_batched_tokens=...` line printed at construction comes from the meta tree built first, not from the engine you get; the engine's own arguments are logged a few lines later as `non-default args: {...}`. |
| Prefix cache, traced requests | skipped | A cached token is served from the KV cache without a forward, so nothing fires for it. Traces ask for a recompute of their own prompt; plain requests on the same engine still hit the cache. An engine-wide [`edit()`](serving.md#edit-the-engine) cannot ask, so it needs `enable_prefix_caching=False` at construction. |
| `VLLM_USE_V2_MODEL_RUNNER` | `0` | nnsight instruments vLLM 0.27's V1 `GPUModelRunner`; the worker refuses any other runner rather than coming up uninstrumented. |
| `worker_cls` | nnsight's | Where the block runs. |

## Lifecycle

- `dispatch=True` builds the engine in the constructor; `dispatch=False` (the default) defers it to
  the first trace.
- A script needs the [`__main__` guard](#main-guard) above; a notebook does not.
- There is no `shutdown()`. An engine lives as long as its process; nnsight registers the
  distributed teardown with `atexit`.
- nnsight targets vLLM's **V1** engine and imports its internals directly, so the release matters.
  The `vllm` extra carries no upper bound: `pip install "nnsight[vllm]"` takes the current release,
  and one that has moved an imported name fails at `import nnsight.modeling.vllm`. This section was
  run on **0.27.1**.
- vLLM starts its workers with `spawn` once CUDA has been initialised in the parent; if it
  complains about forking after CUDA initialisation, set `VLLM_WORKER_MULTIPROC_METHOD=spawn`
  yourself (vLLM's requirement, not nnsight's).
