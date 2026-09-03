# Capabilities and limits

There is no capability table to consult: any module on the tree is a location, and what the engine
cannot do it refuses with an error that names the location and the fix. What is worth knowing is
which questions to ask up front, and which errors mean what.

## Ask the model

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", taps=["model.layers.*.output"], dispatch=True)

print(model.dispatched)              # is the engine built?
# True
print(model.taps[:3], len(model.taps))   # under graphs: the only module locations served
# ('model.model.layers.0.output', 'model.model.layers.1.output', 'model.model.layers.2.output') 36
print(type(model.vllm_entrypoint).__name__)   # LLM or AsyncLLM
# LLM
print(model.model.layers[10])        # what is there to read
# Qwen3DecoderLayer(
#   (self_attn): Qwen3Attention(
#     (qkv_proj): QKVParallelLinear(in_features=4096, output_features=6144, ...)
#     ...
```

`model.taps` is empty on an eager engine, where every location is served.

## Refusals

Nothing degrades silently. Each row is a real error, reproduced on this model.

| You did | You get |
| --- | --- |
| read a location the model already ran past (layer 20, then layer 10) | `OutOfOrderError: '...layers.10.output.i0' was requested but the model already ran past it` |
| read a non-tap location on a `taps=` engine | `OutOfOrderError: '...' is not a tap on this engine, so a replayed CUDA graph never reaches it. Declare it at construction ...` |
| `taps=["model.nope.output"]` | `ValueError` at construction: names no module |
| `VLLM(..., taps=[...], enforce_eager=True)` | `ValueError`: `enforce_eager` contradicts `taps` |
| `tracer.invoke(["a", "b"])` | error: one prompt per invoke |
| `tracer.barrier(n)` | `RuntimeError: NotImplementedError: tracer.barrier(2) cannot work on vLLM ...` — each invoke is its own request; the blocks never share a forward |
| `trace(prompt, temperatur=0.0)` | `TypeError` from `SamplingParams` — not silently ignored |
| an empty `tracer.invoke()` that reads or writes something | error: its work would vanish (a do-nothing empty invoke is a no-op) |
| `enable_chunked_prefill=True` and a prompt that got chunked | the request's error: prompt split across steps, so no block could see it whole |
| `.backward()`, `.grad` | `NotImplementedError`: the forward runs under `torch.inference_mode`, so there is no autograd graph |
| `.scan()` | `NotImplementedError: scan is unavailable on vLLM: it runs the model's forward under a fake-tensor mode ... Trace a prompt and read the shapes off the activations it serves.` |
| a replacement with a different number of rows (`layer.output = t`) | `ValueError: A batched write has to keep its rows: this block owns rows 0:5 of 5, so the replacement must be (5, 4096), not (2, 4096).` |
| an exception in your block (`1/0`) | re-raised in your process with the block's own traceback; the engine keeps serving |
| `tracer.iter[:20]` on a request that made 4 steps | `OutOfOrderError: 'model.samples.i4' was never reached: the loop asked for iteration 4 of 'model.samples' and the run reached it 4 times, so the loop was cut short and nothing after it ran.` Hold the run to the count with `ignore_eos=True` or `min_tokens=N`, or loop with `tracer.all()` and put the trailing statements after the `with` block. |

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

try:
    with model.trace("The capital of France is", temperature=0.0):
        late = model.model.layers[20].output[0].clone().save()
        early = model.model.layers[10].output[0].clone().save()
except Exception as e:
    print(type(e).__name__, str(e).splitlines()[0][:90])
# RuntimeError OutOfOrderError: 'model.model.layers.10.output.i0' was requested but the model already ran

with model.trace("The capital of France is", temperature=0.0):      # the engine is fine
    ok = model.logits.argmax(-1).save()
print(model.tokenizer.decode(ok))
# Paris
```

A bad block errors *its* request. Other clients' requests in the same batch, and every request
after, are unaffected — this is what makes [engine-wide edits](serving.md#edit-the-engine) and
shared servers safe to run. The invokes of one trace are not separate that way: they are one
block, and it raises as a whole, so a sweep loses the invokes that were fine along with the one
that was not.

The case worth knowing is the last row of the table, because a patching sweep is where you meet
it. A replacement is spliced back into the rows the block owns, so it has to keep them; a donor
activation captured at a different prompt length is the usual source of a short one. Slice the
donor to the rows you are writing instead.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

try:
    with model.trace("The capital of France is", temperature=0.0, max_tokens=1):
        out = model.model.layers[10].output
        model.model.layers[10].output = (out[0][:2], out[1][:2])
except Exception as e:
    print(type(e).__name__, str(e).splitlines()[0])
# RuntimeError ValueError: A batched write has to keep its rows: this block owns rows 0:5 of 5, so the replacement must be (5, 4096), not (2, 4096).

with model.trace("The capital of France is", temperature=0.0):      # the engine is fine
    ok = model.logits.argmax(-1).save()
print(model.tokenizer.decode(ok))
# Paris
```

## Where the message is

An error inside a block comes home: it is re-raised in your process as a `RuntimeError` carrying
the original type, its message and an "Intervention traceback" pointing at your line. Because the
type rides the message rather than the class, catch `RuntimeError` and match on the text.

Two things do not come home, because they happen in vLLM's EngineCore subprocess:

- **Warnings.** A `warnings.catch_warnings()` around a trace records nothing; the text is in the
  engine's own output, prefixed `(EngineCore pid=...)`. This is the one place the vLLM path is not
  the local one — the same block warns catchably against a HuggingFace model. What you will meet
  here is an open `tracer.iter[:]` / `tracer.all()` loop that outruns the request: it warns that
  the statements after the loop did not run, and keeps what the loop saved. A *bounded*
  `tracer.iter[:N]` that outruns the request is an error, and that one does come home.
- **Anything that fails while the engine builds**, including a bad `taps=` entry. The caller sees
  `RuntimeError: Engine core initialization failed. See root cause above.`; the message that names
  the ops a forward actually has is in the `(EngineCore pid=...)` lines above it. The same shape
  covers `AssertionError: Error in memory profiling ...`, which is a shared GPU whose free memory
  moved while vLLM profiled — build again.

`Chunked prefill is enabled with max_num_batched_tokens=...`, printed on every construction, is
not the engine you get: it comes from the meta tree built first. The engine's real arguments are
logged a few lines later as `non-default args: {... 'enable_chunked_prefill': False ...}`.

## Two behaviours that differ from a HuggingFace trace

- **Tensors alias engine memory.** Clone what you keep ([Locations](locations.md#clone-what-you-keep)).
- **One prompt per invoke, no barrier.** Prompts become separate requests the scheduler batches;
  cross-prompt patching is a *saved* value from one trace used in the next, not a barrier.
- **Numbers depend on the batch a request was scheduled with.** Reduction order inside a fused
  kernel follows the batch, and in bf16 that moves the last digit, so a sweep on this page
  reproduces to about a part in a hundred rather than exactly, and a nearly-tied greedy argmax
  can land differently between runs. `temperature=0.0` pins the sampler, not the arithmetic.
  Read paired differences taken inside one trace rather than differencing two runs.

## Not on vLLM

| | Where instead |
| --- | --- |
| Gradients and backward | `TransformersModel` — same block |
| `.scan()` | `TransformersModel`; on vLLM, trace a prompt and read the shapes off what it serves |
| The attention pattern as a location | rebuilt from q/k, [Attention](attention.md) |
| Individual MoE experts as modules | mask the router logit, [Steering](steering.md#an-expert) |
| Multimodal vLLM models | text-only today |
| Pipeline parallelism (`pipeline_parallel_size > 1`) | tensor parallelism, [Tensor parallelism](tensor-parallel.md) |
| Speculative decoding | off |
| vLLM's Ray **v1** executor | the default (v2) Ray executor works, [Tensor parallelism](tensor-parallel.md#multi-node-with-ray) |
| `.source` ops inside a fused CUDA kernel | the kernel's inputs and outputs are locations; its interior is not Python |

## Versions

nnsight targets vLLM's V1 engine and imports its internals directly, so the release matters. On
0.27 it selects the V1 `GPUModelRunner` (`VLLM_USE_V2_MODEL_RUNNER=0`) and refuses to come up on
any other. Graph taps need a vLLM with breakable CUDA graphs
(`vllm.compilation.breakable_cudagraph`); without it, `taps=` is refused at construction with that
message. The `vllm` extra carries no upper bound, so `pip install "nnsight[vllm]"` takes the
current release, and one that moves an imported name fails at `import nnsight.modeling.vllm`.
Everything in this section was run on 0.27.1.

## How this is known

`tests/vllm/` in the nnsight repository: ~195 tests on two GPUs covering tracing, request
accounting, async, edits, tensor parallelism, taps, preemption, chunked prefill, serve, Ray, LoRA
and mixture-of-experts batching, on gpt2, Qwen2.5-0.5B, Qwen1.5-MoE, Llama-3.x and
DeepSeek-V2-Lite. Every value read in this section was checked against a HuggingFace forward of
the same checkpoint.
