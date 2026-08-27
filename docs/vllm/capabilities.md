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
| `tracer.barrier(n)` | `NotImplementedError`: each invoke is its own request; the blocks never share a forward |
| `trace(prompt, temperatur=0.0)` | `TypeError` from `SamplingParams` — not silently ignored |
| an empty `tracer.invoke()` that reads or writes something | error: its work would vanish (a do-nothing empty invoke is a no-op) |
| `enable_chunked_prefill=True` and a prompt that got chunked | the request's error: prompt split across steps, so no block could see it whole |
| `.backward()`, `.grad`, `.scan()` | not available on vLLM (inference mode; no autograd graph) |
| an exception in your block (`1/0`) | re-raised in your process with the block's own traceback; the engine keeps serving |

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

A bad block errors *its* request. Other requests in the same batch, and every request after,
are unaffected — this is what makes [engine-wide edits](serving.md#edit-the-engine) and shared
servers safe to run.

## Two behaviours that differ from a HuggingFace trace

- **Tensors alias engine memory.** Clone what you keep ([Locations](locations.md#clone-what-you-keep)).
- **One prompt per invoke, no barrier.** Prompts become separate requests the scheduler batches;
  cross-prompt patching is a *saved* value from one trace used in the next, not a barrier.

## Not on vLLM

| | Where instead |
| --- | --- |
| Gradients, backward, `.scan()` | `TransformersModel` — same block |
| The attention pattern as a location | rebuilt from q/k, [Attention](attention.md) |
| Individual MoE experts as modules | mask the router logit, [Steering](steering.md#an-expert) |
| Multimodal vLLM models | text-only today |
| Pipeline parallelism (`pipeline_parallel_size > 1`) | tensor parallelism, [Tensor parallelism](tensor-parallel.md) |
| Speculative decoding | off |
| vLLM's Ray **v1** executor | the default (v2) Ray executor works, [Tensor parallelism](tensor-parallel.md#multi-node-with-ray) |
| `.source` ops inside a fused CUDA kernel | the kernel's inputs and outputs are locations; its interior is not Python |

## Versions

Tested against vLLM 0.16 through 0.27, on vLLM's V1 engine. On 0.27 nnsight selects the V1
`GPUModelRunner` (`VLLM_USE_V2_MODEL_RUNNER=0`) and refuses to come up on any other. Graph taps need a vLLM with
breakable CUDA graphs (`vllm.compilation.breakable_cudagraph`); on an older vLLM, `taps=` is
refused at construction with that message.

## How this is known

`tests/vllm/` in the nnsight repository: ~195 tests on two GPUs covering tracing, request
accounting, async, edits, tensor parallelism, taps, preemption, chunked prefill, serve, Ray, LoRA
and mixture-of-experts batching, on gpt2, Qwen2.5-0.5B, Qwen1.5-MoE, Llama-3.x and
DeepSeek-V2-Lite. Every value read in this section was checked against a HuggingFace forward of
the same checkpoint.
