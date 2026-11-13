# Limits and errors

Most of what the engine cannot do, it refuses with an error that names the problem. A few
things are not checked, and they are listed here too.

## Refused when you build the engine

| You pass | You get |
| --- | --- |
| A tap that names no module | `ValueError` |
| `taps=[...]` with `enforce_eager=True` | `ValueError` |
| `taps=[...]` with a `compilation_config` that asks for full CUDA graphs | The engine fails to build |
| `speculative_config=...` | `NotImplementedError` |
| `enable_batch_sharded_sampling=True` | `NotImplementedError` |
| `data_parallel_size` above 1 | `NotImplementedError` |
| `VLLM_USE_V2_MODEL_RUNNER=0`, or an option that makes vLLM pick its older model runner | The engine fails to build |

## Refused in a trace

| You do | You get |
| --- | --- |
| Read a module after the model has run past it (layer 20, then layer 10) | `OutOfOrderError` |
| Read a module that is not a tap, on a tapped engine | `OutOfOrderError: ... is not a tap on this engine` |
| `tracer.invoke(["a", "b"])` | An error: one prompt per invoke |
| `tracer.barrier(n)` | `NotImplementedError` |
| A misspelled sampling argument (`temperatur=0.0`) | `TypeError` |
| A backward pass (`with loss.backward():`) | `NotImplementedError`: the forward pass runs without autograd |
| `model.scan(...)` | `NotImplementedError` |
| `edits=["name"]` with no edit installed under that name | `ValueError` |

An error in a block comes back to your process as a `RuntimeError`. Its message carries the
original error type and a traceback that points at your line. The error ends that trace only.
The engine keeps serving, and other clients' requests are not affected.

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

with model.trace("The capital of France is", temperature=0.0):      # the engine still works
    token = model.logits.argmax(-1).save()
print(model.tokenizer.decode(token))
#  Paris
```

## Nothing checks these

- **A replacement with the wrong number of rows.** `layer.output = t` is put back into the
  running batch as given. A wrong size can raise a CUDA error on the GPU, which stops the engine
  and every request in it. When you patch from a prompt of another length, write the rows you
  mean (`served[POS] = donor[POS]`).
- **`.skip()`.** A skip must cover every row of the step, including other requests' rows. If
  anything else is running, the engine stops (`EngineDeadError`). Write the output instead:
  `layer.mlp.output[:] = 0`.
- **A saved view.** A value saved without `.clone()` can hold a later layer's data
  ([Writing a block](blocks.md#clone-what-you-keep)).
- **A loop longer than the request.** A `tracer.iter` loop that asks for more steps than the
  request ran is cut short, and the statements after it do not run. The warning is printed by
  the engine process, not yours.
- **Numbers that depend on the batch.** A request's values vary slightly with the requests it
  was batched with, and in bf16 a nearly tied argmax can change. `temperature=0.0` fixes the
  sampler, not the arithmetic. Compare values read in one trace.

## Where the message is

The block runs in vLLM's engine process, a child of yours.

- **Errors in a block** come back to your process, as above.
- **Warnings** do not. They are printed in the engine's output, on lines that start with
  `(EngineCore pid=...)`.
- **A failure while the engine builds** reaches you as `RuntimeError: Engine core
  initialization failed. See root cause above.` The cause is in the `(EngineCore pid=...)`
  lines earlier in the output.

## Not available on vLLM

| | Use |
| --- | --- |
| Gradients, `.backward()` | `TransformersModel`, with the same block |
| `.scan()` | `TransformersModel`. On vLLM, trace a prompt and read the shapes. |
| Image and video inputs | `TransformersModel`. A vision-language checkpoint loads on vLLM, and its language model is traced with text prompts. |
| The attention pattern as a location | Rebuild it ([Attention patterns](examples/attention.md)) |
| Speculative decoding, pipeline parallelism, data parallelism | Not supported |
