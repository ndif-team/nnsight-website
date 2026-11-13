# Tensor parallelism

`tensor_parallel_size=N` splits the model across `N` GPUs. Your block does not change. Each
value it reads is the whole tensor, and what it writes is split again before vLLM continues.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True, tensor_parallel_size=2)
attn = model.model.layers[10].self_attn

with model.trace("The capital of France is", temperature=0.0, max_tokens=8) as tracer:
    qkv = attn.qkv_proj.output[0].clone().save()        # each GPU computes half; read whole
    heads = attn.o_proj.input.clone().save()            # each GPU consumes half; read whole
    for _ in tracer.iter[:8]:
        attn.o_proj.input[:, 7 * 128:8 * 128] = 0       # zero head 7, on whichever GPU holds it
    out = tracer.result.save()

print(qkv.shape, heads.shape, repr(out.outputs[0].text))
# torch.Size([5, 6144]) torch.Size([5, 4096]) ' Paris. The capital of France is Paris'
```

nnsight gathers a value only at a location a block reads, so layers you do not touch cost
nothing extra.

## What to know

- **Every GPU's worker runs the block.** Do not branch on the worker's rank. Sample greedily
  or with a seed, so every worker sees the same tokens.
- **Weights are not gathered.** Inside a block, `o_proj.weight` is one GPU's slice. For a logit
  lens, call the model's own logits path, which gathers the vocabulary:
  `model.logits_processor(model.lm_head, h)`.
- **A fused projection is gathered in GPU order.** At `tensor_parallel_size=2`, `qkv_proj.output`
  is `[q0 k0 v0 | q1 k1 v1]`, where one GPU's is `[q | k | v]`. Slice it by head. The per-head
  tensor `o_proj.input` keeps the single-GPU order.
- **Mixture-of-experts models work** in both of vLLM's expert layouts. The router is the same
  on every GPU.
- **Use taps to scale.** The default engine does not get faster as GPUs are added, and a tapped
  engine does ([Performance](performance.md#tensor-parallelism)).
- **A script needs the [`__main__` guard](setup.md#in-a-script).**

## Not supported

- **Pipeline parallelism** (`pipeline_parallel_size > 1`). Each stage holds part of the model, so
  a block sent to a stage cannot find the modules of another. Every trace ends in
  `UnknownPersistentIdError`.
- **Data parallelism** (`data_parallel_size > 1`). It is refused at construction. Build one
  `VLLM` per GPU.
