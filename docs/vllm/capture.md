# Capture

Read any location inside `with model.trace(...)`; `.save()` brings it back. The block runs in
vLLM's worker, so what you read is the real tensor, narrowed to your request's rows.

## One location

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    hidden, residual = model.model.layers[10].output
    resid_post = (hidden + residual).save()      # the residual stream leaving block 10

print(resid_post.shape)                          # [pos, d_model] — no batch axis
# torch.Size([5, 4096])
```

`layers[i].output` is `(hidden_states, residual)`; their sum is the residual stream. The sum is a
fresh tensor, so it needs no clone; a raw `.output[0]` does — see [Locations](locations.md).

## Several locations

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
layer = model.model.layers[10]

with model.trace("The capital of France is", temperature=0.0):
    attn_out = layer.self_attn.output.clone().save()      # [pos, 4096]
    mlp_act = layer.mlp.act_fn.output.clone().save()      # [pos, 12288] post-activation neurons
    mlp_out = layer.mlp.output.clone().save()             # [pos, 4096]

print(attn_out.shape, mlp_act.shape, mlp_out.shape)
# torch.Size([5, 4096]) torch.Size([5, 12288]) torch.Size([5, 4096])
```

Read locations in forward order within one block. Reading layer 20 and then layer 10 raises
`OutOfOrderError` when the request ends — the model has already run past 10.

## Every layer

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    resid = list().save()
    for layer in model.model.layers:
        resid.append(sum(layer.output))

print(len(resid), resid[0].shape)
# 36 torch.Size([5, 4096])
```

Or let `tracer.cache()` do it, keyed by module path, with `include_inputs=True` for the inputs and
`device=`/`dtype=` to move captures as they land:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0, max_tokens=1) as tracer:
    cache = tracer.cache(modules=[model.model.layers[i].mlp for i in range(0, 36, 12)])

print(list(cache.keys()))
# ['model.model.layers.0.mlp', 'model.model.layers.12.mlp', 'model.model.layers.24.mlp']
print(cache["model.model.layers.12.mlp"].output.shape)
# torch.Size([5, 4096])
```

A cache must be opened before the first read or edit in the block. When the trace generates more
than one token, each entry is a **list**, one capture per step.

## While generating

A trace spans the prefill and every decode step. `tracer.all()` runs its body once per step;
`tracer.iter[a:b]` on a slice of them. The prefill row count is the prompt length; each decode step
has one row.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0, max_tokens=8) as tracer:
    resid = list().save()
    for _ in tracer.iter[:8]:
        resid.append(sum(model.model.layers[10].output)[-1])   # the running token's row
    out = tracer.result.save()

print(len(resid), resid[0].shape, repr(out.outputs[0].text))
# 8 torch.Size([4096]) ' Paris. The capital of Italy is Rome'
```

Bound the loop when anything follows it: an open-ended `tracer.all()` parks the block waiting
for a step that never comes, so the lines after it do not run. Captures at prompt *and* generated
positions: the prefill rows plus one per step, the last sampled token never being fed back.

## Many prompts

One prompt per `tracer.invoke(...)`; each is its own vLLM request and the scheduler batches them.
A name saved in every invoke comes back as a list, one entry per invoke, in order.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
prompts = ["Paris is in", "Berlin is in", "The capital of Japan is"]

with model.trace(temperature=0.0) as tracer:
    for prompt in prompts:
        with tracer.invoke(prompt):
            resid = sum(model.model.layers[10].output).save()
            token = model.logits.argmax(-1).save()

for p, r, t in zip(prompts, resid, token):
    print(f"{p!r:28} {tuple(r.shape)}  -> {model.tokenizer.decode(t)!r}")
# 'Paris is in'                (3, 4096)  -> ' the'
# 'Berlin is in'               (3, 4096)  -> ' the'
# 'The capital of Japan is'    (5, 4096)  -> ' Tokyo'
```

Prompts keep their true length — no padding. A list of prompts in one invoke is refused; so is
`tracer.barrier`, since the requests never share a forward.

## The fused projections

vLLM merges what HuggingFace keeps separate. Split them yourself:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
attn = model.model.layers[10].self_attn
n_heads, n_kv, d = 32, 8, 128

with model.trace("The capital of France is", temperature=0.0):
    qkv = attn.qkv_proj.output[0]                           # [pos, (32 + 8 + 8) * 128]
    q, k, v = qkv.split([n_heads * d, n_kv * d, n_kv * d], dim=-1)
    v = v.clone().save()
    gate, up = model.model.layers[10].mlp.gate_up_proj.output[0].chunk(2, dim=-1)
    gate = gate.clone().save()                              # pre-activation neurons

print(v.shape, gate.shape)
# torch.Size([5, 1024]) torch.Size([5, 12288])
```

Under tensor parallelism the fused tensor is gathered in rank order, so slice by head rather than
by `[:q_size]` — see [Tensor parallelism](tensor-parallel.md).

## MoE routing

The router is an ordinary linear, so its logits are a location. Run on `Qwen/Qwen1.5-MoE-A2.7B`:

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen1.5-MoE-A2.7B", dispatch=True)
moe = model.model.layers[10].mlp

with model.trace("The capital of France is", temperature=0.0):
    router_logits = moe.gate.output[0].clone().save()       # [pos, n_experts]
    experts_out = moe.experts.output.clone().save()         # [pos, d_model], all experts combined

top = router_logits.float().softmax(-1).topk(4, dim=-1)
print(router_logits.shape, experts_out.shape)
# torch.Size([5, 60]) torch.Size([5, 2048])
print(top.indices[-1].tolist(), [round(w, 3) for w in top.values[-1].tolist()])
# [23, 57, 41, 1] [0.218, 0.17, 0.119, 0.04]
```

Individual experts are not addressable: vLLM stacks the local experts into one fused kernel, so
there is no `experts[3]` to hook. To ablate an expert, mask its router logit — see
[Steering](steering.md#an-expert).

## Gradients

Not on vLLM. The forward runs under `torch.inference_mode()` and the paged-attention kernels keep
no graph, so `.backward()`, `.grad` and everything built on them need the HuggingFace path —
[`TransformersModel`](../features/3_gradients.ipynb) — with the same block.
