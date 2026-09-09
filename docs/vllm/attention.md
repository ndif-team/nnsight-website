# Attention

The paged-attention kernel never materializes the score matrix, so `attn_probs` is not a location
on any vLLM model. Everything *around* the kernel is: the fused projection, q and k after QK-norm
and RoPE, v, and the per-head output before `W_O`. The pattern is rebuilt from q and k, inside the
worker, in a few lines.

## q, k, v

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
attn = model.model.layers[10].self_attn

with model.trace("The capital of France is", temperature=0.0):
    (q, k, v), _ = attn.attn.inputs                 # after QK-norm and RoPE; what the kernel gets
    q, k, v = q.clone().save(), k.clone().save(), v.clone().save()

print(q.shape, k.shape, v.shape)                    # [pos, 32*128], [pos, 8*128], [pos, 8*128]
# torch.Size([5, 4096]) torch.Size([5, 1024]) torch.Size([5, 1024])
```

`attn.qkv_proj.output[0]` is the same three before norm and RoPE, fused into `[pos, 6144]`; split
it as `[4096, 1024, 1024]`.

## Scores and probabilities

Rebuilt from the tensors the kernel was handed. Qwen3-8B has 32 query heads over 8 KV heads, so
each key head serves four query heads.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
attn = model.model.layers[10].self_attn
n_heads, n_kv, d = 32, 8, 128


def pattern(q, k):
    T = q.shape[0]
    qh = q.view(T, n_heads, d).transpose(0, 1).float()                        # [H, T, d]
    kh = k.view(T, n_kv, d).transpose(0, 1).repeat_interleave(n_heads // n_kv, 0).float()
    scores = (qh @ kh.transpose(-1, -2)) * d ** -0.5
    scores.masked_fill_(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return scores, scores.softmax(-1)


with model.trace("The capital of France is", temperature=0.0):
    (q, k, v), _ = attn.attn.inputs
    scores, probs = pattern(q, k)
    scores, probs = scores.save(), probs.save()                             # [H, dest, src]

print(probs.shape, [round(p, 3) for p in probs[0, -1].tolist()])
# torch.Size([32, 5, 5]) [0.904, 0.02, 0.018, 0.019, 0.039]
```

Against HuggingFace's eager attention on the same prompt this matches to bf16 noise (max
difference 0.026, cosine 0.99998). `pattern` is an ordinary function, shipped with the block and
run in the worker, so only what you saved crossed back.

## Per-head outputs and direct attribution

`o_proj.input` is `z`: every head's output, before `W_O`, head *h* at columns `h*128:(h+1)*128`.
`probs @ v == z` holds head by head, which is what direct feature attribution needs.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
attn = model.model.layers[10].self_attn
n_heads, n_kv, d = 32, 8, 128

with model.trace("The capital of France is", temperature=0.0):
    (q, k, v), _ = attn.attn.inputs
    z = attn.o_proj.input.clone().save()                                     # [pos, 32*128]
    _, probs = pattern(q, k)
    vh = v.view(-1, n_kv, d).transpose(0, 1).repeat_interleave(n_heads // n_kv, 0)
    z_rebuilt = (probs.to(v.dtype) @ vh).transpose(0, 1).reshape(-1, n_heads * d).save()

print((z - z_rebuilt).abs().max().item())
# 0.00390625
```

A head's contribution to the residual is its slice of `z` through its slice of `W_O`:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
attn = model.model.layers[10].self_attn
head, d = 7, 128

with model.trace("The capital of France is", temperature=0.0):
    z = attn.o_proj.input
    W_O = attn.o_proj.weight                                                 # [4096, 4096]
    contribution = (z[:, head * d:(head + 1) * d] @ W_O[:, head * d:(head + 1) * d].T).save()

print(contribution.shape)                                                    # [pos, d_model]
# torch.Size([5, 4096])
```

## Several layers

Read in forward order within one block.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    patterns = {}
    for i in (0, 10, 20):
        (q, k, _), _ = model.model.layers[i].self_attn.attn.inputs
        patterns[i] = pattern(q, k)[1]
    patterns = patterns.save()

print({i: tuple(p.shape) for i, p in patterns.items()})
# {0: (32, 5, 5), 10: (32, 5, 5), 20: (32, 5, 5)}
```

## While generating

On a decode step q has one row; k and v have one row too — the rest of the keys live in the paged
KV cache, which is not a module. A pattern *over the whole context* on a decode step therefore
needs the keys you kept from earlier steps:

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
attn = model.model.layers[10].self_attn
n_heads, n_kv, d = 32, 8, 128

with model.trace("The capital of France is", temperature=0.0, max_tokens=4) as tracer:
    keys, last_row = [], list().save()
    for _ in tracer.iter[:4]:
        (q, k, _), _ = attn.attn.inputs
        keys.append(k.clone())
        K = torch.cat(keys)                                                  # every key so far
        qh = q[-1:].view(1, n_heads, d).transpose(0, 1).float()
        kh = K.view(-1, n_kv, d).transpose(0, 1).repeat_interleave(n_heads // n_kv, 0).float()
        last_row.append(((qh @ kh.transpose(-1, -2)) * d ** -0.5).softmax(-1)[:, 0])

print([tuple(r.shape) for r in last_row])                                    # [H, src] grows by one
# [(32, 5), (32, 6), (32, 7), (32, 8)]
```

## Under tensor parallelism

Heads are sharded across ranks; nnsight gathers `qkv_proj.output` and `o_proj.input` before your
block reads them, so the code above is unchanged at `tensor_parallel_size=2`. The gathered fused
tensor is in rank order — `[q0 k0 v0 | q1 k1 v1]` — so split it by head, not by `[:q_size]`. See
[Tensor parallelism](tensor-parallel.md).
