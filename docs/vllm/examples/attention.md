# Attention patterns

vLLM's attention kernel never forms the attention pattern, so it is not a location. The
queries and keys the kernel receives are. Rebuild the pattern from them inside the block.

`self_attn.attn.inputs` is what the kernel is given: the queries, keys and values after
QK-norm and rotary embedding. Qwen3-8B has 32 query heads and 8 key heads, so each key head
serves four query heads.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
attn = model.model.layers[10].self_attn
n_heads, n_kv, d = 32, 8, 128


def pattern(q, k):
    T = q.shape[0]
    qh = q.view(T, n_heads, d).transpose(0, 1).float()                        # [heads, T, d]
    kh = k.view(T, n_kv, d).transpose(0, 1).repeat_interleave(n_heads // n_kv, 0).float()
    scores = (qh @ kh.transpose(-1, -2)) * d ** -0.5
    scores.masked_fill_(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return scores.softmax(-1)                                                 # [heads, to, from]


with model.trace("The capital of France is", temperature=0.0):
    (q, k, v), _ = attn.attn.inputs
    probs = pattern(q, k).save()

print(probs.shape, [round(p, 3) for p in probs[0, -1].tolist()])
# torch.Size([32, 5, 5]) [0.904, 0.02, 0.018, 0.019, 0.039]
```

`pattern` is an ordinary function. It is sent with the block and runs in the worker.

## Check it

The pattern times the values should equal each head's output, which is a location
(`o_proj.input`).

```python
with model.trace("The capital of France is", temperature=0.0):
    (q, k, v), _ = attn.attn.inputs
    heads = attn.o_proj.input.clone().save()                                  # [T, 32 * 128]
    vh = v.view(-1, n_kv, d).transpose(0, 1).repeat_interleave(n_heads // n_kv, 0)
    rebuilt = (pattern(q, k).to(v.dtype) @ vh).transpose(0, 1).reshape(-1, n_heads * d).save()

print((heads - rebuilt).abs().max().item())
# 0.00390625
```

## While generating

On a generation step the block sees one new row of queries and keys. The earlier keys are in
vLLM's cache, which is not a module. Keep the keys from earlier steps to get the new token's
attention over the whole context.

```python
with model.trace("The capital of France is", temperature=0.0, max_tokens=4) as tracer:
    keys, last_row = [], list().save()
    for _ in tracer.iter[:4]:
        (q, k, _), _ = attn.attn.inputs
        keys.append(k.clone())
        K = torch.cat(keys)                                                   # every key so far
        qh = q[-1:].view(1, n_heads, d).transpose(0, 1).float()
        kh = K.view(-1, n_kv, d).transpose(0, 1).repeat_interleave(n_heads // n_kv, 0).float()
        last_row.append(((qh @ kh.transpose(-1, -2)) * d ** -0.5).softmax(-1)[:, 0])

print([tuple(r.shape) for r in last_row])
# [(32, 5), (32, 6), (32, 7), (32, 8)]
```
