# Ablation

Zero a component and measure what the answer loses. Every ablation is an in-place write at the
component's own location — an MLP's output, one head's slice of `z` — and a sweep over
components is one trace with one invoke per component, batched by the scheduler. A fixed
steering vocabulary that writes only at residual points cannot express either write.

## Every MLP, one at a time

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
tok = model.tokenizer
PROMPT = "The capital of France is"
paris = tok.encode(" Paris")[0]

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    with tracer.invoke(PROMPT):
        base = model.logits.save()
    for i in range(36):
        with tracer.invoke(PROMPT):
            model.model.layers[i].mlp.output[:] = 0
            logits = model.logits.save()

p = lambda l: round(l.float().softmax(-1)[0, paris].item(), 3)
print(p(base), [p(l) for l in logits])
# 0.537 [0.0, 0.48, 0.478, 0.52, 0.785, 0.508, 0.0, 0.776, 0.611, 0.749, 0.146, 0.6, 0.603, 0.246,
#        0.624, 0.709, 0.385, 0.515, 0.597, 0.747, 0.522, 0.415, 0.468, 0.686, 0.678, 0.638, 0.564,
#        0.304, 0.457, 0.351, 0.637, 0.431, 0.417, 0.241, 0.29, 0.983]
```

`P(Paris)` is 0.537 unablated. Zeroing the MLP of layer 0 or 6 takes it to zero — those two are
load-bearing for the whole forward, not for this fact — while layers 10, 13, 33 and 34 each cost
about half the probability, and removing the last MLP *raises* it to 0.98 (it spreads mass over
continuations like `" the"`). Thirty-seven requests, one trace.

## Every head at one layer

`o_proj.input` is `z`, `[pos, n_heads * head_dim]`; head `h` is its slice.

```python
L, d = 24, 128
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for h in range(32):
        with tracer.invoke(PROMPT):
            model.model.layers[L].self_attn.o_proj.input[:, h * d:(h + 1) * d] = 0
            logits = model.logits.save()

print([p(l) for l in logits])
# [0.563, 0.524, 0.536, 0.531, 0.538, 0.539, 0.526, 0.527, 0.518, 0.534, 0.505, 0.532, 0.543, 0.508,
#  0.523, 0.52, 0.539, 0.521, 0.52, 0.505, 0.538, 0.532, 0.52, 0.549, 0.464, 0.478, 0.54, 0.457,
#  0.536, 0.529, 0.546, 0.537]
```

No single head at layer 24 is necessary on the clean prompt — including head 26, which
[patching](patching.md#one-head) shows is *sufficient* to carry the answer from the subject to the
last position. Necessity and sufficiency are different questions; the two sweeps answer both in
two traces.

## Over generation

Keep the ablation on for every step by putting it under `tracer.iter`; a bare write fires on the
prefill only.

```python
with model.trace(PROMPT, temperature=0.0, max_tokens=8) as tracer:
    for _ in tracer.iter[:8]:
        model.model.layers[10].mlp.output[:] = 0
    out = tracer.result.save()
```

## Variations

- **Mean ablation** instead of zero: replace with the component's mean output over a reference
  set, captured once with [`tracer.cache()`](capture.md#every-layer) and shipped with the block.
- **One position**: index the rows (`mlp.output[3] = 0`) — on the prefill the rows are the
  prompt positions, on a decode step there is one.
- **One neuron**: `mlp.act_fn.output[:, j] = 0` zeroes neuron `j` before the down-projection.
- **An expert** in a mixture-of-experts model: mask its router logit
  ([Steering](steering.md#an-expert)).
- **Under tensor parallelism** the same writes work unchanged: `o_proj.input` is gathered before
  the block sees it and re-sharded after ([Tensor parallelism](tensor-parallel.md)).
