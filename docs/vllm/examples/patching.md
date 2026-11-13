# Activation patching

Activation patching copies an activation from a clean run into a corrupt run and checks
whether the answer follows. On vLLM the two prompts are two requests. Save the clean
activations in one trace and write them in the next. A sweep is one trace with one invoke per
patch, and vLLM batches the invokes.

## Clean run

The two prompts have the same length and differ in one token.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
tok = model.tokenizer
CLEAN, CORRUPT = "The capital of France is", "The capital of Germany is"    # 5 tokens each
paris, berlin = tok.encode(" Paris")[0], tok.encode(" Berlin")[0]

with model.trace(CLEAN, temperature=0.0):
    clean = list().save()
    for layer in model.model.layers:
        hidden, residual = layer.output
        clean.append((hidden.clone(), residual.clone()))
    logits = model.logits.save()

diff = lambda l: round((l[0, paris] - l[0, berlin]).item(), 1)
print(len(clean), clean[0][0].shape, diff(logits))
# 36 torch.Size([5, 4096]) 7.1
```

The metric is the logit of ` Paris` minus the logit of ` Berlin`: about `+7` on the clean
prompt and `-7` on the corrupt one.

## Sweep the layers

Patch both halves of a layer's output at the subject token (position 3), one layer per invoke.
`clean` was saved by the trace above, so it travels with the block.

```python
POS = 3
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    with tracer.invoke(CORRUPT):
        base = model.logits.save()
    for i in range(36):
        with tracer.invoke(CORRUPT):
            hidden, residual = model.model.layers[i].output
            hidden[POS] = clean[i][0][POS]
            residual[POS] = clean[i][1][POS]
            patched = model.logits.save()

print(diff(base), [diff(p) for p in patched])
# -6.9 [7.1, 7.1, 7.1, 6.8, 6.7, 6.4, 6.4, 6.2, 6.4, 6.4, 7.0, 6.9, 6.9, 7.1, 7.1, 7.2, 7.9, 7.7,
#       7.7, 7.4, 7.4, 7.4, 7.8, 7.6, -0.9, -0.9, -3.6, -4.1, -4.2, -4.6, -4.2, -6.9, -6.9, -6.8, -6.8, -6.9]
```

Patching the subject position restores the answer at every layer up to 23. After that the
information has left the subject position.

## One attention head

`o_proj.input` holds every head's output before the output projection, with head `h` in
columns `h*128:(h+1)*128`. Patching one slice patches one head.

```python
L, d = 24, 128
with model.trace(CLEAN, temperature=0.0):
    clean_heads = model.model.layers[L].self_attn.o_proj.input.clone().save()

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for h in range(32):
        with tracer.invoke(CORRUPT):
            heads = model.model.layers[L].self_attn.o_proj.input
            heads[:, h * d:(h + 1) * d] = clean_heads[:, h * d:(h + 1) * d]
            patched = model.logits.save()

effect = [round(diff(p) - diff(base), 1) for p in patched]
print(max(range(32), key=lambda h: effect[h]), max(effect), sorted(effect)[-2])
# 26 9.1 0.2
```

Head 26 of layer 24 moves the logit difference by `+9.1` on its own. The next largest effect is
`0.2`.

## What to know

- Check that both prompts tokenize to the same length, so a position means the same token.
- Patch both halves of the pair. Patching one patches half of the residual stream.
- Values vary slightly with the batch, so a sweep reproduces to about one part in a hundred.
