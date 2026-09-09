# Activation patching

Paste an activation from a *clean* run into a *corrupt* run and see whether the answer follows.
On vLLM the two prompts are two requests, so the clean activation is saved in one trace and
written in the next; the sweep over layers is one trace with one invoke per layer, batched by
the scheduler.

## Clean and corrupt

Same length, one token apart; the answer token is a single id on each side.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
tok = model.tokenizer
CLEAN, CORRUPT = "The capital of France is", "The capital of Germany is"    # 5 tokens each
paris, berlin = tok.encode(" Paris")[0], tok.encode(" Berlin")[0]

with model.trace(CLEAN, temperature=0.0):
    clean = list().save()
    for layer in model.model.layers:
        hidden, residual = layer.output                  # the pair the next block sums
        clean.append((hidden.clone(), residual.clone()))
    logits = model.logits.save()

print(len(clean), clean[0][0].shape, round((logits[0, paris] - logits[0, berlin]).item(), 2))
# 36 torch.Size([5, 4096]) 7.06
```

The metric is the **logit difference** `Paris − Berlin` at the last position: `+7.06` on the
clean prompt, `−6.94` on the corrupt one. A patch that recovers a positive difference carried
the answer.

## Sweep the layers at one position

Patch both halves of a layer's output at the subject token (position 3, ` France` / ` Germany`),
one layer per invoke. `clean` is a saved value from the trace above, so it ships with the block.

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

diff = lambda l: round((l[0, paris] - l[0, berlin]).item(), 1)
print(diff(base), [diff(p) for p in patched])
# -6.9 [7.1, 7.1, 7.1, 6.8, 6.7, 6.4, 6.4, 6.2, 6.4, 6.4, 7.0, 6.9, 6.9, 7.1, 7.1, 7.2, 7.9, 7.7,
#       7.7, 7.4, 7.4, 7.4, 7.8, 7.6, -0.9, -0.9, -3.6, -4.1, -4.2, -4.6, -4.2, -6.9, -6.9, -6.8, -6.8, -6.9]
```

Patching the subject's residual stream recovers the answer completely at every layer up to 23,
then stops working: after that the information has left the subject position.

## The same sweep at the last position

```python
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for i in range(36):
        with tracer.invoke(CORRUPT):
            hidden, residual = model.model.layers[i].output
            hidden[-1] = clean[i][0][-1]
            residual[-1] = clean[i][1][-1]
            patched = model.logits.save()

print([diff(p) for p in patched])
# [-6.9, -7.0, -6.9, -7.1, -7.0, -6.8, -6.6, -6.7, -6.7, -6.7, -6.2, -6.4, -6.4, -6.1, -5.9, -5.8,
#  -5.2, -5.5, -5.8, -5.9, -5.9, -5.6, -4.9, -3.8, 2.0, 1.8, 2.8, 3.0, 2.6, 2.8, 2.8, 6.8, 6.8, 6.7, 6.9, 7.1]
```

The mirror image: the last position carries nothing until layer 24, exactly where the subject
position stopped mattering — the answer moves from ` France` to ` is` across layers 23–24, and
is fully present at the last position from layer 31.

Each invoke is its own request, so the 36 patched runs and the baseline share one forward
pass per step; a full layer × position grid is one trace with `36 * 5` invokes. The full
grid, with the subject *noised* rather than swapped, is worked through as
[Causal tracing](examples/causal-tracing.md).

## One head

`o_proj.input` is `z`, every head's output before `W_O`; patching a head's slice patches that
head. This is a write to a location that is not a residual point, which a fixed steering
vocabulary cannot express.

```python
L, d = 24, 128
with model.trace(CLEAN, temperature=0.0):
    clean_z = model.model.layers[L].self_attn.o_proj.input.clone().save()

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for h in range(32):
        with tracer.invoke(CORRUPT):
            z = model.model.layers[L].self_attn.o_proj.input
            z[:, h * d:(h + 1) * d] = clean_z[:, h * d:(h + 1) * d]
            patched = model.logits.save()

print([round(diff(p) - diff(base), 2) for p in patched])
# [0.12, 0.12, 0.06, -0.06, 0.06, 0.12, 0.06, 0.06, 0.06, 0.0, 0.06, 0.0, -0.06, 0.12, 0.06, 0.0,
#  -0.06, 0.0, -0.06, 0.12, -0.06, 0.06, -0.06, 0.0, -0.44, 0.25, 9.19, 0.0, 0.0, 0.06, 0.06, 0.0]
```

One head — layer 24, head 26 — carries the answer from the subject to the last position on its
own: patching its `z` alone moves the logit difference by `+9.2`, every other head by less
than `0.5`. That is the mover head the two sweeps above pointed at, found in one trace of 32
invokes.


## Rules

- **Positions line up only on the prefill.** Both prompts must tokenize to the same length for
  index `POS` to mean the same thing; check with `model.tokenizer` first.
- **Write both halves.** `layers[i].output` is `(hidden, residual)` and the stream is their sum;
  patching one half patches half the stream.
- **Two traces, not a barrier.** A saved value from one trace is an ordinary tensor in the next.
  `tracer.barrier` is refused on vLLM: invokes are separate requests and never share a forward.
- **Under tensor parallelism** the same code runs unchanged — `o_proj.input` is gathered whole,
  and the write is re-sharded ([Tensor parallelism](tensor-parallel.md)).
