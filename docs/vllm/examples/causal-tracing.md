# Causal tracing

ROME-style activation patching (Meng et al., 2022): corrupt the subject's embeddings, then
restore the clean residual stream at one `(layer, position)` at a time and see how much of the
answer comes back. The result is a `(layers × positions)` map of where the fact is carried.

Three passes on `Qwen/Qwen3-8B`: a clean run that keeps every layer's residual stream, a
corrupted run, and then one request per `(layer, position)` — all of a layer's positions batched
into one trace, which vLLM schedules together.

## Setup

```python
import torch
import nnsight
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
tok = model.tokenizer
layers = model.model.layers

prompt = "The Eiffel Tower is in the city of"
subject = "Eiffel Tower"
answer = tok.encode(" Paris")[0]

enc = tok(prompt, return_offsets_mapping=True)
T = len(enc["input_ids"])
lo, hi = prompt.index(subject), prompt.index(subject) + len(subject)
subject_pos = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < hi and b > lo]
print(T, "tokens; subject at", subject_pos, [tok.decode(enc["input_ids"][i]) for i in subject_pos])
# 10 tokens; subject at [1, 2, 3, 4] [' E', 'iff', 'el', ' Tower']
```

## 1. Clean run

Keep the residual stream leaving every block, for every position, and the answer probability.
The embedding scale sets the noise level: ROME corrupts with three standard deviations of the
embeddings.

```python
with model.trace(prompt, temperature=0.0, max_tokens=1):
    emb = model.model.embed_tokens.output
    sigma = nnsight.save(emb.float().std().item())
    clean = list().save()
    for layer in layers:
        clean.append(sum(layer.output).clone())          # [T, d_model], resid_post
    p_clean = nnsight.save(model.logits.float().softmax(-1)[0, answer].item())

print(f"P({tok.decode(answer)!r}) clean = {p_clean:.3f}   embedding std = {sigma:.4f}")
# P(' Paris') clean = 0.833   embedding std = 0.0261
```

## 2. Corrupted run

Seeded noise on the subject rows of the embedding output, in place, before block 0 reads it.

```python
def corrupt():
    emb = model.model.embed_tokens.output
    g = torch.Generator(device=emb.device).manual_seed(0)
    noise = torch.randn(len(subject_pos), emb.shape[-1], generator=g, device=emb.device)
    emb[subject_pos] += (3 * sigma * noise).to(emb.dtype)


with model.trace(prompt, temperature=0.0, max_tokens=1):
    corrupt()
    p_corrupt = nnsight.save(model.logits.float().softmax(-1)[0, answer].item())
    guess = model.logits.argmax(-1).save()

print(f"P(answer) corrupted = {p_corrupt:.3f}; the model now says {tok.decode(guess)!r}")
# P(answer) corrupted = 0.000; the model now says ' E'
```

## 3. Restore, one `(layer, position)` per request

`layers[l].output` is `(hidden, residual)` and the stream is their sum, so writing
`hidden[pos] = clean[pos] - residual[pos]` restores exactly the clean stream at that position.
One trace per layer, one invoke per position: the ten requests of a layer run as one batch.

```python
recovery = torch.zeros(len(layers), T)

for l, clean_l in enumerate(clean):
    layer = layers[l]                     # bound outside the block: see Performance
    with model.trace(temperature=0.0, max_tokens=1) as tracer:
        for pos in range(T):
            with tracer.invoke(prompt):
                corrupt()
                hidden, residual = layer.output
                hidden[pos] = clean_l[pos] - residual[pos]
                p = nnsight.save(model.logits.float().softmax(-1)[0, answer].item())
    recovery[l] = (torch.tensor(p) - p_corrupt) / (p_clean - p_corrupt)

chars = " ░▒▓█"
print("       " + "".join(f"{tok.decode(t)[:6]:>7}" for t in enc["input_ids"]))
for l in range(0, len(layers), 3):
    row = "".join(f"{chars[int(min(max(v, 0), 0.999) * 5)]:>7}" for v in recovery[l].tolist())
    print(f"L{l:02d}  {row}")
best = recovery.flatten().topk(3)
for v, i in zip(best.values.tolist(), best.indices.tolist()):
    print(f"layer {i // T:2d}  pos {i % T} ({tok.decode(enc['input_ids'][i % T])!r})  recovery {v:.2f}")
#            The      E    iff     el  Tower     is     in    the   city     of
# L00                             █      █
# L03                             ▓      █
# L06                             ░      █
# L09                             ░      █
# L12                             ░      █
# L15                                    █
# L18                                    ▓
# L21
# L24                                                                       ▓
# L27                                                                       █
# L30                                                                       █
# L33                                                                       █
# layer 11  pos 4 (' Tower')  recovery 1.08
# layer 12  pos 4 (' Tower')  recovery 1.08
# layer 13  pos 4 (' Tower')  recovery 1.07
```

The ROME picture: restoring the last subject token (`Tower`) in the early and middle layers
brings the answer back, and from layer 24 on only the final position matters. All 360 patched
requests took 5.0 s.

`p` is one name saved by ten invokes, so it comes back as a list of ten, in position order.

## What this costs

360 requests. Each trace ships its block once per invoke — a few kilobytes plus the one
`[T, d_model]` clean slice it references — and vLLM batches the invokes of a trace into as few
steps as they fit. The same experiment through a per-request HTTP hook is 360 round trips with
the hook function and the clean vector cloudpickled into each; see
[Comparisons](../comparisons.md#vllm-lens).
