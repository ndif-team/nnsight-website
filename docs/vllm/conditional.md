# Conditional interventions

A block is a program that runs inside the forward, so an intervention can depend on what the
model is doing *right now*: a lens read, a probability, a feature's activation, the token just
sampled. The decision is taken in the worker on every step; nothing round-trips. A fixed
steering spec has no branch in it.

## Act on what an earlier layer already believes

Read the [logit lens](logit-lens.md) at layer 28 each step; if it already says ` Paris`, make
the sampler draw ` Berlin` instead.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
tok = model.tokenizer
PROMPT = "The capital of France is"
paris, berlin = tok.encode(" Paris")[0], tok.encode(" Berlin")[0]

with model.trace(PROMPT, temperature=0.0, max_tokens=8) as tracer:
    swapped = list().save()
    for step in tracer.iter[:8]:
        h = model.model.norm(sum(model.model.layers[28].output)[-1:])
        guess = model.logits_processor(model.lm_head, h).argmax(-1).item()
        if guess == paris:
            model.samples[:] = berlin
        swapped.append(guess == paris)
    out = tracer.result.save()

print([i for i, s in enumerate(swapped) if s], repr(out.outputs[0].text))
# [0] ' Berlin. Is this statement true? Please'
```

The lens fired on the first step only; the engine continued from ` Berlin` as if the model had
chosen it.

## Steer only while a condition holds

Suppress a token only on the steps where the model is about to say it.

```python
with model.trace(PROMPT, temperature=0.0, max_tokens=8) as tracer:
    p_paris = list().save()
    for _ in tracer.iter[:8]:
        p = model.logits.float().softmax(-1)[0, paris]
        if p > 0.2:
            model.logits[:, paris] = float("-inf")
        p_paris.append(round(p.item(), 3))
    out = tracer.result.save()

print(p_paris, repr(out.outputs[0].text))
# [0.537, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0] ' a city in the country of France.'
```

`p_paris` is the model's own probability before the edit, one per step — the record of when the
condition fired comes back with the result.

## Stop when a condition is met

`tracer.stop()` ends the request as soon as it is called; vLLM retires it within a step or two
instead of running to `max_tokens`.

```python
period = tok.encode(".")[0]

with model.trace(PROMPT, temperature=0.0, max_tokens=32) as tracer:
    ids = list().save()
    for _ in tracer.iter[:32]:
        ids.append(model.samples.item())
        if model.samples.item() == period:
            tracer.stop()

print(ids, repr(tok.decode(ids)))
# [12095, 13] ' Paris.'
```

`stop()` unwinds the block: **nothing after it runs**, including a `tracer.result.save()` placed
after the loop, so save what you want before the stop — the values are the record. Here the
model wrote two tokens of a possible 32.

## Where the condition can come from

- A read at any location: the lens, a probability, a residual norm (layer 20's first position
  sits at 9280 against ~130 elsewhere on this prompt — a massive activation, detectable in one
  line), a probe's output, an [SAE feature's live activation](sae.md).
- The sampled token (`model.samples`), the step index (the loop variable), running state kept
  in ordinary Python variables across steps.
- Anything shipped with the block: a direction, a probe, a small module.

## Rules

- **Every rank runs the block**, so under tensor parallelism the condition must evaluate the
  same everywhere — it does, as long as it reads gathered values or sampled ids and not the
  rank. Sampling must be greedy or seeded ([Tensor parallelism](tensor-parallel.md)).
- **A condition is Python in the worker.** `p > 0.2` on a one-element tensor is a device
  sync; keep the test small, or read `.item()` once and branch on the number.
- Under [graph taps](performance.md) the same code runs, as long as every location it reads is
  a tap; `model.logits` and `model.samples` always are.
