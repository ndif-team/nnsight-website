# Logit lens

Send a residual through the model's real final norm and unembedding — *inside the worker*, where
the weights are. Only what you save crosses back.

## The model's own arithmetic

`model.model.norm` is the final norm; `model.logits_processor(model.lm_head, h)` is exactly what
vLLM calls to turn the last hidden state into logits, so any family-specific step (a muP scale,
Gemma's soft-capping) is applied for you. Calling a module inside a trace runs it directly, out of
the forward's order.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    resid = sum(model.model.layers[28].output)                # [pos, d_model], leaving block 28
    lens = model.logits_processor(model.lm_head, model.model.norm(resid))
    top = lens.argmax(-1).save()                              # [pos]; the logits stay in the worker

print([model.tokenizer.decode(t) for t in top])
# ['玿', 'ization', ' China', ' France', ' Paris']
```

The last row is the model's guess at the next token after reading the whole prompt, from layer
28's point of view.

## Every layer, one forward

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    guesses = list().save()
    for layer in model.model.layers:
        h = model.model.norm(sum(layer.output))
        guesses.append(model.logits_processor(model.lm_head, h)[-1].argmax().item())
    final = model.logits.argmax(-1).item().save()

for i, g in enumerate(guesses):
    if i % 4 == 3 or i == 35:
        print(i, repr(model.tokenizer.decode(g)))
print("model:", repr(model.tokenizer.decode(final)), "lens at 35 agrees:", guesses[-1] == final)
# 3 ' ebenfalls'
# 7 '/w'
# 11 ' _______,'
# 15 ' ____'
# 19 ' ____'
# 23 ' ____'
# 27 ' Paris'
# 31 ' Paris'
# 35 ' Paris'
# model: ' Paris' lens at 35 agrees: True
```

Through the middle of the stack the lens reads a fill-in-the-blank continuation; the answer
appears around layer 27 and holds.

## Top-k without shipping the vocab

The vocabulary is 151,936 wide; a `[pos, vocab]` tensor per layer is the expensive thing to send
home. Take the top-k in the worker.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    h = model.model.norm(sum(model.model.layers[28].output))
    probs = model.logits_processor(model.lm_head, h)[-1].float().softmax(-1)
    top = probs.topk(5)
    ids, p = top.indices.save(), top.values.save()

print([(model.tokenizer.decode(i), round(v, 3)) for i, v in zip(ids.tolist(), p.tolist())])
# [(' Paris', 0.771), (' located', 0.104), (' ______', 0.03), (' _____', 0.026), (' ____', 0.023)]
```

## Every prompt position, every step

vLLM computes the real `lm_head` only for the token being sampled, so `model.logits` is one row.
The lens gives you the whole prompt — and under `tracer.all()` it follows generation.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0, max_tokens=6) as tracer:
    rows = list().save()
    for _ in tracer.iter[:6]:
        h = model.model.norm(sum(model.model.layers[28].output))
        rows.append(model.logits_processor(model.lm_head, h).argmax(-1))
    out = tracer.result.save()

print([len(r) for r in rows])                                    # prompt rows, then one per step
# [5, 1, 1, 1, 1, 1]
print(repr(out.outputs[0].text))
# ' Paris. The capital of Italy'
```

## Raw logits

`h @ model.lm_head.weight.T` skips the family arithmetic. On one GPU the weight is the whole
`[vocab, d_model]`; under tensor parallelism it is this rank's shard, whereas
`logits_processor` gathers — see [Tensor parallelism](tensor-parallel.md).

## Optimizing against the lens

Gradients do not flow through a vLLM forward. Fit a lens vector, or optimize a residual against a
logit objective, with the same block on [`TransformersModel`](../tutorials/mini-papers/jacobian-lens.ipynb)
and bring the vector here to [steer](steering.md) with.
