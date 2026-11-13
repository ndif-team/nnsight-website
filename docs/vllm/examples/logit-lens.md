# Logit lens

The logit lens sends a layer's residual stream through the model's final norm and unembedding
to read what it predicts at that depth. On vLLM the block does this inside the worker, where
the weights are, and returns only what you save.

`model.model.norm` is the final norm. `model.logits_processor(model.lm_head, h)` is the call
vLLM itself makes to turn a hidden state into logits, so it works under tensor parallelism and
applies any model-specific scaling.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    guesses = list().save()
    for layer in model.model.layers:
        h = model.model.norm(sum(layer.output))
        guesses.append(model.logits_processor(model.lm_head, h)[-1].argmax().item())
    final = model.logits.argmax(-1).item().save()

for i in range(3, 36, 4):
    print(i, repr(model.tokenizer.decode(guesses[i])))
print("model:", repr(model.tokenizer.decode(final)))
# 3 ' ebenfalls'
# 7 '/w'
# 11 ' _______,'
# 15 ' ____'
# 19 ' ____'
# 23 ' ____'
# 27 ' Paris'
# 31 ' Paris'
# 35 ' Paris'
# model: ' Paris'
```

The answer appears at about layer 27 and holds.

## Send home the top tokens, not the vocabulary

The vocabulary has 151,936 entries, so a `[tokens, vocab]` tensor per layer is costly to
return. Take the top tokens in the block.

```python
with model.trace("The capital of France is", temperature=0.0):
    h = model.model.norm(sum(model.model.layers[28].output))
    probs = model.logits_processor(model.lm_head, h)[-1].float().softmax(-1)
    top = probs.topk(5)
    ids, p = top.indices.save(), top.values.save()

print([(model.tokenizer.decode(i), round(v, 3)) for i, v in zip(ids.tolist(), p.tolist())])
# [(' Paris', 0.771), (' located', 0.104), (' ______', 0.03), (' _____', 0.026), (' ____', 0.023)]
```

## Follow a generation

Under `tracer.iter` the lens reads each generated token. This compares what layer 28 predicts
with what the model samples, on every step.

```python
with model.trace("The capital of France is", temperature=0.0, max_tokens=6) as tracer:
    lens, sampled = list().save(), list().save()
    for _ in tracer.iter[:6]:
        h = model.model.norm(sum(model.model.layers[28].output)[-1:])
        lens.append(model.logits_processor(model.lm_head, h).argmax(-1).item())
        sampled.append(model.samples.item())

decode = model.tokenizer.decode
print([(decode(a), decode(b)) for a, b in zip(lens, sampled)])
# [(' Paris', ' Paris'), (' France', '.'), (' True', ' The'), (' French', ' capital'), (' of', ' of'), (' France', ' Italy')]
```

Layer 28 agrees with the model on the first token. On later steps it still reads words about
France, where the model goes on to write a new sentence.
