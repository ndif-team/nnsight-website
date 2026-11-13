# Steering

Steering adds a direction to the residual stream while the model generates. This example
builds the direction from a contrast between two sets of prompts and adds it at one layer.

## Build a direction

Read the residual stream after layer 12 for each prompt, one invoke per prompt, and take the
difference of the two groups' means. The first token is left out: its norm is far larger than
the others'.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
layer = model.model.layers[12]
HAPPY = ["I love this. It is wonderful and", "What a great, happy day. I feel"]
SAD = ["I hate this. It is terrible and", "What an awful, sad day. I feel"]

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for prompt in HAPPY + SAD:
        with tracer.invoke(prompt):
            stream = sum(layer.output)[1:]
            mean = stream.mean(0).save()
            norm = stream.norm(dim=-1).median().save()

vector = torch.stack(mean[:2]).mean(0) - torch.stack(mean[2:]).mean(0)
vector = vector / vector.norm()
typical = torch.stack(norm).mean().item()
print(vector.shape, round(typical))
# torch.Size([4096]) 74
```

`typical` is the usual norm of a token's residual stream at this layer. Set the steering
strength as a multiple of it.

## Add it while generating

Three invokes steer the same prompt against the direction, not at all, and along it. The edit
is under `tracer.iter`, so it applies on every step. An edit in the block body would apply to
the prompt only.

```python
PROMPT = "I think that this movie is"

with model.trace(temperature=0.0, max_tokens=16) as tracer:
    for strength in (-1.0, 0.0, 1.0):
        with tracer.invoke(PROMPT):
            for _ in tracer.iter[:16]:
                layer.output[0][:] += strength * typical * vector
            out = tracer.result.save()

for strength, o in zip((-1.0, 0.0, 1.0), out):
    print(strength, repr(o.outputs[0].text))
# -1.0 ' not good. I think that this movie is not good. I think that this'
# 0.0 ' a very good example of how to make a movie that is not only entertaining,'
# 1.0 ' a great example of a family film. The story is very engaging and the characters'
```

`layer.output` is the pair `(hidden, residual)` and the next layer adds them, so adding to
either one adds to the residual stream.

## Remove a direction

Any expression is allowed. This removes the direction from the residual stream at every step
by projecting it out of both halves of the pair.

```python
with model.trace(PROMPT, temperature=0.0, max_tokens=16) as tracer:
    for _ in tracer.iter[:16]:
        for half in layer.output:
            half[:] -= (half @ vector).unsqueeze(-1) * vector
    out = tracer.result.save()

print(repr(out.outputs[0].text))
# " a very good example of how to make a bad movie. It's not that"
```

## Steer the sampler

`model.logits` and `model.samples` are writable too. This bans one token on every step.

```python
banned = model.tokenizer.encode(" Paris")[0]

with model.trace("The capital of France is", temperature=0.0, max_tokens=6) as tracer:
    for _ in tracer.iter[:6]:
        model.logits[:, banned] = float("-inf")
    out = tracer.result.save()

print(repr(out.outputs[0].text))
# ' a city in the country of'
```
