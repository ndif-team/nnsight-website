# Generation

A trace *is* a generation: `max_tokens` decides how many steps it runs, and every step is
observable. Sampling settings go to `trace` / `invoke`, since each invoke is one vLLM request.

## Text

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0, max_tokens=32) as tracer:
    out = tracer.result.save()

print(out.outputs[0].text)
# Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Germany is Berlin. ...
print(out.outputs[0].token_ids[:8], out.outputs[0].finish_reason)
# [12095, 13, 576, 6722, 315, 15344, 374, 21718] length
```

`tracer.result` is vLLM's own `RequestOutput` for the request, served once the engine has
assembled it. Anything `vllm.SamplingParams` takes is a keyword here: `temperature`, `top_p`,
`top_k`, `min_p`, `max_tokens`, `stop`, `stop_token_ids`, `seed`, `repetition_penalty`,
`logprobs`, `n`, `ignore_eos`, ... A misspelled one raises rather than being ignored.

## Without a block

`model.generate(...)` outside a `with` is a plain run: vLLM's `RequestOutput`s, a list of prompts
allowed, nothing traced.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
outputs = model.generate(["The capital of France is", "The capital of Japan is"],
                         temperature=0.0, max_tokens=8)
print([o.outputs[0].text for o in outputs])
# [' Paris. The capital of Italy is Rome', ' Tokyo. The capital of Japan is Tokyo']
```

## Per token, with logprobs

Ask the sampler for them; they ride the finished request.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0, max_tokens=4, logprobs=3) as tracer:
    out = tracer.result.save()

for token_id, top in zip(out.outputs[0].token_ids, out.outputs[0].logprobs):
    print(repr(model.tokenizer.decode(token_id)),
          [(lp.decoded_token, round(lp.logprob, 2)) for lp in top.values()])
# ' Paris' [(' Paris', -0.62), (' a', -2.12), (' in', -3.25)]
# '.' [('.', -0.42), (',', -1.8), ('.\n', -2.17)]
# ' The' [(' The', -0.78), (' What', -2.03), (' This', -3.03)]
# ' capital' [(' capital', -0.25), (' E', -2.63), (' population', -3.88)]
```

## Per step, the logits and the draw

`model.logits` is what the sampler is about to draw from on the current step, `[1, vocab]`;
`model.samples` is the id it drew, `[1, 1]`. Read them under `tracer.all()` for every step or
`tracer.iter[a:b]` for some. The step index is the loop variable.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.8, top_p=0.95, seed=0, max_tokens=6) as tracer:
    steps = list().save()
    for step in tracer.all():
        probs = model.logits.float().softmax(-1)
        drawn = model.samples.item()
        steps.append((step, drawn, round(probs[0, drawn].item(), 3), probs.argmax().item()))

for step, drawn, p, greedy in steps:
    print(step, repr(model.tokenizer.decode(drawn)), p, "greedy:", repr(model.tokenizer.decode(greedy)))
# 0 ' Paris' 0.537 greedy: ' Paris'
# 1 '.' 0.654 greedy: '.'
# 2 ' How' 0.014 greedy: ' The'
# 3 ' many' 0.472 greedy: ' many'
# 4 ' times' 0.022 greedy: ' letters'
# 5 ' does' 0.853 greedy: ' does'
```

Both are per *request*, not per token: vLLM computes `lm_head` only for the position being
sampled. Logits at every prompt position come from the [logit lens](logit-lens.md).

## Forcing the draw

Assign `model.samples` and the engine continues from that token; assign `model.logits` and the
sampler draws from yours.

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
berlin = model.tokenizer.encode(" Berlin")[0]

with model.trace("The capital of France is", temperature=0.0, max_tokens=8) as tracer:
    for step in tracer.iter[:8]:
        if step == 0:
            model.samples = torch.full_like(model.samples, berlin)
    out = tracer.result.save()

print(out.outputs[0].text)
# Berlin. Is this statement true? Please
```

## Stopping early

`tracer.stop()` ends the request at that step and winds it up within a step or two rather than
running to `max_tokens`; whatever was saved *before* the stop comes back. The stop unwinds the
block, so nothing after it runs — a `tracer.result.save()` placed after the loop is never
reached; the saved values are the record.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0, max_tokens=32) as tracer:
    ids = list().save()
    for step in tracer.iter[:32]:
        ids.append(model.samples.item())
        if step == 2:
            tracer.stop()

print(ids, repr(model.tokenizer.decode(ids)))
# [12095, 13, 576] ' Paris. The'
```

A stop that depends on what the model wrote — "stop at the first period" — is on
[Conditional interventions](conditional.md#stop-when-a-condition-is-met).

## Several sequences

`n=k` fans one prompt into `k` sampled continuations. The block runs once per sequence, so a saved
name comes back as a list of `k`, and `tracer.result.outputs[i]` is sequence `i`.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=1.0, seed=0, max_tokens=8, n=3) as tracer:
    first = model.samples.item().save()
    out = tracer.result.save()

print(first)
# [12095, 12095, 12095]
print([c.text for c in out.outputs])
# [' Paris.\n\nThe area of the circle is', ' Paris. The capital of Australia is Canberra', " Paris. I'll have to find a"]
```

## Several prompts

One prompt per invoke; the engine batches them. Sampling settings on an invoke override the
trace's.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace(max_tokens=8) as tracer:
    with tracer.invoke("The capital of France is", temperature=0.0):
        cold = tracer.result.save()
    with tracer.invoke("The capital of France is", temperature=1.5, seed=1):
        hot = tracer.result.save()

print(repr(cold.outputs[0].text), repr(hot.outputs[0].text))
# ' Paris. The capital of Italy is Rome' ' Paris. Where would one go to see'
```

## Prompt forms

A string, a list of token ids, a tokenizer's `{input_ids, attention_mask}` dict, or one of vLLM's
own prompt dicts (`TokensPrompt`, `TextPrompt`) — one per invoke.

```python
from vllm.inputs import TokensPrompt
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
ids = model.tokenizer("The capital of France is")["input_ids"]

with model.trace(temperature=0.0, max_tokens=4) as tracer:
    with tracer.invoke(ids):
        a = tracer.result.save()
    with tracer.invoke(TokensPrompt(prompt_token_ids=ids)):
        b = tracer.result.save()

print(a.outputs[0].text == b.outputs[0].text, repr(a.outputs[0].text))
# True ' Paris. The capital'
```

## Streaming

Token-by-token output needs the async engine: `VLLM(..., mode="async")`, then
`async for output in tracer.backend`. See [Async and servers](serving.md).
