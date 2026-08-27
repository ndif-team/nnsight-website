# Chat and tokens

`model.tokenizer` is the HuggingFace tokenizer vLLM resolved for the checkpoint, available on the
meta tree too. Use the model's own chat template rather than building a prompt by hand.

## Apply a chat template

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
tok = model.tokenizer
messages = [{"role": "user", "content": "In one word, what is the capital of France?"}]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)          # Qwen3: skip the <think> block
print(repr(text[-60:]))
# ' France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

with model.trace(text, temperature=0.0, max_tokens=8) as tracer:
    out = tracer.result.save()

print(repr(out.outputs[0].text))
# 'Paris.'
```

`tokenize=True` gives token ids, which a trace takes as a prompt just as well — and guarantees
the positions you computed are the positions the engine sees. Template controls differ per family
(`enable_thinking` is Qwen3's; `tok.chat_template` shows what yours accepts).

## Prefill an assistant turn

```python
messages = [
    {"role": "user", "content": "In one word, what is the capital of France?"},
    {"role": "assistant", "content": "The capital is"},
]
ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                              continue_final_message=True)

with model.trace(ids, temperature=0.0, max_tokens=4) as tracer:
    out = tracer.result.save()

print(repr(out.outputs[0].text))
# ' Paris.'
```

## Tokens

```python
ids = tok("Hello, world")["input_ids"]
print(ids, tok.convert_ids_to_tokens(ids), repr(tok.decode(ids)))
# [9707, 11, 1879] ['Hello', ',', 'Ġworld'] 'Hello, world'
print(tok.all_special_ids[:4], tok.eos_token)
# [151645, 151643, 151644, 151646] <|im_end|>
```

A string prompt is tokenized by the engine with the same tokenizer, so `tok(text)["input_ids"]`
is the prefill's row order.

## Where a message's tokens are

Offsets from the tokenizer, not length deltas of re-rendered prefixes — templates can rewrite
earlier turns. One contiguous `[start, end)` per message content, for pooling activations per
turn or steering only the user's words:

```python
messages = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "In one word, what is the capital of France?"},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
enc = tok(text, return_offsets_mapping=True)


def span(content):
    start = text.index(content)
    end = start + len(content)
    rows = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
    return rows[0], rows[-1] + 1


spans = {m["role"]: span(m["content"]) for m in messages}
print(spans, len(enc["input_ids"]))
# {'system': (3, 7), 'user': (12, 23)} 32

import nnsight

with model.trace(enc["input_ids"], temperature=0.0):
    resid = sum(model.model.layers[10].output)
    per_turn = nnsight.save({role: resid[a:b].mean(0) for role, (a, b) in spans.items()})

print({role: tuple(v.shape) for role, v in per_turn.items()})
# {'system': (4096,), 'user': (4096,)}
```

## What the model just wrote

`tracer.result.outputs[0].token_ids` are the generated ids in order; decode them one at a time
for per-token labels, and split on the family's markers for channels.

```python
messages = [{"role": "user", "content": "What is 2 + 2?"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)   # thinking on

with model.trace(text, temperature=0.0, max_tokens=512) as tracer:
    out = tracer.result.save()

ids = list(out.outputs[0].token_ids)
close = tok.convert_tokens_to_ids("</think>")
cut = ids.index(close) + 1 if close in ids else len(ids)     # no close tag: still thinking
print(len(ids), "tokens;", "reasoning:", repr(tok.decode(ids[:cut])[:60]))
# 280 tokens; reasoning: '<think>\nOkay, the user is asking "What is 2 + 2?" That\'s a b'
print("answer:", repr(tok.decode(ids[cut:]).strip()[:60]))
# answer: 'The sum of 2 and 2 is **4**. \n\n**Step-by-Step Explanation:**'
```

## Without a model

The tokenizer needs no weights and no GPU: `VLLM("Qwen/Qwen3-8B").tokenizer` on the meta tree,
or `AutoTokenizer.from_pretrained` directly.
