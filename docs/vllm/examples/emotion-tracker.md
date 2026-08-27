# Concept directions during generation

A replication of the emotion-concepts recipe (Sofroniew et al., 2026): build one direction per
emotion from stories written to express it, project out what neutral stories share, then watch
each generated token's projection onto those directions as the model writes.

The stories are `ryancodrai/emotion-probes` on the Hub; this page uses 100 per emotion rather
than the paper's 1000.

## Data

```python
import torch
import pandas as pd
import nnsight
from huggingface_hub import hf_hub_download
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True, enable_prefix_caching=False)
tok = model.tokenizer
L = 20
EMOTIONS = ["anxious", "amused", "desperate", "proud", "defiant"]
N = 100
SKIP = 50                      # positions to skip: the story's preamble

stories = pd.read_parquet(hf_hub_download("ryancodrai/emotion-probes", "expression/stories.parquet",
                                          repo_type="dataset"))
neutral = pd.read_parquet(hf_hub_download("ryancodrai/emotion-probes", "expression/neutral_stories.parquet",
                                          repo_type="dataset"))
texts = {e: stories[stories.emotion == e].story.tolist()[:N] for e in EMOTIONS}
neutral_texts = neutral.story.tolist()[:N]
print({e: len(t) for e, t in texts.items()}, len(neutral_texts))
# {'anxious': 100, 'amused': 100, 'desperate': 100, 'proud': 100, 'defiant': 100} 100
```

## Mean activations, one installed block

The block runs on every request the engine serves; the mean over positions from `SKIP` on
rides each output. Prompt-only, so `max_tokens=1`.

```python
def mean_activations(prompts):
    with model.edit() as (tracer, edit):
        h = sum(model.model.layers[L].output)
        act = h[min(SKIP, h.shape[0] - 1):].mean(0).save()
    try:
        outputs = model.generate(prompts, temperature=0.0, max_tokens=1)
    finally:
        edit.clear()
    return torch.stack([o.saves["act"].float().cpu() for o in outputs])        # [n, d_model]


emotion_acts = {e: mean_activations(t) for e, t in texts.items()}
neutral_acts = mean_activations(neutral_texts)
print({e: tuple(a.shape) for e, a in emotion_acts.items()})
# {'anxious': (100, 4096), 'amused': (100, 4096), 'desperate': (100, 4096), 'proud': (100, 4096), 'defiant': (100, 4096)}
```

## Directions

Per-emotion mean minus the grand mean, with the top principal components of the neutral stories
(50% of their variance) projected out.

```python
means = {e: a.mean(0) for e, a in emotion_acts.items()}
grand = torch.stack(list(means.values())).mean(0)
_, S, Vt = torch.linalg.svd(neutral_acts - neutral_acts.mean(0), full_matrices=False)
k = int(((S ** 2).cumsum(0) / (S ** 2).sum() < 0.5).sum()) + 1
P = Vt[:k]
directions = {e: (m - grand) - ((m - grand) @ P.T) @ P for e, m in means.items()}
V = torch.stack([directions[e] for e in EMOTIONS]).cuda()                       # [5, d_model]
print(f"projected out {k} neutral components")
# projected out 8 neutral components
```

## Track, token by token

The projections are computed on the worker every step; only `[5]` numbers per token come home.

```python
messages = [{"role": "user", "content": "Hi, what's on your mind? Write a short poem about it."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)

with model.trace(prompt, temperature=0.0, max_tokens=40) as tracer:
    proj = list().save()
    for _ in tracer.iter[:40]:
        h = sum(model.model.layers[L].output)[-1].float()              # the running token
        proj.append(h @ V.T)
    out = tracer.result.save()

proj = torch.stack(proj)[1:].cpu()                                     # drop the prefill row
scale = proj.abs().flatten().float().quantile(0.99)
ids = out.outputs[0].token_ids
print(repr(out.outputs[0].text))
print(f"{'token':>12} " + " ".join(f"{e:>9}" for e in EMOTIONS))
for i, t in enumerate(ids[:len(proj)]):
    print(f"{tok.decode(t)!r:>12} " + " ".join(f"{v / scale:9.2f}" for v in proj[i].tolist()))
# "Hi! I'm just a thinking machine, but I love exploring ideas and creativity. Today, I'm thinking
#  about the quiet moments between thoughts—those spaces where imagination can bloom. ..."
#        token   anxious    amused desperate     proud   defiant
#         'Hi'     -0.11      0.07     -0.55      0.52      0.06
#          '!'     -0.12      0.05     -0.58      0.67     -0.01
#         ' I'     -0.01      0.02     -0.69      0.73     -0.05
#     ...
#      ' love'     -0.19      0.02     -0.63      0.97     -0.17
# ' exploring'      0.05     -0.06     -0.73      1.02     -0.28
#     ...
#     ' quiet'      0.31     -0.01     -0.97      1.00     -0.32
#   ' moments'      0.46     -0.18     -0.86      1.08     -0.49
#   ' between'      0.47     -0.21     -0.60      0.72     -0.38
#  ' thoughts'      0.53     -0.17     -0.62      0.64     -0.38
#    ' spaces'      0.66     -0.56     -0.50      0.79     -0.38
```

The reply runs *proud* throughout and *anxious* climbs exactly where it turns to "the quiet
moments between thoughts".

The paper's interactive visualization colours each token by its strongest emotion; the table is
the same data.
