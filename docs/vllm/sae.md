# SAE features

A sparse autoencoder turns a residual into a few named features. Reading them needs the whole
SAE; *acting* on one needs only its two rows — an encoder row to compute the feature's activation
from the live residual, a decoder row to add or remove its contribution — so the intervention
runs in the worker on every step with a few kilobytes shipped, and the decision (is the feature
on, how strongly) is made from the model's actual state rather than fixed in advance.

The SAE here is [Qwen-Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50) for
layer 24: TopK-50, 65,536 features, trained on the residual stream leaving the block of
`Qwen3-8B-Base` (its card says applying it to the post-trained checkpoint is reasonable; the
reconstruction error below is the price).

## Find the features: full SAE, once, on the client

```python
import torch
from huggingface_hub import hf_hub_download
from nnsight.modeling.vllm import VLLM

LAYER, K = 24, 50
sae = torch.load(hf_hub_download("Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", f"layer{LAYER}.sae.pt"),
                 map_location="cuda")
W_enc, W_dec, b_enc, b_dec = sae["W_enc"], sae["W_dec"], sae["b_enc"], sae["b_dec"]   # (65536, 4096), (4096, 65536)

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
PROMPT = "The capital of France is"

with model.trace(PROMPT, temperature=0.0):
    resid = sum(model.model.layers[LAYER].output).save()        # the stream leaving block 24, [pos, 4096]

pre = resid.float() @ W_enc.T + b_enc                            # the encoder takes the raw residual
top = pre.topk(K, dim=-1)
acts = torch.zeros_like(pre).scatter_(-1, top.indices, top.values)
recon = acts @ W_dec.T + b_dec
print([round(e, 3) for e in ((recon - resid.float()).norm(dim=-1) / resid.float().norm(dim=-1)).tolist()])
# [0.291, 0.363, 0.421, 0.467, 0.488]
last = acts[-1]
feats = last.nonzero().flatten()
feats = feats[last[feats].argsort(descending=True)][:8].tolist()
print([(f, round(last[f].item(), 1)) for f in feats])
# [(16957, 42.8), (51672, 35.2), (41823, 31.4), (37656, 23.4), (25446, 21.3), (32674, 20.5), (61073, 20.0), (53363, 18.4)]
```

`sum(layers[24].output)` is the hook point the SAE was trained on (the block's output, as a
HuggingFace forward hook sees it). One trace, one tensor home; the 2 GB SAE never leaves the
client.

## Ablate each feature in the worker

Two rows per feature ship with the block. The feature's activation is recomputed from the live
residual at the last position, and its decoder direction is subtracted in proportion — nothing
here is a fixed vector.

```python
def rows(f):
    return W_enc[f].to(torch.bfloat16), b_enc[f].item(), W_dec[:, f].to(torch.bfloat16)

paris = model.tokenizer.encode(" Paris")[0]
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    with tracer.invoke(PROMPT):
        base = model.logits.save()
    for f in feats:
        w_in, b_in, w_out = rows(f)
        with tracer.invoke(PROMPT):
            hidden, residual = model.model.layers[LAYER].output
            x = hidden[-1] + residual[-1]
            a = torch.relu(x @ w_in.to(x.device) + b_in)       # this feature, on this forward
            hidden[-1] -= a * w_out.to(x.device)               # remove what it contributes
            logits = model.logits.save()

p = lambda l: round(l.float().softmax(-1)[0, paris].item(), 3)
print(p(base), [(f, p(l)) for f, l in zip(feats, logits)])
# 0.537 [(16957, 0.351), (51672, 0.431), (41823, 0.499), (37656, 0.402), (25446, 0.491), (32674, 0.649), (61073, 0.56), (53363, 0.547)]
```

Feature 16957 — the most active at ` is` — carries a third of `P(Paris)` on its own; 37656
and 51672 carry some; 32674 *suppresses* it.

## Clamp a feature through generation

Scale the feature's activation to a target multiple, at every step, from its live value.

```python
w_in, b_in, w_out = rows(16957)
for target in (0.0, 3.0):
    with model.trace(PROMPT, temperature=0.0, max_tokens=8) as tracer:
        seen = list().save()
        for _ in tracer.iter[:8]:
            hidden, residual = model.model.layers[LAYER].output
            x = hidden[-1] + residual[-1]
            a = torch.relu(x @ w_in.to(x.device) + b_in)
            seen.append(round(a.item(), 1))
            hidden[-1] += (target - 1) * a * w_out.to(x.device)   # a -> target * a
        out = tracer.result.save()
    print(target, seen, repr(out.outputs[0].text))
# 0.0 [42.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0] ' Paris. The capital of France is Paris'
# 3.0 [42.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 34.5] ' Paris. The capital of Italy is Rome'
```

The feature fires on the prompt's ` is` and again on the generated ` is` seven steps later, and
nowhere between — `seen` is the feature's own timeline, read for free while acting on it.
Removing it changes what follows the second ` is`; tripling it leaves the model's own
continuation alone.

## Rules

- **Encoder on the raw residual**, as the SAE's card does — no `b_dec` subtraction before
  encoding for this family; check yours.
- **Base-model SAE, post-trained model.** The reconstruction error (0.29–0.49 by position)
  reflects that mismatch and the 50-feature budget; use the features as handles, not as ground
  truth.
- **Whole-SAE reads stay on the client.** Shipping a 2 GB `W_enc` with a block would serialise
  it on every request; ship rows.
- **Under tensor parallelism** the residual is whole on every rank and the write is re-sharded,
  so the same block runs unchanged ([Tensor parallelism](tensor-parallel.md)).
