# Direct logit attribution

How much each component *wrote* toward the answer, read off in one forward. The final residual
stream is a sum of every attention and MLP output; after the final norm, the logit difference
between two tokens is linear in that sum, so each component's share is its output projected onto
the difference of the two unembedding rows. All of it is computed in the worker, where the
weights are; only the per-component scores come home.

## Per layer

```python
import torch
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
tok = model.tokenizer
PROMPT = "The capital of France is"
paris, berlin = tok.encode(" Paris")[0], tok.encode(" Berlin")[0]

with model.trace(PROMPT, temperature=0.0):
    attn, mlp = [], []
    for layer in model.model.layers:                      # forward order: attention, then MLP
        attn.append(layer.self_attn.output[-1].clone())   # last position, [4096]
        mlp.append(layer.mlp.output[-1].clone())
    final = model.model.norm.output[1][-1]                # the un-normed final residual
    scale = model.model.norm.weight.float() / final.float().pow(2).mean().sqrt()   # RMSNorm, linearised
    direction = (model.lm_head.weight[paris] - model.lm_head.weight[berlin]).float()
    dla = lambda x: (x.float() * scale) @ direction
    attn_dla = torch.stack([dla(x) for x in attn]).save()
    mlp_dla = torch.stack([dla(x) for x in mlp]).save()
    total = dla(final).save()
    logits = model.logits.save()

print(round(total.item(), 2), round((logits[0, paris] - logits[0, berlin]).item(), 2))
# 7.03 7.06
print([round(x, 1) for x in attn_dla.tolist()])
# [0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, -0.0, -0.0, -0.0,
#  -0.0, 0.1, 0.0, 0.1, 0.5, -0.1, 0.3, 0.0, -0.1, 0.1, 0.0, 2.6, 0.1, 0.1, 0.2, 0.6]
print([round(x, 1) for x in mlp_dla.tolist()])
# [-0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.1, -0.1, 0.0, -0.0, 0.0, 0.0, -0.2, -0.1, 0.1, 0.1, -0.0,
#  -0.1, -0.1, -0.1, 0.0, 0.0, 0.1, 0.0, 0.7, 0.2, 1.6, 1.3, 1.7, -1.1, 0.1, -0.2, -1.7]
```

The decomposition reproduces the model's own logit difference (7.03 against 7.06; the gap is
bf16). The answer is written late: attention at layer 31 contributes 2.6 on its own, the MLPs of
layers 27–31 add 0.7 to 1.7 each, and the last MLPs push *against* it (−1.1, −1.7) — the same
layers that [ablation](ablation.md) found to *raise* `P(Paris)` when removed.

`scale` linearises the final RMSNorm around this forward: the norm divides by the residual's
RMS, which is a single number per position once the forward has run, so every component's
contribution is its output times that number, elementwise by the norm's weight. Read the final
residual from `model.model.norm.output[1]` (the norm returns `(normed, residual)`).

## Per head

`o_proj.input` is `z`, every head's output before `W_O`; a head's contribution to the residual is
its slice of `z` through its slice of `W_O`.

```python
L, d = 24, 128
with model.trace(PROMPT, temperature=0.0):
    z = model.model.layers[L].self_attn.o_proj.input[-1]
    W_O = model.model.layers[L].self_attn.o_proj.weight                 # [4096, 4096]
    final = model.model.norm.output[1][-1]
    scale = model.model.norm.weight.float() / final.float().pow(2).mean().sqrt()
    direction = (model.lm_head.weight[paris] - model.lm_head.weight[berlin]).float()
    heads = torch.stack([
        ((z[h * d:(h + 1) * d] @ W_O[:, h * d:(h + 1) * d].T).float() * scale) @ direction
        for h in range(32)
    ]).save()

print([round(x, 2) for x in heads.tolist()])
# [-0.02, 0.02, 0.01, -0.0, -0.0, 0.01, 0.01, -0.02, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.01, -0.01,
#  0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.15, 0.03, 0.63, -0.06, -0.0, 0.01, -0.0, -0.0]
```

Head 26 of layer 24 writes 0.63 of the difference directly. [Patching](patching.md#one-head)
shows the same head is *sufficient* to move the answer from the subject to the last position
(+9.2 when patched in): it is a mover, and most of what it moves is written to the logits by
later layers, not by the head itself. Direct attribution measures the last hop only.

## Rules

- **Read in forward order.** Attention before MLP within a layer, layer by layer; the final norm
  after all of them. Reading `layers[0].mlp.output` after `layers[35].self_attn.output` raises
  `OutOfOrderError`.
- **The lens is a different question.** Sending a *residual* through the unembed
  ([Logit lens](logit-lens.md)) asks what the stream says so far; attribution asks what each
  component added. Both use `model.model.norm` and `model.lm_head` in the worker.
- **Under tensor parallelism** `o_proj.weight` is this rank's shard, so the per-head form above
  is single-GPU; per-layer attribution reads only gathered activations and works at any degree.
