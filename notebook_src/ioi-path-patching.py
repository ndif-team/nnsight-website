# %% [markdown]
# <details>
# <summary><b>Info</b></summary>
#
# **Last Execution:** 2026-08-04
#
# | Package | Version |
# |---------|---------|
# | **nnsight** | **0.8** |
# | Python | 3.12.13 |
# | torch | 2.13.0+cu126 |
# | transformers | 5.15.0 |
#
# </details>

# %% [markdown]
# # Path Patching the IOI Circuit

# %% [markdown]
# ## Introduction

# %% [markdown]
# 🔌 Activation patching finds *which components matter*. **Path patching** finds *which
# components talk to each other*.
#
# When you patch a head's output, the patched value flows everywhere downstream at once — into
# every later head's queries, keys and values, into every MLP, and directly into the logits. A
# large effect tells you the head is load-bearing, but not what it is load-bearing *for*. Path
# patching restricts the patch to a single **edge**: the sender's contribution reaches one
# receiver, through one of its three input pathways, and every other component's view of the
# sender is left exactly as it was.
#
# We reproduce the edge structure of the **Indirect Object Identification** circuit:
#
# - Clean: *"When Mary and John went to the store, **John** gave a drink to"* → **Mary**
# - Corrupted: *"When Mary and John went to the store, **Mary** gave a drink to"* → **John**
#
# The circuit as described in the paper: **duplicate-token** and **induction** heads detect that
# one name appears twice; **S-inhibition** heads write a signal saying *which* name is the
# repeated subject; **name-mover** heads read that signal and copy the *other* name to the
# output. The claim we will test at the edge level is the arrow in the middle — that the
# S-inhibition heads reach the name movers specifically through their **queries**.
#
# 📗 Circuit from [*Interpretability in the Wild: a Circuit for Indirect Object Identification in
# GPT-2 small*](https://arxiv.org/abs/2211.00593) (Wang et al., ICLR 2023). The path-patching
# technique is developed in [*Localizing Model Behavior with Path
# Patching*](https://arxiv.org/abs/2304.05969) (Goldowsky-Dill et al., 2023).

# %% [markdown]
# ## Setup

# %% [markdown]
# If using Colab, install NNsight:
# ```
# !pip install -U nnsight
# ```

# %%
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    !pip install -U nnsight

# %%
import torch
import nnsight
import plotly.express as px
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", device_map="auto", dispatch=True)

blocks = model.transformer.h
n_layers = model.config.n_layer
n_heads = model.config.n_head
d_model = model.config.n_embd
d_head = d_model // n_heads

print(f"gpt2: {n_layers} layers x {n_heads} heads, d_head={d_head}")

# %% [markdown]
# ## The task and the metric

# %%
CLEAN = "When Mary and John went to the store, John gave a drink to"
CORRUPTED = "When Mary and John went to the store, Mary gave a drink to"

IO_TOKEN = model.tokenizer.encode(" Mary")[0]     # the indirect object — correct answer
S_TOKEN = model.tokenizer.encode(" John")[0]      # the subject — the distractor

def logit_diff(logits):
    """logit(Mary) - logit(John) at the final position. Positive = correct behaviour."""
    return logits[0, -1, IO_TOKEN] - logits[0, -1, S_TOKEN]

with model.trace(CLEAN):
    clean_ld = logit_diff(model.output.logits).item().save()

with model.trace(CORRUPTED):
    corrupted_ld = logit_diff(model.output.logits).item().save()

print(f"clean logit diff     : {clean_ld:+.3f}   (predicts Mary)")
print(f"corrupted logit diff : {corrupted_ld:+.3f}   (predicts John)")

def recovery(ld):
    """0 = corrupted behaviour, 1 = clean behaviour."""
    return (ld - corrupted_ld) / (clean_ld - corrupted_ld)

# %% [markdown]
# Swapping which name is repeated flips the answer, and the two runs differ only at one token —
# so every patch we make below is measured against a well-matched baseline.
#
# All experiments run in the **denoising** direction: the base run is corrupted, we splice
# *clean* values in, and `recovery` reports how much of the correct behaviour comes back.

# %% [markdown]
# ## Node level: which heads matter

# %% [markdown]
# In GPT-2 the per-head attention results are concatenated and fed to `attn.c_proj`, so
# `attn.c_proj.input` is the stacked $z$ for all heads, and head $h$ owns channels
# `[h*d_head : (h+1)*d_head]`. Patching that slice replaces one head's output — and
# **everything downstream sees the patched value**.
#
# The whole 12 × 12 sweep is one forward pass: one `tracer.invoke` per head.

# %%
def head_channels(head):
    return slice(head * d_head, (head + 1) * d_head)

with model.trace(CLEAN):
    clean_z = nnsight.save([blocks[layer].attn.c_proj.input for layer in range(n_layers)])

with model.trace() as tracer:
    node_results = nnsight.save([])
    for layer in range(n_layers):
        for head in range(n_heads):
            with tracer.invoke(CORRUPTED):
                blocks[layer].attn.c_proj.input[:, :, head_channels(head)] = \
                    clean_z[layer][:, :, head_channels(head)]
                node_results.append(logit_diff(model.output.logits))

node_scores = torch.tensor([recovery(r.item()) for r in node_results]).reshape(n_layers, n_heads)

fig = px.imshow(
    node_scores,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    labels=dict(x="head", y="layer", color="recovery"),
    title="Patching each head's output (all downstream paths)",
    height=500,
)
fig.show()

top = node_scores.flatten().topk(8)
for value, index in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  ({index // n_heads:2d}, {index % n_heads:2d})  {value:+.3f}")

# %% [markdown]
# The named cast of the IOI circuit shows up unprompted: the S-inhibition heads **(8,10)**,
# **(7,3)**, **(7,9)**, **(8,6)**, the induction head **(5,5)**, the name mover **(9,9)**, and
# the duplicate-token head **(3,0)**.
#
# And that is as far as node-level patching goes. It cannot tell us whether (8,6) matters
# *because* it feeds (9,9), or because it feeds an MLP, or because it writes to the logits
# directly. For that we have to cut the graph by edges.

# %% [markdown]
# ## Edge level: direct path patching

# %% [markdown]
# A head's contribution to the residual stream is its own slice of $z$ mapped through its slice
# of $W_O$:
#
# $$\text{contribution}_{(\ell,h)} = z_{(\ell,h)}\, W_O^{(\ell,h)}$$
#
# To patch the single edge $(\ell_s, h_s) \rightarrow (\ell_d, h_d).\text{q}$ we
#
# 1. take $\Delta$ = (clean − corrupted) contribution of the **sender**,
# 2. recompute the **receiver block's** q/k/v from `resid + Δ` — the residual stream as it
#    *would* have been if only the sender had been patched,
# 3. write back **only** the receiver head's slice of **only** the chosen pathway.
#
# Everything else in the model — the sender's real output, every other head's queries, all the
# MLPs — still sees the unpatched corrupted stream. `nnsight` makes step 2 direct: modules can be
# called ad hoc on any input inside a trace, so `ln_1` and `c_attn` are reused rather than
# reimplemented.

# %%
W_O = [blocks[layer].attn.c_proj.weight for layer in range(n_layers)]
QKV_OFFSET = {"q": 0, "k": d_model, "v": 2 * d_model}

def head_contribution(z, layer, head):
    """One head's additive write into the residual stream."""
    return z[:, :, head_channels(head)] @ W_O[layer][head_channels(head), :]

def patch_edge(sender, receiver, component):
    """Splice the clean sender contribution into one receiver head's q, k, or v only."""
    sender_layer, sender_head = sender
    receiver_layer, receiver_head = receiver

    delta = (
        head_contribution(clean_z[sender_layer], sender_layer, sender_head)
        - head_contribution(blocks[sender_layer].attn.c_proj.input, sender_layer, sender_head)
    )
    residual = blocks[receiver_layer].ln_1.input
    patched_qkv = blocks[receiver_layer].attn.c_attn(
        blocks[receiver_layer].ln_1(residual + delta)
    )
    offset = QKV_OFFSET[component]
    channels = slice(offset + receiver_head * d_head, offset + (receiver_head + 1) * d_head)
    blocks[receiver_layer].attn.c_attn.output[:, :, channels] = patched_qkv[:, :, channels]

# %% [markdown]
# ### Who writes to a name mover's query?
#
# Rather than assume the S-inhibition heads, sweep **every** upstream head into the query of
# name mover (9,9) and see what survives.

# %%
NAME_MOVER = (9, 9)

query_scores = torch.zeros(NAME_MOVER[0], n_heads)
for sender_layer in range(NAME_MOVER[0]):
    with model.trace() as tracer:
        results = nnsight.save([])
        for sender_head in range(n_heads):
            with tracer.invoke(CORRUPTED):
                patch_edge((sender_layer, sender_head), NAME_MOVER, "q")
                results.append(logit_diff(model.output.logits))
    for sender_head in range(n_heads):
        query_scores[sender_layer, sender_head] = recovery(results[sender_head].item())

fig = px.imshow(
    query_scores,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    labels=dict(x="sender head", y="sender layer", color="recovery"),
    title=f"Direct path into the query of name mover {NAME_MOVER}",
    height=450,
)
fig.show()

top = query_scores.flatten().abs().topk(6)
print("strongest direct paths into (9,9).q:")
for index in top.indices.tolist():
    layer, head = index // n_heads, index % n_heads
    print(f"  ({layer:2d}, {head:2d})  {query_scores[layer, head]:+.4f}")

# %% [markdown]
# Four heads, and they are exactly the four S-inhibition heads named in the paper — **(8,6)**,
# **(8,10)**, **(7,9)**, **(7,3)** — clearing the next-best head by a factor of three. The
# induction head (5,5) and the duplicate-token head (3,0), both clearly important at the node
# level, contribute essentially **nothing** through this edge: they act on the S-inhibition
# heads, not on the name mover.
#
# That distinction is the entire reason to do path patching. The node-level heatmap ranks (5,5)
# above (8,6); the edge-level sweep says (5,5) is upstream of the arrow we are looking at, not
# on it.

# %% [markdown]
# ### Through which pathway?
#
# The same construction, holding the sender/receiver pairs fixed and varying the input pathway.
# If the S-inhibition signal is telling name movers *where to look*, it should arrive on the
# **query** and not on the key or value.

# %%
S_INHIBITION = [(7, 3), (7, 9), (8, 6), (8, 10)]
NAME_MOVERS = [(9, 9), (9, 6), (10, 0)]

pathway_scores = {}
for component in ["q", "k", "v"]:
    with model.trace() as tracer:
        results = nnsight.save([])
        for sender in S_INHIBITION:
            for receiver in NAME_MOVERS:
                with tracer.invoke(CORRUPTED):
                    patch_edge(sender, receiver, component)
                    results.append(logit_diff(model.output.logits))
    pathway_scores[component] = [recovery(r.item()) for r in results]

print(f"{'edge':>18} | {'q':>8} | {'k':>8} | {'v':>8}")
print("-" * 52)
for index, (sender, receiver) in enumerate(
    [(s, r) for s in S_INHIBITION for r in NAME_MOVERS]
):
    row = "  ".join(f"{pathway_scores[c][index]:+8.4f}" for c in ["q", "k", "v"])
    print(f"{str(sender) + ' -> ' + str(receiver):>18} |  {row}")

# %% [markdown]
# The queries carry everything. Keys and values are at the 10⁻⁴ level — numerically zero next to
# the query column — for all twelve edges. This is the mechanism Wang et al. describe: the
# S-inhibition heads do not move name information themselves, they change *what the name movers
# attend to*, and attention destinations are set by queries.
#
# The strongest single edge is **(8,6) → (9,9).q**, with (8,10) → (9,9).q close behind. The
# edges are not uniform: (8,10) feeds all three name movers, while (8,6) is specific to (9,9) —
# its path into (9,6) is slightly *negative*. Individual heads have individual jobs, which is the
# kind of statement node-level patching cannot make.

# %% [markdown]
# ## Caveats

# %% [markdown]
# Worth stating plainly, because circuit results are easy to overclaim:
#
# - **One prompt pair.** Wang et al. average over a generated IOI dataset with counterbalanced
#   names and templates. A single pair is enough to see the structure and not enough to
#   establish it — the numbers move with the names you pick.
# - **"Direct path" here means one hop.** We patch the sender straight into the receiver's
#   pathway. A sender that reaches the receiver *via* an intervening MLP is not counted, which
#   is what makes the edge attribution clean but also what makes these numbers small relative to
#   the node-level effects.
# - **Absolute values are metric-dependent.** Recovery fractions depend on the normalization and
#   on the patching direction; other implementations (including TransformerLens's
#   `get_act_patch_direct_path`, which also runs on weight-processed models) report different
#   magnitudes for the same edges. What reproduces across implementations is the ranking and the
#   q/k/v split.

# %% [markdown]
# ## Conclusion

# %% [markdown]
# 🎉 Path patching is activation patching plus an accounting of where the patched value is
# allowed to go, and in `nnsight` that accounting is written directly: take one head's slice of
# `c_proj.input`, map it through its slice of $W_O$, re-run `ln_1` and `c_attn` on the perturbed
# residual, and assign into one head's slice of one pathway. No hook-name bookkeeping, and the
# whole sweep batches into a handful of forward passes.
#
# Related: [Activation Patching](../tutorials/causal_mediation_analysis/activation_patching.ipynb)
# for the node-level method, [Attribution Patching](../tutorials/causal_mediation_analysis/attribution_patching.ipynb)
# for the gradient approximation that scales this to every edge at once, and
# [Batching](../../features/8_batching.ipynb) for the `tracer.invoke` pattern used throughout.

# %% [markdown]
# ## References
#
# - Wang, Variengien, Conmy, Shlegeris, Steinhardt, [*Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*](https://arxiv.org/abs/2211.00593), ICLR 2023
# - Goldowsky-Dill, MacLeod, Sato, Arora, [*Localizing Model Behavior with Path Patching*](https://arxiv.org/abs/2304.05969), 2023
# - Adapted from the TransformerLens [Direct Path Patching](https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/direct_path_patching_ioi.ipynb) demo
