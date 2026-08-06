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
# # Patchscopes

# %% [markdown]
# ## Introduction

# %% [markdown]
# 🔎 **Patchscopes** turns the model itself into the decoder for its own hidden states. Instead
# of training a probe or projecting an activation through the unembedding, you *paste* the
# activation into a **different prompt** — one chosen so that the model, continuing normally,
# has to say out loud what the activation contains.
#
# The framework's contribution is noticing that a large family of interpretability methods are
# the same operation with different settings. An inspection is a pair:
#
# - a **source** $(S, i, M, \ell)$ — prompt $S$, position $i$, model $M$, layer $\ell$;
# - a **target** $(T, i^{*}, f, M^{*}, \ell^{*})$ — target prompt $T$, target position $i^{*}$,
#   mapping $f$, target model $M^{*}$, target layer $\ell^{*}$.
#
# Run $M$ on $S$, take the hidden state $h^{\ell}_{i}$, and run $M^{*}$ on $T$ with position
# $i^{*}$ of layer $\ell^{*}$ overwritten by $f(h^{\ell}_{i})$. Whatever $M^{*}$ generates is the
# readout. Defaults are $S = T$, $i = i^{*}$, $M = M^{*}$, $\ell = \ell^{*}$, $f = \mathrm{id}$.
#
# Different settings recover different known methods:
#
# | Method | Patchscopes configuration |
# |---|---|
# | Logit lens | $T = S$, $i^{*} = i$, $f = \mathrm{id}$, $\ell^{*} = L$ (last layer) |
# | Entity description | $T$ = few-shot descriptions ending in `x`, $i^{*}$ = the `x` position |
# | Zero-shot feature extraction | $T$ = a verbalized relation ending in `x` |
# | Cross-model inspection | $M^{*} \neq M$, with $f$ an affine map between the two spaces |
#
# 📗 Introduced in [*Patchscopes: A Unifying Framework for Inspecting Hidden Representations of
# Language Models*](https://arxiv.org/abs/2401.06102) (Ghandeharioun et al., ICML 2024).
#
# Two things make `nnsight` a natural fit. First, "overwrite position $i^{*}$ of layer
# $\ell^{*}$" is one assignment. Second, a Patchscope usually needs **many tokens** of
# generation, not one — descriptions are phrases — and `nnsight` patches during the prefill of a
# real `generate` call, so the KV cache carries the edit forward without any re-tokenize loop.

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
from nnsight import TransformersModel

# %% [markdown]
# We use `gemma-2-2b`. Patchscopes needs a model with enough capacity to follow the target
# prompt's pattern — `gpt2`-scale models mostly produce noise here. `google/gemma-2-2b` is a
# gated repository, so accept the license on the model page and log in with `huggingface-cli
# login` (or set `HF_TOKEN`) first.

# %%
model = TransformersModel("google/gemma-2-2b", dtype=torch.bfloat16,
                          device_map="auto", dispatch=True)

layers = model.model.layers
n_layers = model.config.num_hidden_layers

print(f"{n_layers} layers, d_model={model.config.hidden_size}")

# %% [markdown]
# ## The two halves of a Patchscope

# %% [markdown]
# Reading the source is one trace; writing it into the target is one assignment inside another.
# Everything else in the framework is a choice of arguments.

# %%
def source_representation(prompt, layer, position=-1):
    """h^ℓ_i — the residual stream at one position of one layer, for the source prompt."""
    with model.trace(prompt):
        hidden = layers[layer].output[0, position].save()
    return hidden

def patchscope(target_prompt, hidden, target_layer, target_position,
               f=lambda h: h, max_new_tokens=10):
    """Write f(h) into (target_layer, target_position) of the target prompt, then generate."""
    prompt_length = len(model.tokenizer(target_prompt).input_ids)
    with model.generate(target_prompt, max_new_tokens=max_new_tokens) as tracer:
        layers[target_layer].output[:, target_position, :] = f(hidden)
        generated = tracer.result.save()
    return model.tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True)

# %% [markdown]
# The patch lands during the prefill pass. Every generated token afterwards attends to the
# patched position through the KV cache, so a single assignment is enough — there is no need to
# re-run the prompt once per token the way a hook-and-retokenize loop would.

# %% [markdown]
# ## 1. The logit lens is a Patchscope

# %% [markdown]
# Set $T = S$, $i^{*} = i$, $f = \mathrm{id}$, $\ell^{*} = L$: paste layer $\ell$'s activations
# into the *last* layer, so the model's own final norm and unembedding decode them. That is
# exactly the [logit lens](../tutorials/probing/logit_lens.ipynb) — but expressed as a patch,
# which means the whole layer sweep is a batch of invokes inside **one forward pass**.

# %%
PROMPT = "Patchscopes is a nice tool to inspect the hidden representations of a language model"

with model.trace() as tracer:
    top_tokens = nnsight.save([])
    top_probs = nnsight.save([])
    for layer in range(n_layers):
        with tracer.invoke(PROMPT):
            layers[n_layers - 1].output[:] = layers[layer].output
            probs = model.output.logits[0].softmax(dim=-1)
            top_tokens.append(probs.argmax(dim=-1))
            top_probs.append(probs.max(dim=-1).values.float())

source_tokens = model.tokenizer.convert_ids_to_tokens(model.tokenizer(PROMPT).input_ids)
decoded = [[model.tokenizer.decode([t]) for t in row.tolist()] for row in top_tokens]
print(f"{'prompt':>6}  {' '.join(repr(t.replace(chr(9601), ' ')) for t in source_tokens[:8])}")
for layer in [0, 6, 12, 18, 25]:
    print(f"{layer:>6}  {' '.join(repr(t) for t in decoded[layer][:8])}")

# %%
import plotly.graph_objects as go

fig = go.Figure(
    data=go.Heatmap(
        z=[row.tolist() for row in top_probs][::-1],
        x=[f"{i}: {t}" for i, t in enumerate(source_tokens)],
        y=[f"layer {i}" for i in range(n_layers)][::-1],
        text=decoded[::-1],
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorscale="Blues",
        colorbar=dict(title="p(top token)"),
        hovertemplate="%{y}<br>%{x}<br>%{text} (p=%{z:.3f})<extra></extra>",
    )
)
fig.update_layout(
    title="Logit lens as a Patchscope (ℓ* = L, T = S, f = identity)",
    xaxis=dict(tickangle=-45),
    height=650,
    width=1000,
)
fig.show()

# %% [markdown]
# Two regimes are visible. Through roughly the first half the readout copies the *current*
# token — the residual stream is still dominated by what was embedded there. Above the middle
# it flips to the *next* token, and the confidence (colour) climbs. The layer where a column
# switches is where that position stops being about its own identity and starts being about the
# prediction.
#
# This is a faithful logit lens, and also a demonstration of its limitation: everything it can
# ever say is one token long, at whatever position it was read from. The rest of the framework
# is about lifting both restrictions.

# %% [markdown]
# ## 2. Entity description

# %% [markdown]
# The paper's question: *how does a model contextualize the tokens of an entity name, and at
# which layer is the entity fully resolved?*
#
# The target prompt is a few-shot list of `entity: description` pairs ending in a placeholder
# `x`. We patch the source entity's **last token** into the `x` position and let the model
# write the description. Now the readout can be a phrase, and it is decoded at a position whose
# context asks for an explanation rather than a continuation.

# %%
TARGET = (
    "Syria: Country in the Middle East, "
    "Leonardo DiCaprio: American actor, "
    "Samsung: South Korean multinational major appliance and consumer electronics "
    "corporation, x"
)
X_POSITION = len(model.tokenizer(TARGET).input_ids) - 1

ENTITY = "Diana, Princess of Wales"

print("unpatched:", repr(patchscope(TARGET, source_representation(TARGET, 0, X_POSITION),
                                    0, X_POSITION, max_new_tokens=14)))
print()
for source_layer in [6, 10, 14, 20]:
    hidden = source_representation(ENTITY, source_layer)
    for target_layer in [1, 4, 6]:
        description = patchscope(TARGET, hidden, target_layer, X_POSITION, max_new_tokens=14)
        print(f"  source L{source_layer:>2} → target L{target_layer}: {description!r}")

# %% [markdown]
# Left alone, the model reads `x` as the start of a new entry and invents one — `xQc: Canadian
# Twitch streamer`. With Diana's representation patched in, real facts about her come out of a
# target prompt that never mentions her: her name (`source L6 → target L1`), her title and the
# year of the wedding (`L10 → L6`: *"Princess of Wales, 1981-199…"*), her birth year
# (`L20 → L1`: *"1961: The year of her birth"*).
#
# The misses are as informative as the hits, and there are plenty — *"British fashion designer"*,
# *"English singer, songwriter, and actress"* — delivered with the same fluency. Two things are
# going on. The readout only works where the entity is *already* resolved in the source (deeper
# source layers) and where the target still has depth left to verbalize it (early target layers),
# and even inside that region a 2B base model is at the edge of what this needs. The paper runs
# Patchscopes on Vicuna-13B and GPT-J, and measures success rates over hundreds of entities
# rather than reading single generations — which is the right way to use this, since any single
# generation is fluent whether or not it is right.

# %% [markdown]
# ## 3. Zero-shot feature extraction

# %% [markdown]
# For a knowledge triple $(\sigma, \rho, \omega)$ — subject, relation, object — how much of
# $\omega$ can be extracted from the last-token representation of $\sigma$ alone, in a context
# that never mentions the relation?
#
# The source prompt mentions Apple and nothing about founders. The target prompt verbalizes the
# relation and ends at the placeholder. If the answer comes out, the subject representation
# carried it.

# %%
SOURCE = "The tech giant Apple"
RELATION = "The co-founder of x is"

x_position = model.tokenizer.convert_ids_to_tokens(
    model.tokenizer(RELATION).input_ids
).index("▁x")

print("unpatched:", repr(patchscope(RELATION, source_representation(RELATION, 0, x_position),
                                    0, x_position, max_new_tokens=8)))
print()
print(f"{'src':>4} | " + " | ".join(f"target L{tl:<26}" for tl in [0, 3, 6]))
for source_layer in range(0, n_layers, 2):
    hidden = source_representation(SOURCE, source_layer)
    row = [
        patchscope(RELATION, hidden, target_layer, x_position, max_new_tokens=8).replace("\n", " ")
        for target_layer in [0, 3, 6]
    ]
    print(f"{source_layer:>4} | " + " | ".join(f"{text[:34]!r:<36}" for text in row))

# %% [markdown]
# The answer is there, and where it is there is structured. Source layers **6–12** patched into
# target layer **6** give *"a man named Steve Jobs"* every time — from a source prompt that says
# only "The tech giant Apple" and a target prompt that never says "Apple". Above layer ~14 the
# readout collapses into generic Apple-news continuations (`' reportedly in talks to buy'`):
# the representation has stopped being *about Apple* and become about what comes next in *its
# own* sentence, which is not what the target prompt is asking for.
#
# The very early source layers (2–4) also emit Jobs and Wozniak, but erratically and mostly into
# target layer 0 — at that depth the patched vector is still close to the token embedding of
# "Apple", so the target model is doing the retrieval itself rather than reading it out. The
# region that means something is the stable one in the middle.
#
# This is why Patchscopes is run as a **grid** over $(\ell, \ell^{*})$: a single layer pair tells
# you almost nothing, and the shape of the region is the result.

# %% [markdown]
# ## Choosing a mapping function

# %% [markdown]
# Everything above used $f = \mathrm{id}$. The `patchscope` helper takes `f` as an argument
# because the interesting variations are all mappings: scaling the activation to the target
# layer's typical norm, projecting through a learned affine map when $M^{*} \neq M$ (the paper's
# cross-model setting), or zeroing a direction before the readout to ask what the representation
# says *without* it.
#
# It is also a quick sanity check that the readout depends on the vector rather than on the
# target prompt doing the work by itself:

# %%
hidden = source_representation(SOURCE, 8)

for name, f in [
    ("identity", lambda h: h),
    ("×2", lambda h: h * 2),
    ("×0.5", lambda h: h * 0.5),
]:
    print(f"{name:>8}: {patchscope(RELATION, hidden, 6, x_position, f=f, max_new_tokens=8)!r}")

# %% [markdown]
# Scaling the activation up by 2 destroys the readout and scaling it down by half degrades it to
# the wrong co-founder — the patched norm matters, and matching it to the target layer's typical
# scale is the first thing to try when a Patchscope returns nothing.

# %% [markdown]
# ## Conclusion

# %% [markdown]
# 🎉 Patchscopes is a small amount of machinery — read one activation, write it somewhere else,
# keep generating — around a large idea: the model is already a competent decoder of its own
# representations, if you ask it in a context where answering is the natural continuation.
#
# In `nnsight` the source half is `layers[ℓ].output[0, i]` and the target half is an assignment
# inside `model.generate`, which means the configuration table at the top of this notebook maps
# one-to-one onto arguments you pass, not onto separate implementations.
#
# Related: [Logit Lens](../tutorials/probing/logit_lens.ipynb) (the $\ell^{*} = L$ corner of the
# framework), [Activation Patching](../tutorials/causal_mediation_analysis/activation_patching.ipynb)
# (patching to measure causal effect rather than to decode), and
# [Multiple Token Generation](../../features/4_multiple_token.ipynb) for the generation API.

# %% [markdown]
# ## References
#
# - Ghandeharioun, Caciularu, Pearce, Dixon, Geva, [*Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models*](https://arxiv.org/abs/2401.06102), ICML 2024
# - nostalgebraist, [*Interpreting GPT: the Logit Lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), 2020
# - Adapted from the TransformerLens [Patchscopes & Generation with Patching](https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/Patchscopes_Generation_Demo.ipynb) demo
