---
date: 2026-02-26
authors:
  - ndif-team
categories:
  - Ecosystem
---

# NNterp Integration

We welcome `nnterp` to the NDIF ecosystem! The `nnterp` library is built on top of `nnsight`, providing standardized transformer architecture for many LLMs and implementations of common interpretability techniques. Let's explore how it works!

<!-- more -->

## `nnterp`: One Interface to Rule All Transformers

If you've ever tried to write `nnsight` code that works across different transformer models, you know that it can be a pain. GPT-2 calls its layers `transformer.h`, LLaMA uses `model.layers`, while OPT does something else entirely. Every time you switch models, you're rewriting plumbing instead of doing research.

**`nnterp`** solves this with a standardized interface for transformer internals, built on top of [NNsight](https://nnsight.net/) and the [NDIF](https://ndif.us) ecosystem. It gives you a single, consistent API for accessing and intervening on any supported transformer architecture without reimplementing the models themselves.

## The Core Idea: `StandardizedTransformer`

At the heart of `nnterp` is `StandardizedTransformer`, a wrapper that maps every supported model onto a common naming convention inspired by LLaMA's architecture:

```
StandardizedTransformer
├── embed_tokens
├── layers
│   ├── self_attn
│   └── mlp
├── ln_final
└── lm_head
```

Loading a model is straightforward — whether it's GPT-2, LLaMA, Gemma, or any other supported architecture:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("gpt2")
# or
model = StandardizedTransformer("meta-llama/Llama-2-7b-hf")
```

Once loaded, you get immediate access to useful model properties:

```python
print(model.num_layers)   # Number of transformer layers
print(model.hidden_size)  # Dimensionality of hidden states
print(model.num_heads)    # Number of attention heads
print(model.vocab_size)   # Vocabulary size
```

No more digging through config files or `model.named_modules()` to figure out how a particular architecture is wired.

## Accessing Internals with a Unified API

The real power of the standardized interface shows up when you start probing model internals. Within an NNsight tracing context, you can access layer inputs and outputs using a consistent indexing scheme:

```python
with model.trace("The Eiffel Tower is in the city of"):
    # Layer outputs — same syntax for every model
    layer_5_output = model.layers_output[5]

    # Attention and MLP outputs at a specific layer
    attn_output = model.attentions_output[3]
    mlp_output = model.mlps_output[3]

    # Built-in logit access
    logits = model.logits.save()
```

This is the same code regardless of whether the underlying model is GPT-2, LLaMA, or Gemma. The naming convention chaos is handled for you.

## Connecting to NDIF Backend

With the release of `nnsight 0.6`, you can now use `StandardizedTransformer` to experiment on models hosted on the remote NDIF (National Deep Inference Fabric) backend! [NDIF](https://ndif.us) is a research computing platform that lets you run intervention code on large models hosted remotely.

All you need to do is [sign up for an API key](https://login.ndif.us/), configure NNsight for remote access, and add `remote=True` to your tracing context:

### 1. Get your API key at [login.ndif.us](https://login.ndif.us/).
### 2. Add API Key to NNsight CONFIG

```python
from nnsight import CONFIG

CONFIG.API.APIKEY = input("Enter your API key: ")
```

### 3. Run your `nnterp` experiment remotely on NDIF

```python
from nnterp import StandardizedTransformer
import nnterp

model = StandardizedTransformer("meta-llama/Llama-3.1-70B")

with model.trace("hello", remote=True):
    layer_5_output = model.layers_output[5]
    model.layers_output[10] = layer_5_output
```

No GPUs needed!

## Built-in Interventions

nnterp doesn't just standardize naming. It also packages common interpretability interventions so you don't have to rewrite them for every experiment.

### Skip Layers

Want to ablate a layer or a range of layers? There's a clean API for that:

```python
with model.trace("Hello world"):
    model.skip_layer(1)        # Zero out layer 1's contribution
    model.skip_layers(2, 3)    # Skip layers 2 through 3
```

You can also skip layers while substituting in saved activations, which is useful for causal tracing and circuit analysis:

```python
import torch

# First, save activations from a clean run
with model.trace("Hello world") as tracer:
    layer_6_out = model.layers_output[6].save()
    tracer.stop()

# Then, skip early layers and inject the saved activations
with model.trace("Hello world"):
    model.skip_layers(0, 6, skip_with=layer_6_out)
    result = model.logits.save()
```

### Project to Vocabulary

Decode hidden states at any layer by projecting through the final layer norm and unembedding head:

```python
with model.trace("The capital of France is"):
    hidden = model.layers_output[5]
    logits = model.project_on_vocab(hidden)
```

This is essentially a one-liner for the logit lens technique — letting you see what a model "thinks" at intermediate layers.

### Activation Steering

Apply steering vectors to specific layers during a forward pass:

```python
steering_vector = torch.randn(model.hidden_size)

with model.trace("The weather today is"):
    model.steer(layers=[1, 3], steering_vector=steering_vector, factor=0.5)
```

### Logit Lens and Patchscopes

Higher-level intervention functions are also available out of the box:

```python
from nnterp.interventions import logit_lens, patchscope_lens, TargetPrompt

# Logit lens across all layers
layer_probs = logit_lens(model, ["The capital of France is"])

# Patchscope: patch representations across prompts
target = TargetPrompt("The capital of France is", index_to_patch=-1)
patchscope_probs = patchscope_lens(
    model,
    source_prompts=["The capital of England is"],
    target_patch_prompts=target,
    layer_to_patch=10,
)
```

## How It Works Under the Hood

Unlike libraries such as TransformerLens that reimplement transformer architectures from scratch, `nnterp` works with the **original HuggingFace implementations** via NNsight's renaming feature. This is an important design choice: it means you get perfect compatibility with the original model behavior, avoiding subtle bugs that can creep in when models are re-implemented.

When you load a model, `nnterp` runs automatic validation checks to ensure the renaming was applied correctly and that all accessors return tensors of the expected shape. If a model doesn't follow the standard naming patterns, you can provide a custom `RenameConfig` to map its internals to the standardized interface (learn more [here](https://ndif-team.github.io/nnterp/adding-model-support.html)).

## Getting Started

Install nnterp with pip:

```bash
pip install nnterp
```

Note that nnterp builds on NNsight, so familiarity with NNsight's tracing API is helpful. nnterp handles the naming and common patterns, but for more complex interventions you'll still reach for NNsight directly.

Check out the full documentation at [ndif-team.github.io/nnterp](https://ndif-team.github.io/nnterp/), explore the [GitHub repository](https://github.com/ndif-team/nnterp), or join the conversation on [Discord](https://discord.gg/ndif).

---

*nnterp is part of the [NDIF](https://ndif.us) ecosystem for transparent AI research infrastructure, and was created by [Clément Dumas](https://github.com/Butanium). If you use nnterp in your research, please consider [citing the paper](https://arxiv.org/abs/2511.14465).*
