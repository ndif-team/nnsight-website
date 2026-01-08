---
template: home.html
title: nnsight - Interpretable Neural Networks
hide:
  - navigation
  - toc
---

<div class="fixed-background"></div>

<section class="hero-section">
<div class="hero-content">
<div class="hero-text">

# Interpretable Neural Networks

<p class="tagline">
<strong>NNsight</strong> <span class="pronunciation">(/ɛn.saɪt/)</span> is a Python package for interpreting and manipulating the internals of deep learning models.
</p>

<div class="hero-buttons">
<a href="getting-started/quickstart/" class="btn-primary">Get Started</a>
<a href="features/getting/" class="btn-secondary">Features</a>
<a href="tutorials/get_started/walkthrough/" class="btn-secondary">Tutorials</a>
<a href="api/" class="btn-secondary">API Docs</a>
</div>

</div>
<div class="hero-logo">
<img src="assets/logo.svg" alt="NNsight Logo">
</div>
</div>
</section>

<section class="features-section">
<div class="features-container">

<div class="feature-row">
<div class="feature-text">

### Wrap Any PyTorch Model

The NNsight class wraps any PyTorch model, enabling powerful tracing and intervention capabilities.

[Get Started →](getting-started/quickstart/)

</div>
<div class="feature-code">

```python
from nnsight import NNsight, LanguageModel

# Wrap any PyTorch model
model = NNsight(my_pytorch_model)

# Or load a language model directly
llm = LanguageModel('openai-community/gpt2')
```

</div>
</div>

<div class="feature-row">
<div class="feature-text">

### Access Hidden States

Easily expose and save module inputs and outputs during forward passes.

[Walkthrough →](tutorials/get_started/walkthrough/)

</div>
<div class="feature-code">

```python
with model.trace('The Eiffel Tower is in'):
    
    # Save hidden states from any layer
    hidden = model.transformer.h[5].output.save()
    
    # Access inputs too
    layer_input = model.transformer.h[6].input.save()

print(hidden.shape)  # Access saved values
```

</div>
</div>

<div class="feature-row">
<div class="feature-text">

### Develop Complex Interventions

Edit activations, patch between prompts, and measure causal effects.

[Tutorials →](tutorials/causal_mediation_analysis/activation_patching/)

</div>
<div class="feature-code">

```python
with model.trace() as tracer:
    with tracer.invoke('The Eiffel Tower is in'):
        # Zero out MLP output
        model.transformer.h[-1].mlp.output[:] = 0
        intervened = model.lm_head.output.save()

    with tracer.invoke('The Eiffel Tower is in'):
        original = model.lm_head.output.save()
```

</div>
</div>

</div>
</section>
