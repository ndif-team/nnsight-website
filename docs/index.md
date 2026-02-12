---
hide:
  - navigation
  - toc
  - path
  - title
---

<style>
  /* Make content area full width for home page */
  .md-main__inner {
    max-width: 100% !important;
    margin: 0 !important;
  }

  .md-content {
    max-width: 100% !important;
    margin: 0 !important;
  }

  .md-content__inner {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
  }

  .md-sidebar {
    display: none !important;
  }

  .md-grid {
    max-width: 100% !important;
    margin: 0 !important;
  }

  .md-content h1,
  .md-content__inner > h1:first-child,
  .md-typeset h1 {
    display: none !important;
  }

  /* Hero section */
  .nn-hero {
    position: relative;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }

  .fixed-background {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    z-index: 0;
  }

  .nn-hero-content {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 4rem;
    max-width: 1400px;
    padding: 2rem;
    background: transparent;
    border-radius: 0.5rem;
  }

  .nn-hero-text {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .nn-hero-desc {
    font-size: 1.15rem;
    line-height: 1.5;
    color: var(--md-default-fg-color--light);
    margin: 0;
  }

  .nn-hero-links {
    display: flex;
    flex-direction: row;
    gap: 0.5rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 2;
  }

  .nn-hero-links a {
    padding: 0.5rem 1.2rem;
    border-radius: 4px;
    font-size: 0.9rem;
    font-weight: 600;
    text-decoration: none;
    transition: opacity 0.2s;
  }

  .nn-hero-links a:hover {
    opacity: 0.85;
  }

  .nn-hero-links .nn-btn-primary {
    background: linear-gradient(135deg, #8a8cc4, #49a5d2);
    color: #fff;
  }

  .nn-hero-links .nn-btn-secondary {
    background: var(--md-default-bg-color);
    border: 1px solid var(--md-default-fg-color--lighter);
    color: var(--md-default-fg-color);
  }

  .nn-hero-logo {
    flex-shrink: 0;
  }

  .nn-hero-logo img {
    width: 650px;
    height: auto;
  }

  /* Content section below hero */
  .nn-home-content {
    position: relative;
    z-index: 1;
    max-width: 1600px;
    margin: 0 auto;
    padding: 3rem 2rem 4rem;
    background: transparent;
  }

  .nn-home-content h2 {
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 2rem;
    margin-bottom: 0.75rem;
  }

  .nn-home-content p,
  .nn-home-content li {
    font-size: 1rem;
    line-height: 1.6;
    color: var(--md-default-fg-color--light);
  }

  .nn-home-content ul {
    padding-left: 1.25rem;
  }

  .nn-home-content code {
    font-size: 0.85rem;
  }

  /* Responsive - intermediate */
  @media (max-width: 1100px) {
    .nn-hero-logo img {
      width: 400px;
    }

    .nn-hero-content {
      gap: 2rem;
    }
  }

  /* Responsive - mobile */
  @media (max-width: 768px) {
    .nn-hero-content {
      flex-direction: column-reverse;
      gap: 2rem;
      text-align: center;
    }

    .nn-hero-links {
      justify-content: center;
    }

    .nn-hero-logo img {
      width: 350px;
    }

    .nn-hero {
      min-height: 70vh;
    }
  }
</style>

<div class="fixed-background"></div>

<div class="nn-hero">
  <div class="nn-hero-content">
    <div class="nn-hero-text">
      <p class="nn-hero-desc">
        <strong>NNsight</strong> (/ɛn.saɪt/) is a package for interpreting and manipulating the internals of deep learning models.
      </p>
      <div class="nn-hero-links">
        <a href="getting-started/" class="nn-btn-primary">Get Started</a>
        <a href="documentation/" class="nn-btn-secondary">Docs</a>
        <a href="features/" class="nn-btn-secondary">Features</a>
        <a href="about/" class="nn-btn-secondary">About</a>
      </div>
    </div>
    <div class="nn-hero-logo">
      <img src="assets/logo.svg" alt="NNsight Logo">
    </div>
  </div>
</div>

<div class="nn-home-content md-typeset" markdown>

## What is nnsight?

**nnsight** is a Python library that enables interpreting and intervening on the internals of deep learning models. It provides a clean, Pythonic interface for:

- :material-magnify: **Accessing activations** at any layer during forward passes
- :material-pencil: **Modifying activations** to study causal effects
- :material-chart-line: **Computing gradients** with respect to intermediate values
- :material-layers-triple: **Batching interventions** across multiple inputs efficiently

Originally developed by the [NDIF team](https://ndif.us/) at Northeastern University, nnsight supports local execution on any PyTorch model and remote execution on large models via the NDIF infrastructure.

## What does that look like?

Install nnsight:

```bash
pip install nnsight
```

Intervene:

```python
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto', dispatch=True)

with model.trace('The Eiffel Tower is in the city of', remote=True/False):
    # Intervene on activations
    model.transformer.h[0].output[0][:] = 0

    # Access and save hidden states
    hidden_states = model.transformer.h[-1].output[0].save()

    # Get model output
    output = model.output.save()

print(model.tokenizer.decode(output.logits.argmax(dim=-1)[0]))
```

</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vanta@0.5.24/dist/vanta.dots.min.js"></script>

<script>
var vantaEffect = null;

function initVanta() {
  if (vantaEffect) {
    vantaEffect.destroy();
  }

  var darkMode = document.body.getAttribute('data-md-color-scheme') === 'slate'
    || document.documentElement.getAttribute('data-md-color-scheme') === 'slate';

  vantaEffect = VANTA.DOTS({
    el: ".fixed-background",
    THREE: THREE,
    mouseControls: false,
    touchControls: false,
    gyroControls: false,
    minHeight: 200.00,
    minWidth: 200.00,
    scale: 1.00,
    scaleMobile: 1.00,
    color: darkMode ? 0xCECDC3 : 0x000000,
    color2: darkMode ? 0xCECDC3 : 0x000000,
    backgroundColor: darkMode ? 0x100F0F : 0xFFFFFF,
    size: darkMode ? 0.50 : 1.00,
    spacing: 10.00,
    showLines: false
  });
}

document.addEventListener('DOMContentLoaded', function() {
  initVanta();

  new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.attributeName === 'data-md-color-scheme') {
        initVanta();
      }
    });
  }).observe(document.body, { attributes: true, attributeFilter: ['data-md-color-scheme'] });
});
</script>
