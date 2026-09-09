# Tutorials

Comprehensive tutorials covering core concepts and research techniques.

## Getting Started

- **[Walkthrough](get_started/walkthrough.ipynb)** - Complete introduction to nnsight
- **[Chat Templates](get_started/chat_templates.ipynb)** - Working with chat-formatted models
- **[Remote Access](get_started/start_remote_access.ipynb)** - Getting started with NDIF

## Migrating

- **[From TransformerLens](migrating/from_transformerlens.ipynb)** - Port a TransformerLens script: the hook-name/module-path table, and the weight-processing conventions that make a correct port produce different numbers

## Causal Tracing

- **[Causal Models Intro](causal_mediation_analysis/causal_models_intro.ipynb)** - Introduction to causal modeling
- **[Activation Patching](causal_mediation_analysis/activation_patching.ipynb)** - Patching activations to trace information flow
- **[Attribution Patching](causal_mediation_analysis/attribution_patching.ipynb)** - Gradient-based attribution methods
- **[Causal Mediation I](causal_mediation_analysis/causal_mediation_analysis_i.ipynb)** - Causal mediation analysis part 1
- **[Causal Mediation II](causal_mediation_analysis/causal_mediation_analysis_ii.ipynb)** - Causal mediation analysis part 2
- **[DAS](causal_mediation_analysis/DAS.ipynb)** - Distributed Alignment Search

## Probing

- **[Logit Lens](probing/logit_lens.ipynb)** - Interpreting intermediate representations
- **[Diffusion Lens](probing/diffusion_lens.ipynb)** - Probing diffusion models
- **[Concept Erasure](probing/concept_erasure.ipynb)** - Removing a linear subspace inside the forward pass, and measuring what it costs

## Attention

- **[Attention Patterns and Per-Head Detection](attention/attention_heads.ipynb)** - Reading the attention probability matrix with `.source`, and scoring every head for induction, previous-token and sink behaviour

## Circuits

- **[Attribution Graphs](circuits/attribution_graphs.ipynb)** - Replace the MLPs with a cross-layer transcoder and trace which features caused an output

## Architectures

- **[MoE Routing](architectures/moe_routing.ipynb)** - Reading a mixture-of-experts router, and the two ablations that quietly measure the wrong thing
- **[State-Space Models](architectures/state_space_models.ipynb)** - Reading and writing the recurrent state of a Mamba-style model, where there is no KV cache

## Engineering

- **[Dataset-Scale Activation Caching](engineering/activation_caching.ipynb)** - Running a corpus to disk: throughput, memory, resumability, and the anchor that says the cache still matches the model

## Steering

- **[Evaluating SAEs](steering/evaluating_saes.ipynb)** - Given a pretrained sparse autoencoder, is it any good?
- **[Dictionary Learning](steering/dict_learning.ipynb)** - Learning interpretable feature dictionaries
- **[LoRA Tutorial](steering/LoRA_tutorial.ipynb)** - Low-rank adaptation with nnsight

## Vision

- **[Cross-Attention Ablation](vision/cross-attention-ablation.ipynb)** - Localising where a prompt attaches in Stable Diffusion
