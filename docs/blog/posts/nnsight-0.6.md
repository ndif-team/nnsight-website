---
date: 2026-02-11
authors:
  - jaden
categories:
  - Release
---

# NNsight 0.6

*By Jaden Fiotto-Kaufman*

NNsight is releasing its sixth major version, focused on addressing user feedback about common hurdles with the library. Before getting into what changed, here's what nnsight is and why it exists.

<!-- more -->

## What is NNsight?

NNsight is a Python library for interpreting and intervening on the internals of PyTorch models. You wrap a model, open a tracing context, and read or write activations at any layer.

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

with model.trace("The Eiffel Tower is in"):
    # Read the hidden states at layer 5
    hidden = model.transformer.h[5].output[0].save()

    # Zero out the MLP output at layer 0
    model.transformer.h[0].mlp.output[:] = 0
```

Under the hood, nnsight uses deferred execution. When you enter `with model.trace(...)`, your code is extracted via AST, compiled into a function, and run in a worker thread. When that thread accesses `.output`, it blocks until the model's forward pass reaches that layer and provides the real tensor through a PyTorch hook. This means your intervention code runs inline with the forward pass—no proxies, no fake tensors. You're working with real PyTorch values.

### Remote Execution with NDIF

NNsight pairs with [NDIF](https://ndif.us) (National Deep Inference Fabric), a research computing platform that lets you run the same intervention code on large models hosted remotely. You don't need a local GPU. Just add `remote=True`:

```python
model = LanguageModel("meta-llama/Llama-3.1-70B")

with model.trace("The Eiffel Tower is in", remote=True):
    hidden = model.transformer.h[5].output[0].save()
```

The model loads on meta device locally (no GPU memory used), and NDIF handles execution on its infrastructure. Same API, same code.

## What Was Painful

We heard consistent feedback about a few friction points:

- **Cryptic errors.** When something went wrong inside a trace, the stack trace pointed to nnsight internals instead of your code. Debugging meant guessing which line in your script actually caused the error.
- **Custom code on NDIF didn't work.** If you had local analysis functions or modules, they wouldn't run on NDIF because the server didn't have your packages installed. You had to inline everything.
- **Keyword-only inputs broke.** Passing `input_ids=my_ids` to `.trace()` without positional arguments would incorrectly wait for sub-invokers instead of running a forward pass.
- **Performance overhead.** Every trace paid the full cost of source extraction, AST parsing, and compilation—even if you'd run the exact same trace a thousand times. Thread creation and pymount lifecycle added up.

## What's New

### Custom Code on NDIF

The biggest change: nnsight now serializes functions and classes by value (source code) rather than by reference. Your local packages get sent along with your request and rebuilt on the server—even if they aren't installed on NDIF.

```python
from nnsight import ndif, LanguageModel
import mymodule

ndif.register(mymodule)

model = LanguageModel("meta-llama/Llama-3.1-70B")

with model.trace("Hello world", remote=True):
    result = mymodule.my_analysis_function(model).save()
```

Anything defined in your main script or working directory is auto-registered. More broadly, any package on your local Python path that isn't in `site-packages` (i.e. not pip-installed) is auto-registered too. You only need to call `ndif.register()` for pip-installed local packages. Python 3.9+ clients now work with NDIF regardless of server Python version.

A real example of where this matters: [nnterp](https://github.com/ndif-team/nnterp) is a library built on nnsight that standardizes transformer interfaces across model families. NDIF doesn't have nnterp installed, but that doesn't matter—just register it:

```python
from nnterp import StandardizedTransformer
from nnsight import ndif
import nnterp

ndif.register(nnterp)

model = StandardizedTransformer("meta-llama/Llama-3.1-70B")

with model.trace("hello", remote=True):
    layer_5_output = model.layers_output[5]
    model.layers_output[10] = layer_5_output
```

This decouples library development from server deployment. nnterp can ship new features and fixes without waiting for NDIF to update its installation—you always run the version you have locally.

You can also test serialization locally before submitting remote jobs with `remote='local'`, and compare your environment against NDIF's with `ndif.compare()`.

### Cleaner Error Messages

Exceptions inside traces now show clean stack traces pointing to your original code:

```
Traceback (most recent call last):
  File "my_script.py", line 9, in <module>
    with model.trace("Hello world"):
  File "my_script.py", line 11, in <module>
    output = model.transformer.h[999].output.save()

IndexError: list index out of range
```

No nnsight internals in the traceback. If you need the full trace for debugging nnsight itself, run with `python -d my_script.py`.

### Smarter Input Detection

Previously, nnsight checked for positional arguments to decide whether to create an implicit invoker. Keyword-only inputs like `input_ids=my_ids` would silently fail. Now the batching logic inspects whether arguments affect batch size. This just works:

```python
with model.trace(input_ids=my_ids):
    hidden = model.transformer.h[0].output[0].save()
```

### For-Loop Iteration

`tracer.iter` now supports standard Python `for` loops as an alternative to `with` blocks. The `for` version is faster because it runs inline in the worker thread—no source extraction or compilation for the loop body.

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits = list().save()
    for step in tracer.iter[:]:
        logits.append(model.lm_head.output[0][-1].argmax(dim=-1))
```

Bounded slices (`iter[:3]`), single indices (`iter[0]`), and lists (`iter[[0, 2, 4]]`) all work.

### Performance: 2.4–3.9x Faster Traces

Trace overhead has been cut significantly. On a 12-layer MLP benchmark (CPU):

| Scenario | v0.5.15 | v0.6.0 | Speedup |
|----------|---------|--------|---------|
| Empty trace | 1,196 µs | 308 µs | **3.9x** |
| 1 `.save()` | 1,370 µs | 474 µs | **2.9x** |
| 12 `.save()` calls | 1,697 µs | 716 µs | **2.4x** |

The fixed setup cost (source extraction, AST parsing, compilation, thread creation) dropped from ~1,100 µs to ~210 µs. Per-intervention cost dropped from ~42 µs to ~34 µs.

![v0.5.15 vs v0.6.0 comparison](version_comparison.png)

Most of the improvement is in setup cost. The overhead breakdown shows how trace setup dominates in v0.5.15, while v0.6.0 shrinks it to a fraction:

![Overhead breakdown: setup vs per-save cost](overhead_breakdown.png)

As the number of interventions grows, v0.6.0 scales much better than v0.5.15. Both are still above bare PyTorch hooks, but the gap narrows with more saves:

![Trace time vs number of saved activations](scaling.png)

The remaining overhead beyond raw PyTorch hooks comes from nnsight's feature set: thread-based deferred execution, source extraction, automatic batching, cross-invoke variable sharing, and the mediator protocol. This overhead is constant regardless of model size—for real models where the forward pass takes milliseconds or seconds, it's negligible.

![Overhead: PyTorch hooks vs nnsight v0.6.0](overhead_vs_hooks.png)

Where the savings come from:

- **Always-on trace caching.** Source, AST, and compiled code objects are cached per call site. First trace pays full cost; subsequent calls skip compilation entirely.
- **Persistent pymount.** `.save()` and `.stop()` are mounted once at import and never unmounted, eliminating `PyType_Modified()` calls that invalidated all Python type caches on every trace enter/exit.
- **Removed `torch._dynamo.disable` wrappers.** The decorator on hook functions added unnecessary `set_eval_frame` C calls on every module forward. Removing it saves ~4 C calls per hook.
- **Batched `PyFrame_LocalsToFast`.** Cross-invoker variable sharing now syncs all variables in one C API call instead of one per variable.
- **Filtered globals copy.** Intervention threads now only copy the global names referenced in the bytecode, not the entire module globals dict.

### Agent Support

NNsight now has first-class support for AI coding agents. We've built a [skills repository](https://github.com/ndif-team/skills) that integrates with Claude Code and OpenAI Codex—install it once and your agent knows how to write nnsight code. We also support [Context7](https://github.com/upstash/context7) as an MCP server, so any MCP-compatible LLM client can pull up-to-date nnsight documentation on the fly. For agents that work with raw context files, the repo includes [CLAUDE.md](https://github.com/ndif-team/nnsight/blob/main/CLAUDE.md) and [NNsight.md](https://github.com/ndif-team/nnsight/blob/main/NNsight.md)—comprehensive guides covering the full API, common patterns, gotchas, and debugging tips. The goal is to make nnsight as easy for agents to use as it is for humans.

### VisionLanguageModel

NNsight now supports vision-language models out of the box. The new `VisionLanguageModel` class extends `LanguageModel` with an `AutoProcessor` that handles both text tokenization and image preprocessing. You can trace, intervene on, and generate from models like LLaVA, Qwen2-VL, and other HuggingFace VLMs with the same API you already know:

```python
from nnsight import VisionLanguageModel
from PIL import Image

model = VisionLanguageModel(
    "llava-hf/llava-interleave-qwen-0.5b-hf",
    device_map="auto",
    dispatch=True,
)
img = Image.open("photo.jpg")

# Trace with text + image
with model.trace("<image>\nDescribe this image", images=[img]):
    hidden = model.model.language_model.layers[-1].output.save()

# Generation
with model.generate("<image>\nDescribe this image", images=[img], max_new_tokens=50):
    output = model.generator.output.save()
```

When no `images` are passed, it falls back to standard text-only tokenization—so you can use the same model object for both modalities. Batching across invokes handles `pixel_values` alongside `input_ids`, and all existing nnsight features (scan, edit, barriers, caching) work as expected.

### DiffusionModel

NNsight now supports diffusion pipelines as first-class citizens. The new `DiffusionModel` class wraps any `diffusers.DiffusionPipeline`—UNet-based (Stable Diffusion) and transformer-based (Flux, DiT) alike—so you can trace, intervene on, and iterate over denoising steps with the same API as language models.

```python
from nnsight import DiffusionModel

sd = DiffusionModel("stabilityai/stable-diffusion-2-1")

# Quick single-step trace
with sd.trace("A cat"):
    denoiser_out = sd.unet.output.save()

# Full generation with step-by-step access
with sd.generate("A cat", num_inference_steps=50) as tracer:
    denoiser_outputs = list().save()
    for step in tracer.iter[:]:
        denoiser_outputs.append(sd.unet.output[0].clone())
```

`.trace()` defaults to a single denoising step for fast exploration; `.generate()` runs the full pipeline with whatever step count you specify. The code is architecture-agnostic—the denoiser is accessible as whatever attribute the pipeline exposes (`sd.unet` for Stable Diffusion, `flux.transformer` for Flux).

With `dispatch=False`, only lightweight config files are downloaded and the model architecture is created with meta tensors—no GPU memory used until the first `.trace()` or `.generate()` call triggers auto-dispatch. This also means diffusion models on NDIF are coming soon.

### vLLM Integration

The vLLM integration got a major upgrade. nnsight now supports the full range of vLLM deployment configurations—single GPU, multi-GPU tensor parallelism, Ray distributed execution, and multi-node inference—all with the same tracing API.

Single GPU works like any other nnsight model:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("meta-llama/Llama-3.1-8B", dispatch=True)

with model.trace("The Eiffel Tower is in the city of", temperature=0.0):
    hidden = model.model.layers[16].output[0].save()
    logits = model.logits.output.save()
```

Scale up to multiple GPUs by setting `tensor_parallel_size`. Intervention code always sees complete, unsharded tensors—nnsight gathers shards before your code runs and re-shards afterward:

```python
model = VLLM("meta-llama/Llama-3.1-8B", tensor_parallel_size=2, dispatch=True)

with model.trace("The Eiffel Tower is in the city of", temperature=0.0):
    hidden = model.model.layers[16].output[0].save()
```

For distributed setups, pass `distributed_executor_backend="ray"` and nnsight handles the rest. Ray workers use the same intervention pipeline as local multiprocessing—mediators, batch groups, and saved values all work identically.

For multi-node inference where TP workers are on different machines, point `RAY_ADDRESS` at an existing Ray cluster:

```bash
export RAY_ADDRESS="head-node:6379"
```

```python
model = VLLM(
    "meta-llama/Llama-3.1-70B",
    tensor_parallel_size=8,
    distributed_executor_backend="ray",
    dispatch=True,
)

with model.trace("Hello world", temperature=0.0):
    hidden = model.model.layers[40].output[0].save()
```

nnsight joins the cluster as a driver-only node (no GPUs consumed on the client machine) and places workers across available nodes. If no cluster exists and `RAY_ADDRESS` isn't set, a fresh local Ray cluster is started instead.

The [vLLM integration README](https://github.com/ndif-team/nnsight/blob/main/src/nnsight/modeling/vllm/README.md) covers the full architecture. The [`examples/multi_node_with_ray/`](https://github.com/ndif-team/nnsight/blob/main/src/nnsight/modeling/vllm/examples/multi_node_with_ray/) directory has a runnable Docker-based multi-node setup you can use as a starting point.

### Other Changes

- **Multiple wrappers on the same model.** You can now wrap the same PyTorch model with multiple `NNsight` instances without breaking hooks.
- **`python -c` support.** `python -c "from nnsight import ..."` now works.
- **Compressed NDIF results.** Results are compressed with zstandard for smaller downloads.
- **Memory leak fixes.** Fixed reference loops in the interleaver and tracer.

### Breaking Changes

The v0.4 compatibility layer has been removed. If you're still using `nnsight.apply()`, `nnsight.list`, `nnsight.cond()`, or the `trace=False` parameter, you'll need to update. These were deprecated in v0.5.0 with warnings pointing to the replacements — standard Python builtins and calling methods without a `with` context. `model.iter`, `model.all()`, and `model.next()` still work but now emit deprecation warnings — use `tracer.iter`, `tracer.all()`, and `tracer.next()` instead. The `with tracer.iter[...]:` block syntax is also deprecated in favor of the faster `for step in tracer.iter[...]:` form. The full list is in the [release notes](https://github.com/ndif-team/nnsight/blob/main/0.6.0.md).

If you have custom model classes that implement `_prepare_input` or `_batch`, you may need to update them to match the new signature in `nnsight/intervention/batching.py`.

## Where NNsight Is Headed

Interpretability research is fragmented. Every paper rolls its own hooks, its own activation patching, its own way of naming layers. Techniques that work on GPT-2 break on Llama because the module paths are different. Results are hard to reproduce because the tooling isn't shared.

NNsight and the projects around it are trying to fix that—standardize how people write, share, and run interpretability code so the field can build on itself instead of reimplementing the same primitives.

Here's what's in progress:

- **[nnterp](https://github.com/ndif-team/nnterp)** — a library built on nnsight that gives all transformer architectures the same interface. Different model families use different naming conventions for identical components. nnterp maps them to a common API so techniques like logit lens, activation patching, and steering work out of the box on any supported model. Write your method once, run it on GPT-2, Llama, Gemma, or Qwen without changing a line.
- **[Cookbook](https://github.com/ndif-team/cookbook)** — a growing collection of mechanistic interpretability paper replications, implemented with nnsight and nnterp. Each one is a runnable Colab notebook that executes on NDIF, so you can reproduce published results on large models without local GPUs. The goal is a shared reference library the field can build on.
- **[Circuit Tracer](https://github.com/safety-research/circuit-tracer-dev/)** — tools for finding and visualizing circuits using cross-layer transcoder features. Find attribution graphs, annotate them, and intervene on specific features to observe effects on model output.
- **[Workbench](https://workbench.ndif.us)** — a web UI for exploratory interpretability research. Built on nnsight and NDIF, it gives you an interactive environment for probing model internals without writing code.
- **SAEs and LoRA adapters on NDIF** — support for running sparse autoencoders and adapter layers on NDIF's hosted models, so you can do feature-level analysis on large models without downloading anything.
- **[Open-source NDIF](https://github.com/ndif-team/ndif)** — the NDIF server is open source and pip-installable, so anyone can deploy their own NDIF cluster. If you have GPUs and want to host models for your lab or organization, `pip install ndif` gets you a working instance.

---

NNsight 0.6 is available now: `pip install nnsight --upgrade`

Docs: [nnsight.net](https://nnsight.net) · GitHub: [github.com/ndif-team/nnsight](https://github.com/ndif-team/nnsight) · Forum: [discuss.ndif.us](https://discuss.ndif.us) · Discord: [discord.gg/6uFJmCSwW7](https://discord.gg/6uFJmCSwW7)
