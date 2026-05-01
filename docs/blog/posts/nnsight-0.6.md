---
date: 2026-02-26
authors:
  - jaden
categories:
  - Release
---

# Introducing NNsight 0.6

*By Jaden Fiotto-Kaufman*

NNsight is releasing its sixth major version, focused on addressing user feedback about common hurdles with the library.

## Wait, What is NNsight Again?

If you are a new user or you haven't used NNsight in a while, here's a small refresher! NNsight is a Python library for interpreting and intervening on the internals of PyTorch models. You wrap a model, open a tracing context, and read or write activations at any layer. While the tool supports any Pytorch model, we also provide first-class support for popular architectures such as :hugging_face: `transformers` and `diffusers`.

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

with model.trace("The Eiffel Tower is in"):
    # Read the hidden states at layer 5
    hidden = model.transformer.h[5].output[0].save()

    # Zero out the MLP output at layer 0
    model.transformer.h[0].mlp.output[:] = 0
```

Under the hood, NNsight uses deferred execution. When you enter `with model.trace(...)`, your code [AST](https://en.wikipedia.org/wiki/Abstract_syntax_tree) is extracted, compiled into a function, and run in a worker thread. When that thread accesses `.output`, it waits until the model's forward pass reaches the selected module, extracting the desired output tensor through a PyTorch hook. This means your intervention code is fully aligned with the forward pass—no proxies, no fake tensors. You're working with real PyTorch values!

<!-- more -->

### Remote Execution with NDIF

NNsight is fully integrated with the [National Deep Inference Fabric (NDIF)](https://ndif.us), our research computing platform enabling the remote execution of large models hosted remotely. You don't need a local GPU to run your analyses with NNsight! Just add `remote=True`, and your interventions will automatically run on NDIF's infrastructure:

```python
model = LanguageModel("meta-llama/Llama-3.1-70B")

with model.trace("The Eiffel Tower is in", remote=True):
    hidden = model.transformer.h[5].output[0].save()
```

The model loads on a [meta device](https://docs.pytorch.org/docs/stable/meta.html) locally (no GPU memory needed), and NDIF fully handles execution and data transfer of requested values. Same API, same code as if you were running your code locally.

## What Was Painful Before

Users gave us consistent feedback about a few friction points with previous NNsight versions:

- **Cryptic errors.** When something went wrong inside a trace, the stack trace pointed to NNsight internals instead of your code, making debugging very hard.

- **No custom code on NDIF.** If you had local analysis functions or modules, they wouldn't run remotely due to missing packages and mismatching versions. You had to inline everything to make it work.

- **Keyword-only inputs not supported.** Calling `.trace(input_ids=my_ids)` would incorrectly hang to wait for sub-invokers instead of running a forward pass right away.

- **Performance overhead.** Every trace paid the full cost of source extraction, AST parsing, and compilation, even if you'd run the exact same trace a thousand times. Thread creation and pymount lifecycle added up too, especially for short traces.

## What Changed in 0.6

### Running Custom Code on the NDIF Remote Server

The biggest change in this release: NNsight now serializes functions and classes by value (source code) rather than by reference. Your local packages get sent along with your request and rebuilt on the server even if they aren't installed on NDIF.

```python
from nnsight import ndif, LanguageModel
import mymodule

ndif.register(mymodule)

model = LanguageModel("meta-llama/Llama-3.1-70B")

with model.trace("Hello world", remote=True):
    result = mymodule.my_analysis_function(model).save()
```

Anything defined in your main script or working directory is auto-registered. More broadly, any package on your local Python path that isn't in `site-packages` (i.e. not pip-installed) is auto-registered too. You only need to call `ndif.register()` for pip-installed local packages. Python 3.9+ clients now work with NDIF regardless of server Python version.

Concretly, this allows other developers to build upon NNsight while still being able to run their model calls on NDIF. For example, our latest addition to the NDIF ecosystem, [NNterp](https://github.com/ndif-team/nnterp), is not installed by default on NDIF, but can now be easily used by simply registering it:

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

We belive this is great, since it decouples library development from server deployment. NNterp can ship new features and fixes without waiting for NDIF to update its installation, and you always get your local version on the remote server. Win-win!

You can also test serialization locally before submitting remote jobs using `remote='local'`, and compare your environment against NDIF's with `ndif.compare()`.

### Scaling Interpretability Workflows with 2.4–3.9x Faster Traces

Our optimization efforts paid off, and tracing overhead has been cut significantly for the new release. On a 12-layer MLP benchmark (CPU):

| Scenario | v0.5.15 | v0.6.0 | Speedup |
|----------|---------|--------|---------|
| Empty trace | 1,196 µs | 308 µs | **3.9x** |
| 1 `.save()` | 1,370 µs | 474 µs | **2.9x** |
| 12 `.save()` calls | 1,697 µs | 716 µs | **2.4x** |

The fixed setup cost (source extraction, AST parsing, compilation, thread creation) dropped from ~1,100 µs to ~210 µs. Per-intervention cost dropped from ~42 µs to ~34 µs.

![v0.5.15 vs v0.6.0 comparison](version_comparison.png)

Most of the improvement is in setup cost. The overhead breakdown shows how trace setup dominates in v0.5.15, while v0.6.0 shrinks it to a fraction:

![Overhead breakdown: setup vs per-save cost](overhead_breakdown.png)

As the number of interventions grows, v0.6.0 scales much better than v0.5.15. Both are still above bare PyTorch hooks, but the gap is now significantly smaller:

![Trace time vs number of saved activations](scaling.png)

The remaining overhead beyond raw PyTorch hooks comes from nnsight's feature set: thread-based deferred execution, source extraction, automatic batching, cross-invoke variable sharing, and the mediator protocol. This overhead is constant regardless of model size, and is mostly negligible for models where the forward pass can take up to several seconds.

![Overhead: PyTorch hooks vs nnsight v0.6.0](overhead_vs_hooks.png)

Curious on how we achieved these improvements? The main optimizations include:

- **Always-on trace caching.** Source, AST, and compiled code objects are cached per call site. First trace pays full cost; subsequent calls skip compilation entirely.
- **Persistent pymount.** `.save()` and `.stop()` are mounted once at import and never unmounted, eliminating `PyType_Modified()` calls that invalidated all Python type caches on every trace enter/exit.
- **Removed `torch._dynamo.disable` wrappers.** The decorator on hook functions added unnecessary `set_eval_frame` C calls on every module forward. Removing it saves ~4 C calls per hook.
- **Batched `PyFrame_LocalsToFast`.** Cross-invoker variable sharing now syncs all variables in one C API call instead of one per variable.
- **Filtered globals copy.** Intervention threads now only copy the global names referenced in the bytecode, not the entire module globals dict.

### First-class Support for `VisionLanguageModel` and `DiffusionModel`

You got that right, NNsight now supports vision-language models and diffusion models out of the box! The new `VisionLanguageModel` class extends `LanguageModel` with an `AutoProcessor` that handles both text tokenization and image preprocessing. You can trace, intervene on, and generate from models like [LLaVA](https://huggingface.co/liuhaotian/llava-v1.6-34b), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), and other HuggingFace VLMs with the same API you already know:

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

When no `images` are passed, it falls back to standard text-only tokenization—so you can use the same model object for both modalities. Batching across invokes handles `pixel_values` alongside `input_ids`, and all existing NNsight features work as expected.

Moving now to the vision-only domain, the new `DiffusionModel` class wraps any `diffusers.DiffusionPipeline`—UNet-based ([Stable Diffusion](https://huggingface.co/stabilityai)) and Transformer-based ([Flux](https://huggingface.co/docs/diffusers/v0.36.0/en/api/pipelines/flux), [DiT](https://huggingface.co/docs/diffusers/api/pipelines/dit)) alike—so you can trace, intervene on, and iterate over denoising steps with the same API as language models.

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

`.trace()` defaults to a single denoising step for fast exploration; `.generate()` runs the full pipeline with whatever step count you specify. The code is fully architecture-agnostic: the denoiser is accessible as whatever attribute the pipeline exposes (`sd.unet` for Stable Diffusion, `flux.transformer` for Flux).

With `dispatch=False`, only lightweight config files are downloaded and the model architecture is created with meta tensors. This means that no GPU memory gets used until the first `.trace()` or `.generate()` call triggers auto-dispatch.

VLMs and diffusion models are also coming to NDIF soon, keep an eye out for that! :eyes:

### Full Support forvLLM Integration

The [vLLM](https://github.com/vllm-project/vllm) integration introduced in NNsight 0.5 allowed users to efficiently extract activations and perform interventions on inference-optimized models. This release introduces a major upgrade, with full support for all vLLM deployment configurations, including single GPU, multi-GPU tensor parallelism, Ray distributed execution, and multi-node inference—all using the same tracing API.

Single GPU vLLM execution works like any other NNsight model:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("meta-llama/Llama-3.1-8B", dispatch=True)

with model.trace("The Eiffel Tower is in the city of", temperature=0.0):
    hidden = model.model.layers[16].output[0].save()
    logits = model.logits.output.save()
```

Now you can scale your setup to multiple GPUs by setting `tensor_parallel_size`. No need to worry about tensor sharding: NNsight gathers shards behind the scenes before running interventions and re-shards tensors afterwards, ensuring that your interventions always see full, unsharded tensors:

```python
model = VLLM("meta-llama/Llama-3.1-8B", tensor_parallel_size=2, dispatch=True)

with model.trace("The Eiffel Tower is in the city of", temperature=0.0):
    hidden = model.model.layers[16].output[0].save()
```

For distributed setups, simply pass `distributed_executor_backend="ray"`, and NNsight handles the rest. Ray workers use the same intervention pipeline as local multiprocessing—mediators, batch groups, and saved values all work identically.

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

NNsight joins the cluster as a driver-only node (no GPUs consumed on the client machine) and places workers across available nodes. If no cluster exists and `RAY_ADDRESS` isn't set, a fresh local Ray cluster is started instead.

On top of all this, NNsight now supports **async mode** with real-time token streaming. Pass `mode="async"` to get an async interface powered by vLLM's `AsyncLLM`. Interventions are set up in the trace block, then tokens stream back as they're generated:

```python
model = VLLM("meta-llama/Llama-3.1-8B-Instruct", dispatch=True, mode="async")

with model.trace(prompt, temperature=0.7, max_tokens=256) as tracer:
    # Interventions apply on every forward pass during generation
    model.model.layers[16].output[0][:] += steering_vector

# Tokens stream back as they're generated
async for output in tracer.backend():
    if output.outputs:
        print(output.outputs[0].text)
```

This makes it straightforward to build interactive applications—chat interfaces, real-time visualization tools, or anything else that benefits from streaming output—while still applying interventions on every forward pass. Async mode works with all executor backends: single GPU, multiprocessing TP, and Ray. For a hands-on example, check out our [nnsight-vllm-demos](https://github.com/ndif-team/nnsight-vllm-demos) repo, which includes an async chat interface with SAE-based feature steering.

The [vLLM integration README](https://github.com/ndif-team/nnsight/blob/main/src/nnsight/modeling/vllm/README.md) covers the full architecture. The [`examples/multi_node_with_ray/`](https://github.com/ndif-team/nnsight/blob/main/src/nnsight/modeling/vllm/examples/multi_node_with_ray/) directory has a runnable Docker-based multi-node setup you can use as a starting point.

### Cleaner Error Messages

Thanks to our simplified remote execution approach, exceptions inside traces now show clean stack traces pointing to your original code:

```
Traceback (most recent call last):
  File "my_script.py", line 9, in <module>
    with model.trace("Hello world"):
  File "my_script.py", line 11, in <module>
    output = model.transformer.h[999].output.save()

IndexError: list index out of range
```

Finally no NNsight code in sight within the traceback! But if you really need the full trace for debugging NNsight itself, you can always get it with `python -d my_script.py`.

### Initial Support for AI Coding Assistants

The rapid growth of tools like Claude Code, Codex and Cursor agents prompted us to rethink how users interact with our library. For this reason, NNsight 0.6 introduces first-class support for AI coding agents! We've built a [skills repository](https://github.com/ndif-team/skills) that integrates with Claude Code and OpenAI Codex: you can install NNsight-related skills as plugins to improve your agent's ability in writing interventions using our expected format. We also support [Context7](https://github.com/upstash/context7) as an MCP server, so any MCP-compatible LLM client can pull up-to-date NNsight documentation on the fly. For agents that work with raw context files, the NNsight repository now includes a [CLAUDE.md](https://github.com/ndif-team/nnsight/blob/main/CLAUDE.md) and a [NNsight.md](https://github.com/ndif-team/nnsight/blob/main/NNsight.md), two comprehensive guides covering the full API, common patterns, gotchas, and debugging tips. Our goal for this and future releases is to make NNsight the most human- and agent-friendly interpretability library on the market!

### Other Notable Changes

#### Smarter Input Detection

Previously, NNsight checked for positional arguments to decide whether to create an implicit invoker. Keyword-only inputs like `input_ids=my_ids` would silently fail. Now the batching logic inspects whether arguments affect batch size. This just works:

```python
with model.trace(input_ids=my_ids):
    hidden = model.transformer.h[0].output[0].save()
```

#### For-Loop Iteration

`tracer.iter` now supports standard Python `for` loops as an alternative to `with` blocks. The `for` version is faster because it runs inline in the worker thread—no source extraction or compilation for the loop body.

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits = list().save()
    for step in tracer.iter[:]:
        logits.append(model.lm_head.output[0][-1].argmax(dim=-1))
```

Bounded slices (`iter[:3]`), single indices (`iter[0]`), and lists (`iter[[0, 2, 4]]`) all work.

#### And Even More!

- **Multiple wrappers on the same model.** You can now wrap the same PyTorch model with multiple `NNsight` instances without breaking hooks.
- **`python -c` support.** `python -c "from nnsight import ..."` now works.
- **Compressed NDIF results.** Results are compressed with zstandard for smaller downloads.
- **Memory leak fixes.** Fixed reference loops in the interleaver and tracer.
- - **SAEs and LoRA adapters on NDIF.** Support for running sparse autoencoders and adapter layers on NDIF's hosted models, so you can do feature-level analysis on large models without downloading anything.

### Breaking Changes

The v0.4 compatibility layer has been removed. If you're still using `nnsight.apply()`, `nnsight.list`, `nnsight.cond()`, or the `trace=False` parameter, deprecated in v0.5.0 with warnings, you will need to update your code. `model.iter`, `model.all()`, and `model.next()` still work but now emit deprecation warnings — you should use `tracer.iter`, `tracer.all()`, and `tracer.next()` instead. The `with tracer.iter[...]:` block syntax is also deprecated in favor of the faster `for step in tracer.iter[...]:` form.

If you have custom model classes that implement `_prepare_input` or `_batch`, you may also need to update them to match the new signature in `nnsight/intervention/batching.py`.

The full list of breaking changes can be found in the [v0.6 release notes](https://github.com/ndif-team/nnsight/blob/main/0.6.0.md).

## Towards an Open-source Interpretability Ecosystem

Interpretability research is a relatively new field of research, and the tooling landscape remains fragmented to this day. Many new papers implement their own hooking logic, even going as far as reimplementing full models to perform small interventions within their internals. Beyond inefficiency, this fragmentation is also bad for reproducibility: methods originally implemented on GPT-2 require major adjustments to evaluate on newer model architectures, effectively making it harder to verify that findings generalize across various models.

Our NDIF team is addressing this by developing a robust open-source ecosystem to standardize how people write, share, and run interpretability code. Our goal is to abstract away the challenges of doing interpretability research at scale, enabling researchers to focus on the science instead of the engineering. NNsight is just the beginning of our vision for the NDIF ecosystem. Some of the other projects related to NNsight now include:

- **[NNterp](https://github.com/ndif-team/nnterp)** — A standardized interface for NNsight-supported models and common interpretability methods, enabling easy reuse for large-scale analyses. Write your method once, run it efficiently on GPT-2, Llama, Gemma, or Qwen without changing a line.
- **[Workbench](https://workbench.ndif.us)** — A no-code web UI for exploratory interpretability research. Built on top of NNsight and NDIF, it gives access to large models and powerful interventions through an intuitive interface, making interpretability research more accessible to educators and domain experts without coding experience.
- **[Cookbook](https://github.com/ndif-team/cookbook)** — A growing collection of mechanistic interpretability paper replications, implemented with NNsight and NNterp. Each one is a runnable Colab notebook that executes on NDIF, so you can reproduce published results on large models without local GPUs. The goal is a shared reference library the field can build on.
- **[NDIF Server](https://github.com/ndif-team/ndif)** — The NDIF server is open-source and pip-installable, so anyone can deploy their own NDIF cluster for remote access. If you have GPUs and want to host models for your lab or organization, `pip install ndif` gets you a working instance.
- **[Circuit Tracer](https://github.com/safety-research/circuit-tracer-dev/)** — Tools for finding and visualizing circuits using [cross-layer transcoder](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) features. Find attribution graphs, annotate them, and intervene on specific features to observe effects on model output, now also supporting remote execution on NDIF!
- **[Interpreto](https://github.com/FOR-sight-ai/interpreto)** — A toolkit for attribution and concept-based interpretability on Transformer language models, exploiting NNsight for efficient activation extraction and intervention.

---

NNsight 0.6 is available now: `pip install nnsight --upgrade`

Docs: [nnsight.net](https://nnsight.net) · GitHub: [github.com/ndif-team/nnsight](https://github.com/ndif-team/nnsight) · Forum: [discuss.ndif.us](https://discuss.ndif.us) · Discord: [discord.gg/6uFJmCSwW7](https://discord.gg/6uFJmCSwW7)
