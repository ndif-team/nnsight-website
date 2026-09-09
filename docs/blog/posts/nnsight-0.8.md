---
date: 2026-09-09
authors:
  - jaden
categories:
  - Release
---

# NNsight 0.8

We spent the summer rebuilding NNsight from the inside, and it's finally ready. v0.8 is the
biggest release the library has had: the execution engine is rewritten from scratch, and a lot
of what was hard or impossible in v0.7 falls out of that. The way you use NNsight hasn't changed
at all. You still open a `with model.trace(...)` block and read, edit, and `.save()` a model's
internals as ordinary Python.

The parts we're most excited about:

- **One class for everything in `transformers`.** `TransformersModel` is backed by a
  `transformers.pipeline`, so any task the pipeline factory can build is now traceable with the
  same code you write for GPT-2. Whisper and the audio models, LLaVA and Qwen3-VL, BERT and the
  encoder tasks, classifiers, and PEFT adapters applied at load. Interpretability on speech and
  vision models stops being a porting exercise.

- **Speed was a big focus.** Your code and the forward pass now interleave as greenlets rather than
  OS threads. NNsight's own overhead is cheaper and, more to the point, stays flat as you touch
  more of the model: per read it is 3x flatter than v0.7, per batched prompt 6x. Caching every
  layer, sweeping every head, and running big batches stop being the slow path.

- **NNsight :handshake: vLLM is now the fastest and most complete way to do interpretability at
  scale.** Interventions run inside the engine worker and survive CUDA graph replay, so you keep
  91% to 96% of vanilla vLLM's throughput while tracing it. You can install a block on the engine
  itself and serve it over HTTP to clients that have no GPU of their own.

- **And plenty more,** like tensor parallelism for models too big for one card, quantization by
  naming it where you would name a dtype, and `.source`, which opens up the operations inside a
  forward pass that never had a module to attach to.

This is a pre-release. You can install it and use it locally today. The full release lands next
month, when v0.8 goes live on NDIF and remote execution comes with it.

<!-- more -->

## What NNsight does

If you're new here, or you haven't used NNsight in a while: NNsight is a Python library for interpreting and intervening on the internals of PyTorch models. You wrap a model, open a tracing context, and read or write activations at any layer.

```python
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of"):
    # zero out layer 0's output; the model computes on the edited value
    model.transformer.h[0].output[:] = 0

    # read a hidden state further down
    hidden = model.transformer.h[6].output.save()

    logits = model.output.logits.save()
```

Inside the block you aren't running the model. You're describing what to do when it runs. Reading `.output` hands you the real tensor once the forward pass reaches that module, and assigning to it splices your value in. There are no proxies or fake tensors involved, and you never register a hook. The one rule is that your block has to touch the model in the order the model runs, top to bottom, which is why layer 0 comes before layer 6 above.

## One class for every Hugging Face Transformers model

NNsight used to have a class per model family: `LanguageModel`, `VisionLanguageModel`, and a growing pile of special cases behind them. v0.8 has one class for everything in `transformers`, `TransformersModel`, backed by a `transformers.pipeline`, so it works for any task the pipeline factory knows how to build. (`diffusers` pipelines have their own class, `DiffusionModel`, and vLLM has `VLLM`.)

```python
from nnsight import TransformersModel

gpt2 = TransformersModel("openai-community/gpt2", task="text-generation")
bert = TransformersModel("google-bert/bert-base-uncased", task="fill-mask")
vlm  = TransformersModel("llava-hf/llava-interleave-qwen-0.5b-hf", task="image-text-to-text")
asr  = TransformersModel("openai/whisper-tiny", task="automatic-speech-recognition")
```

Every one of those gets the same tracing API, and the pipeline handles tokenizing, featurizing, chat templating, and batch padding. Leave `task=` off and it's inferred from the checkpoint. You can also hand it an already-loaded `torch.nn.Module` instead of a repo id, which is the escape hatch for checkpoints transformers 5 no longer has a pipeline for.

`LanguageModel` and `VisionLanguageModel` still work. They're thin subclasses now, and they warn on construction telling you what to write instead.

A few tasks can't fit their input into one forward pass, so the pipeline chops it up: a document longer than the model's context becomes several overlapping token windows, and zero-shot classification turns one sentence into one entailment pair per candidate label. Those pieces are batched together as rows of the trace's single forward pass, so a read inside the block comes back with one row per piece rather than one row for the input you passed. They're in the order the pipeline produced them.

You can also attach a PEFT adapter at load:

```python
model = TransformersModel(
    "meta-llama/Llama-3.2-1B",
    task="text-generation",
    peft="some-org/some-lora",
    dispatch=True,
)
```

The adapter is grafted onto the base model at load, so everything you trace afterwards is the adapted model.

### generate and pipe

In v0.7 you called `generate` and then went looking for the token ids on `model.generator.output`, while `generate` itself handed back the pipeline's decoded records. Those are two different jobs, so they're two methods now:

```python
# generate: runs the model's generate, returns token ids
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    ids = tracer.result.save()

# pipe: runs the whole task pipeline, returns its records
with model.pipe("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    records = tracer.result.save()      # [{"generated_text": "..."}]
```

`model.trace(...)` runs one forward. `model.scan(...)` runs one forward under fake tensors, so you can check shapes without loading weights or touching a GPU.

## vLLM is a first-class runtime now

The vLLM integration used to be a side project, bolted on and not something the rest of NNsight was designed around. Enough people have asked for interpretability on a real serving engine that we've made it a primary target in v0.8, and rebuilt the integration around that.

Your block is serialized onto the request and executed *inside* vLLM's worker, interleaved with the forward pass, rather than wrapped around the engine from outside. PagedAttention, continuous batching and tensor parallelism all keep working, and every module stays a location: `.input`, `.output`, `.source` operations, the pre-sampling `model.logits`, and the token `model.samples` actually drew.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    resid = model.model.layers[10].output[0].clone().save()   # [pos, d_model]
    logits = model.logits.save()

print(resid.shape, model.tokenizer.decode(logits.argmax(-1)))
# torch.Size([5, 4096])  Paris
```

Three things differ from a Transformers trace. There's no batch axis, because vLLM packs every in-flight request's tokens into one `[total_tokens, hidden]` slab and NNsight narrows your block to its own request's rows. A decoder layer's output is a `(hidden_states, residual)` pair, and the residual stream leaving the block is their sum. And you have to `.clone()` anything you keep, because vLLM reuses and overwrites activation buffers in place.

### Tracing without giving up throughput

vLLM's decode speed comes largely from replaying CUDA graphs, and a replayed graph runs no Python, so nothing inside one can be served. A default `VLLM(...)` therefore builds an eager engine, where every location is reachable and you pay vLLM's own eager price. That price is vLLM's, not NNsight's: built back to back and asked to generate with no trace running, plain `vllm.LLM(..., enforce_eager=True)` and a dispatched `VLLM(...)` measure the same thing, 69.6 against 69.7 tok/s.

v0.8 adds the other option. Name the locations you need with `taps=`, and they get recorded into the graph as breaks and served on every replay, while everything else stays vanilla vLLM with graphs on:

```python
model = VLLM(
    "Qwen/Qwen3-8B",
    dispatch=True,
    taps=["model.layers.*.output", "model.layers.10.self_attn.o_proj.input"],
)
```

You keep most of the engine's throughput while tracing it:

| | vanilla vLLM | with taps |
|---|---:|---:|
| 8B, 1 GPU | 92 tok/s | 89 (96%) |
| 8B, tp=4 | 229 tok/s | 213 (93%) |
| 8B, tp=8 | 313 tok/s | 284 (91%) |
| 70B, tp=8 | 61 tok/s | 58 (95%) |

Capturing every layer at every step runs at 88 tok/s against vanilla's 92. A tap can also reach inside a forward, so `taps=["model.layers.10.self_attn.source.qkv_split_0.output"]` serves a `.source` operation on every replay. The tradeoff is that only taps are reachable on such an engine, and each one splits the graph, so keep the set small.

### How that compares

We ran the same ten jobs through nnsight, [interp-engine](https://github.com/EleutherAI/interp-engine) and [vLLM-Lens](https://github.com/UKGovernmentBEIS/vllm-lens) on Llama-3.1-70B across four cards. Each dot is a library's throughput as a share of plain vLLM generating the same tokens with nothing attached, so 100% means the tooling costs nothing. Hover a dot for its number and run count.

--8<-- "docs/vllm/assets/throughput-llama-70b.svg"

??? note "The numbers, Llama-3.1-70B, tp=4"

    --8<-- "docs/vllm/assets/throughput-llama-70b.md"

Most of the rows are close. Two aren't. Capturing every layer at every step is where a per-module Python round trip stops being affordable: taps hold 35 tok/s against vanilla's 37, while the alternatives drop to 7 and 10. And the bottom two rows are ✗ for everything but NNsight, because zeroing an attention head or overriding the sampled token means running your own code in the worker instead of picking from a fixed menu of hooks.

### Installing a block on the engine

`model.edit()` sends a block over once and leaves it on the engine:

```python
with model.edit() as (tracer, edit):
    out = model.model.layers[16].output
    hidden = (out[0] + out[1]).clone().save()

# Not traces, just plain vLLM requests. The block still runs for them.
outputs = model.generate(["The Eiffel Tower is in", "The capital of Japan is"],
                         max_tokens=5)
outputs[1].saves["hidden"]
```

Every request the engine runs afterwards gets its own copy with its own scope, including requests from clients that have never heard of NNsight. Name your edits and pick them per request with `edits=[...]`. That's also what makes `nnsight-serve` work: one engine behind HTTP with interventions installed on it, and clients with no GPU on the other end.

For streaming, pass `mode="async"` to the constructor, `VLLM("Qwen/Qwen3-8B", mode="async")`. The engine is then vLLM's `AsyncLLM`, and a trace hands you its outputs as they arrive:

```python
with model.trace("The capital of France is", max_tokens=5) as tracer:
    logits = model.logits.save()

async for output in tracer.backend:       # an attribute, not a call
    print(output.outputs[0].text)
```

### The full breakdown

There's now a whole section of the docs for this, at [nnsight on vLLM](../../vllm/index.md): 23 pages covering locations, capture, attention patterns the paged kernel never forms, steering, activation patching, ablation, SAE features, conditional interventions, generation, tensor parallelism, async and serving, four worked end-to-end examples, the measured performance grid, and a comparison against interp-engine and vLLM-Lens. Every snippet in it was run against Qwen3-8B on an A100, and the outputs shown are what came back.

If you want the design rather than the recipes, Zikai wrote it up in [NNsight × vLLM: Interpretability at Production Scale](vllm-integration.md).

## A new engine: greenlets instead of threads

Your intervention code and the model's forward pass have to run interleaved. Your block runs until it asks for `h[6].output`, then it stops and lets the model run until the forward reaches layer 6, then takes the value and continues. In v0.7 the block ran in an OS worker thread, so every one of those handoffs cost a lock and a GIL handoff between two real threads, and the cost grew as more workers piled up contending for the same locks.

In v0.8 each block runs in its own greenlet, a cooperative coroutine that switches in-process. Only one greenlet runs at a time, so there are no locks and no queues anywhere in the engine. A single `Interleaver` installs persistent pass-through hooks on every module, and workers park and resume against it.

NNsight's own overhead is cheaper as a result, and it grows more slowly as a trace touches more of the model:

![Overhead scaling, v0.7 vs v0.8](nnsight-0.8-assets/scaling-light.svg#only-light)
![Overhead scaling, v0.7 vs v0.8](nnsight-0.8-assets/scaling-dark.svg#only-dark)

| | v0.7 | v0.8 |
|---|---:|---:|
| Opening a trace with nothing in it | 1.12 ms | 0.51 ms |
| Each activation read | 36.7 µs | 11.8 µs |
| Each activation edited | 43.9 µs | 17.1 µs |
| Each extra prompt batched into one pass | 94.3 µs | 15.6 µs |

At the ends of those sweeps, reading 192 activations in one pass costs 8.2 ms of NNsight on v0.7 and 2.8 ms on v0.8. Batching 128 prompts into one forward costs 13.1 ms on v0.7 and 2.6 ms on v0.8.

Read those numbers with two things in mind. They come from a stack of tiny linear layers with the untraced forward subtracted out, so that the model's own compute cancels and what's left is the engine. And on a real model with a handful of interventions you will not notice any of it: a GPT-2 forward on CPU takes roughly 50 ms, and a dozen reads at 11.8 µs each comes to 0.14 ms, about a quarter of one percent. The greenlet engine pays off when you touch a lot of the model at once, which is what caching everything, per-head sweeps, source tracing and large batches all do.

### Ordering mistakes now say what they are

A worker holds one pending request at a time and can only be served locations in the order the model reaches them. So if you ask for the final logits before you ask for layer 8, layer 8 has already fired and its value is gone:

```python
with model.trace(prompt):
    logits = model.output.logits.save()             # runs the whole model
    hidden = model.transformer.h[8].output.save()   # too late
```

```
OutOfOrderError: 'model.transformer.h.8.output.i0' was requested but the model
already ran past it
```

v0.7 told you the value "was not provided" and suggested investigating why the module wasn't called, which sent people hunting for a module-path typo that wasn't there. v0.8 distinguishes a location that never ran from one the model has already gone past, says which, and points the traceback at the line that was waiting. `model.output` is the one that catches people first: it's the root of the tree, the very end of the forward pass, so reading it before any layer strands that layer's request.

## Reaching inside a forward pass with `.source`

Plenty of the values you want aren't a module's output. They're computed halfway through a `forward` and thrown away: an intermediate in an MLP, the attention pattern before it's cast and dropped out, a running state inside a loop. There's no submodule to attach to, so historically you reimplemented the forward pass to get at them.

`module.source` makes every call site inside a `forward` addressable. NNsight rewrites the forward's AST so each operation is bracketed by the interleaver, and you get the same handles a module has, `.input`, `.output` and `.skip`, one level finer.

Printing it shows the forward with every operation labelled at its call site, and works outside a trace:

```python
print(model.transformer.h[0].mlp.source)
```

```
                     * def forward(self, hidden_states: ...) -> torch.FloatTensor:
 self_c_fc_0     ->  0     hidden_states = self.c_fc(hidden_states)
 hidden_states_0 ->  +     ...
 self_act_0      ->  1     hidden_states = self.act(hidden_states)
 hidden_states_1 ->  +     ...
 self_c_proj_0   ->  2     hidden_states = self.c_proj(hidden_states)
 hidden_states_2 ->  +     ...
 self_dropout_0  ->  3     hidden_states = self.dropout(hidden_states)
 hidden_states_3 ->  +     ...
                     4     return hidden_states
```

You can drill further: calling `.source` on an operation reaches inside the function that operation called, resolved from whatever value is actually flowing through the call at run time. Here are GPT-2's raw attention probabilities, read out of the softmax inside the attention interface, before the dtype cast and dropout that the pattern on `attn.output[1]` has already been through:

```python
model = TransformersModel("openai-community/gpt2", dispatch=True,
                          attn_implementation="eager")

with model.trace("The Eiffel Tower is in the city of"):
    interface = model.transformer.h[0].attn.source.attention_interface_1
    probs = interface.source.nn_functional_softmax_0.output.save()

print(probs.shape)          # torch.Size([1, 12, 10, 10])
print(probs[0, 0].sum(-1))  # rows sum to 1
```

NNsight installs the instrumentation lazily, on first `.source` access, and outside a trace each operation calls straight through. Decorated forwards, closures, and forwards that call `super()` are all instrumented now: a wrapper gets peeled and rebuilt around the instrumented function rather than refusing. `SourceNotAvailable` now means only that there is no Python source to instrument, because the target is a builtin or a C function.

## Models that don't fit on one GPU

### Quantization

Holding a model in fewer bits per weight normally means building a quantizer config and knowing which of transformers' several backends your format belongs to. That's a lot of ceremony for a choice that's really just *how wide is a weight*, so in v0.8 it goes where you'd write the dtype:

```python
model = TransformersModel(
    "meta-llama/Llama-3.2-3B",
    task="text-generation",
    dtype="nf4",              # where you would write "bfloat16"
    dispatch=True,
)
```

`nf4`, `int4` and `4bit` all mean NF4, while `fp4`, `int8` and `fp8` mean what they say. Nothing else about your code changes, since module paths and activations are unchanged. You'll need `bitsandbytes` and `accelerate` installed; neither ships with NNsight. `fp8` needs a GPU of compute capability 8.9 or better and is refused below it, because transformers itself will quietly load bfloat16 at twice the width you asked for instead.

### Tensor parallelism

A model too big for one GPU can be split across several with transformers' native tensor parallelism, where each rank holds a slice of every attention and MLP projection. That's different from `device_map="auto"`, which puts whole layers on different GPUs and runs them one after another.

The catch for interpretability is that on any one rank, a sharded module's activation is only that rank's slice of the real tensor. NNsight gathers the slices before your intervention sees the value and re-splits whatever you leave behind, so the trace you write is the trace you'd write against one GPU:

```python
# torchrun --nproc_per_node=4 tp_trace.py
from transformers.distributed import DistributedConfig
from nnsight import TransformersModel

model = TransformersModel(
    "meta-llama/Llama-3.2-3B",
    task="text-generation",
    dispatch=True,
    distributed_config=DistributedConfig(tp_size=4),
)

with model.trace("The Eiffel Tower is in the city of"):
    # gate_proj is column-parallel: each rank computes 2048 of these 8192
    # features. Read it and you get all 8192.
    gate = model.model.layers[5].mlp.gate_proj.output.save()

print(gate.shape)  # (1, 11, 8192) on every rank, not (1, 11, 2048)
```

There's nothing to install or enable. This runs on the DTensor backend transformers introduced in 5.16, so that's the floor. On 5.15 and earlier nothing is recognized as sharded and your trace quietly sees one rank's slice.

For a model you'd rather shard under a serving engine, [vLLM](#vllm-is-a-first-class-runtime-now) does tensor parallelism too, and gathers sharded activations the same way.

## Docs and agents

A lot of NNsight code is written with an agent in the loop now, and when we watched agent runs against the old docs they got NNsight wrong in specific, repeatable ways. So v0.8 ships 105 documentation pages written as recipes, covering usage, concepts, patterns, errors, gotchas, remote, models and developing, with a `CLAUDE.md` at the repo root that routes an agent to the right one. We checked every claim in them by executing it, which turned up a pile of things the old docs asserted and the library didn't do.

The [skills repository](https://github.com/ndif-team/skills) installs as a plugin for Claude Code and Codex, [Context7](https://context7.com/ndif-team/nnsight) serves the docs to any MCP-compatible client, and `NNsight.md` is a design-and-implementation manual for when an agent needs to know why something works the way it does. There's also a [runnable walkthrough notebook](https://colab.research.google.com/github/ndif-team/nnsight/blob/main/NNsight_Walkthrough.ipynb) whose every cell is verified on CPU.

## Other changes

- `tracer.cache(...)` records many modules at once, every layer and every generation step, without per-module `.save()` calls.
- `module.skip(replacement)` bypasses a module entirely, `tracer.stop()` exits a run early, and `tracer.barrier(n)` synchronizes value sharing across invokes.
- `model.edit()` stores interventions on the envoy so every later trace replays them, on a copy by default so the original model stays clean.
- `model.session()` bundles several traces so values flow between them without `.save()`.
- `@eproperty` is the descriptor behind `.input`, `.output` and `tracer.result`, and it's public, so you can declare your own hookable values on a model subclass.
- `DiffusionModel` wraps any `diffusers` pipeline, UNet- or transformer-based, with `seed=`, per-invoke batching, and denoising-step iteration. `automodel=` picks the class the weights load through.
- `remote="local"` runs the whole serialize, deserialize and execute round trip in-process, so you can catch serialization problems offline without a key or a queue.
- `pip install nnsight` now gets you a working local and remote install. transformers, huggingface-hub and the remote dependencies are core, with `dev`, `vllm` and `serve` extras on top.
- `nnsight login` stores an NDIF key the way `huggingface-cli login` does, and verifies it.
- Python 3.10 through 3.14 are supported, and CI tests both ends of that range.

## Upgrading from v0.7

The rewrite kept the API's shape, but a handful of things moved. Two of them can change your numbers without raising anything, so start with those.

### `.source` labels shifted

Every assignment in an instrumented forward is now an operation in its own right, sharing the per-name counter with calls. GPT-2's attention call moved from `attention_interface_0` to `attention_interface_1`, and `attention_interface_0` is now the line that chooses the implementation. Asking for the old label doesn't raise; it returns the assigned value instead. If you have code that names source operations, print the source and re-check the labels.

### A loop that outruns the run is cut short

`tracer.iter[:8]` over three generated steps, or an open `iter[:]` or `all()`, now ends where the model ends, with a warning. Values saved inside the loop are kept and the statements after it are discarded, so the result looks complete while being shorter than the bound you asked for. v0.7 bounded `all()` internally, so code written against it should hold the run to the count (`min_new_tokens=` on transformers, `min_tokens=` or `ignore_eos=True` on vLLM) or move its trailing statements into a separate `tracer.invoke()`.

### Everything else raises

| v0.7 | v0.8 |
|---|---|
| `LanguageModel(...)` / `VisionLanguageModel(...)` | `TransformersModel(repo, task=...)`; the old names work but warn |
| `model.generator.output` | `tracer.result` (same tensor); `model.pipe(...)` for decoded records |
| `x.save()` outside a trace (silent no-op) | raises; call it inside the trace |
| `saved.value` | a saved value is the value |
| `tracer.next()` / `module.next()` | gone; use `for step in tracer.iter[...]:` |
| `with tracer.iter[...]:` | deprecated in favor of the `for` form |
| `nnsight.apply/log/cond/iter/session(...)` | plain Python and `model.session()` |
| `nnsight.list/dict/int/...` wrappers | plain Python containers |
| `CONFIG.APP.CROSS_INVOKER` / `CACHE_DIR` / `TRACE_CACHING` | gone |
| `tracer.local()` | not ported |

Custom model classes that implement `_prepare_input` or `_batch` will need updating to the new signatures.

Everything still reachable under an old name warns under `nnsight.NNsightDeprecationWarning`. That's a `FutureWarning`, not a `DeprecationWarning`, on purpose: Python's default filters only show a `DeprecationWarning` raised at the top level of the running script, so a package or helper module being ported would have warned to nobody. Silence NNsight's alone with:

```python
warnings.filterwarnings("ignore", category=nnsight.NNsightDeprecationWarning)
```

NNsight registers no filters of its own. The full mapping is in [`docs/reference/version-history.md`](https://github.com/ndif-team/nnsight/blob/0.8/docs/reference/version-history.md).

## What's next

Everything above is available today as a pre-release: `pip install nnsight --pre`. It runs
locally, on your own hardware, and we would like you to break it before the full release does.

**The full release is at the beginning of October**, when v0.8 goes live on NDIF. That is the
piece the pre-release is missing: `remote=True`, and with it the models you cannot host yourself.
Until then, remote execution stays on v0.7.

NDIF is getting more than a version bump alongside it, and that deserves its own post rather than
a paragraph here. [Join the Discord](https://discord.gg/6uFJmCSwW7) if you want to hear about it
first, and keep an eye on this blog.

**Found a bug?** Please tell us. A pre-release is exactly when it is most useful to hear, and
[GitHub issues](https://github.com/ndif-team/nnsight/issues) is the best place, since we can tie
it to a fix. Porting problems from v0.7 count as bugs, especially the silent kind in
[Upgrading from v0.7](#upgrading-from-v07).

**Office hours.** We are running two sessions to help people port code, talk through the new
API, and answer whatever comes up. Both are at 11am Eastern on
[this Zoom link](https://northeastern.zoom.us/j/97460947245):

- Tuesday, September 15
- Tuesday, September 22

---

Docs: [nnsight.net](https://nnsight.net) · GitHub: [github.com/ndif-team/nnsight](https://github.com/ndif-team/nnsight) · Discord: [discord.gg/6uFJmCSwW7](https://discord.gg/6uFJmCSwW7)
