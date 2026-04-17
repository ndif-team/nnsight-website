---
date: 2026-04-16
authors:
  - jaden
categories:
  - Guide
---

# Extending NNsight: From Custom Envoys to Your Own Model Class

*By Jaden Fiotto-Kaufman*

NNsight works out of the box on any `torch.nn.Module`. Wrap it, open a trace, read `.output`, save it. For a lot of interpretability work, that's all you need.

But the longer you spend doing this work, the more patterns you notice. You catch yourself writing the same six-line projection chain for every layer of a logit-lens sweep. You reshape attention heads in every single notebook. You wrap a model that isn't on HuggingFace and discover you now have to rebuild tokenization, batching, and generation by hand. You start wanting NNsight to speak *your* model's vocabulary.

NNsight is designed to be extended at exactly these points. This post is a cookbook of the extension surface — from lightweight per-module conveniences all the way down to custom execution backends. Pick the cheapest primitive that solves your problem; don't reach for a custom backend when a three-line eproperty will do.

<!-- more -->

## The menu

From cheapest to deepest:

| Extension point | Use when… |
| --- | --- |
| `rename={…}` on construction | The module names differ from what your code expects. Pure aliasing — no new behavior. |
| **Custom `eproperty`** | You want a new *property* on a module (or an `OperationEnvoy`) that computes a derived view of inputs/outputs. |
| **Custom `Envoy` subclass** + `envoys={…}` mapping | You want to attach methods or multiple eproperties to a specific module type (e.g. every `LlamaAttention`). |
| **Custom root `NNsight` subclass** | You want to wrap a model family with its own loading, tokenization, batching, or generation contract — the niche `LanguageModel`, `DiffusionModel`, and `VLLM` fill. |
| **Custom `Tracer`** | You need a new execution mode that differs from `trace` / `generate` / `edit` / `scan` at the *control-flow* level. |
| **Custom `Backend`** | You need to dispatch the compiled intervention somewhere NNsight doesn't know about (another process, an inference engine, an NDIF-style job queue). |

The rest of this post is six recipes, each with a *before* and *after*, grounded in real code that ships in NNsight or lives in its orbit.

---

## Recipe 0 — `rename={…}`: the zero-code customization

Before reaching for anything on the class hierarchy, check whether the thing you want is just an aliasing problem. A lot of friction when writing research code that works across architectures comes from each model family using a different path for the same structural component. GPT-2's transformer layers live at `transformer.h`, LLaMA puts them at `model.layers`, OPT buries them under `model.decoder.layers`. Code you write against one can't be pasted against another without a find-replace.

`rename=` on any Envoy constructor — including on `NNsight` subclasses — lets you publish alternative names onto the envoy tree. It takes a dict; the keys are the current paths, and the values are either a single alias or a list of aliases:

```python
from nnsight import LanguageModel

model = LanguageModel(
    "openai-community/gpt2",
    rename={
        "transformer.h": "layers",             # single alias
        "transformer.ln_f": "final_norm",
        ".transformer": ["model", "backbone"],  # multiple aliases
    },
)

with model.trace("hello"):
    # Both paths refer to the same Envoy — either works.
    h0_new = model.layers[0].output[0].save()
    h0_old = model.transformer.h[0].output[0].save()
```

Three shapes worth knowing:

- **Simple rename** — `{"transformer.h": "layers"}` exposes `model.layers` as an alias for `model.transformer.h`. The original path still works; you've added a name, not replaced one.
- **Multiple aliases** — `{".transformer": ["model", "backbone"]}` makes both `model.model` and `model.backbone` reach the same envoy. Useful when you want a short name for quick work and a long name that reads well in papers.
- **Mount from below** — `{".model.layers": ".layers"}` promotes a deep module to the root, so `model.layers` hits what was previously `model.model.layers`. The leading dot distinguishes "a path from the root of this envoy" from "a child name."

### Where this fits in the cookbook

`rename=` alone covers a surprising amount of cross-architecture plumbing. If all you want is "let me write `model.layers[i].self_attn` regardless of whether this is GPT-2 or LLaMA," no new classes are required — a dict bound to a per-architecture config is enough. [nnterp's architecture dispatch](https://github.com/ndif-team/nnterp) is built on exactly this: a rename config per model family applied on construction.

Where `rename=` stops being enough is when you want *new behavior* on a module, not just new names for it — a `.heads` property, a `.lens` projection, a tuple-unpacked `.hidden`. That's when you cross over into Recipe 1.

---

## Recipe 1 — Custom `eproperty`: turn a projection chain into a property

### The motivation

Logit lens, or really any "apply a shared projection to every layer's output," looks like this:

```python
with model.trace(prompt):
    tokens_per_layer = []
    for block in model.transformer.h:
        hidden = block.output[0]
        logits = model.lm_head(model.transformer.ln_f(hidden))
        tokens_per_layer.append(logits.argmax(dim=-1).save())
```

Five lines of plumbing per layer. The thing you actually care about — "what does this layer predict?" — is two tokens deep in the loop body. Worse, the `block.output[0]` unpacking is a per-architecture quirk: a lot of transformer blocks return a tuple `(hidden_states, attention_weights, …)`, and "get me the actual residual stream" is something you're writing out longhand in every notebook.

A custom `eproperty` collapses all of that into `block.hidden`.

### The recipe

An `eproperty` is a descriptor for a hookable property on any object that looks like an `IEnvoy` (anything with `.interleaver` and `.path`). The stub method you decorate is a *placeholder* — the real work is done by whichever `requires_*` decorator from [`nnsight.intervention.hooks`](https://github.com/ndif-team/nnsight/blob/main/src/nnsight/intervention/hooks.py) you stack on top, plus the optional `preprocess` / `transform` callbacks:

```python
from nnsight.intervention.envoy import Envoy, eproperty
from nnsight.intervention.hooks import requires_output


class BlockEnvoy(Envoy):
    """Exposes `.hidden` = the residual stream at the output of a transformer block."""

    @eproperty(key="output")
    @requires_output
    def hidden(self):
        """The block's hidden-state output, unpacked from the tuple layout."""

    @hidden.preprocess
    def hidden(self, value):
        return value[0] if isinstance(value, tuple) else value
```

- `@eproperty(key="output")` says "this property reads from the `output` hook point." Multiple eproperties can share a key (e.g. `Envoy.input` and `Envoy.inputs` both read `"input"`) to provide different views on the same underlying value.
- `@requires_output` registers the one-shot forward hook that will eventually produce the value. Without it (or an equivalent hook-registering decorator), the worker thread's `request()` would block forever.
- The body of `hidden(self)` is never executed — it's a carrier for the decorators and a donor of the name / docstring. `help(block.hidden)` will show your docstring verbatim.
- `@hidden.preprocess` runs on `__get__` before the user sees the value. Here it's a pure tuple unpack; in real code this is where you'd crunch the value into whatever view is ergonomic.

You attach this envoy to a specific module type via the `envoys=` mapping (covered in the next recipe) or, for one-off experiments, by wrapping the block directly.

### When edits need to flow back: `transform`

`preprocess` controls *what the user sees*. If it returns a derived object — a clone, a reshape, a view — and the user mutates it in place, those edits don't necessarily reach the model. Whether they do depends entirely on whether the preprocessed object still shares storage with the model's tensor.

Two patterns to remember:

**Views share storage** — the simplest case.

```python
@heads.preprocess
def heads(self, value):
    # [B, S, hidden] → list of [B, S, head_dim] views.
    B, S, H = value.shape
    n = self._module.n_heads
    return list(value.view(B, S, n, H // n).unbind(dim=2))
```

Each list entry is a view onto the same storage as the original `value`. `heads[3][:] = 0` mutates the model's tensor directly; the downstream forward pass sees it. No `transform` needed.

**Clones break the link** — you need `transform` to close the loop.

```python
@thing.preprocess
def thing(self, value):
    return value.clone()     # user gets a fresh tensor

@thing.transform
@staticmethod
def thing(value):
    return value             # swap the (post-edit) clone back into the model
```

At `__get__` time the preprocessed value is captured by the transform via `functools.partial`, parked on the current mediator, and fired after the user has had a chance to mutate it. Whatever the transform returns is then `batcher.swap`ped back into the running forward pass. Any reshape you want — from `[B, n_heads, S, head_dim]` back to `[B, S, hidden]`, for instance — happens in the transform.

Rule of thumb: **if `preprocess` returns a view, you don't need `transform`. If it returns a clone or a fresh tensor, you do — assuming you want edits to propagate.**

---

## Recipe 2 — Custom `Envoy` + `envoys={…}`: attention heads as a list

### The motivation

Every attention module in every architecture exposes its output as `[batch, seq, hidden]`, where `hidden = n_heads * head_dim`. Every researcher eventually writes this:

```python
B, S, H = attn_out.shape
per_head = attn_out.view(B, S, n_heads, H // n_heads).transpose(1, 2)
# ...edit head 4...
attn_out_new = per_head.transpose(1, 2).reshape(B, S, H)
```

It works, but it's fragile. Off-by-one on `transpose`, forget `.contiguous()`, mix up `view` vs. `reshape`, and you're debugging quietly-wrong activations. It also has to be repeated in every cell.

### The recipe

Define an `Envoy` subclass for your attention module, give it a `.heads` property that returns a list of per-head views, and use the `envoys={…}` mapping on the root to apply it to every `LlamaAttention` (or `GPT2Attention`, or whatever your model family uses) in the tree:

```python
from nnsight import NNsight
from nnsight.intervention.envoy import Envoy, eproperty
from nnsight.intervention.hooks import requires_output


class AttnEnvoy(Envoy):
    """Exposes attention output as a list of per-head `[B, S, head_dim]` views."""

    @eproperty(key="output")
    @requires_output
    def heads(self):
        """Per-head views of the attention output. Mutate in place to edit a head."""

    @heads.preprocess
    def heads(self, value):
        n_heads = self._module.n_heads
        B, S, H = value.shape
        return list(value.view(B, S, n_heads, H // n_heads).unbind(dim=2))


model = NNsight(my_net, envoys={LlamaAttention: AttnEnvoy})

with model.trace(prompt):
    heads = model.layers[5].self_attn.heads.save()
    heads[3][:] = 0                 # zero head 3 — in place, on the model's tensor
    out = model.output.save()
```

Because `.unbind` returns views that share storage with the original attention output, `heads[3][:] = 0` mutates the model's tensor directly. No `transform`, no clone, no reshape dance on the return path. The test suite for this exact pattern lives at [`tests/test_transform.py`](https://github.com/ndif-team/nnsight/blob/main/tests/test_transform.py) if you want a runnable reference.

### How `envoys={…}` works

`envoys` accepts three shapes:

- `None` (default) — every descendant is a plain `Envoy`.
- An `Envoy` subclass — used for every descendant.
- A `Dict[Type[torch.nn.Module], Type[Envoy]]` — for each descendant, NNsight walks the module's MRO and uses the first matching class. Unmatched modules fall back to `Envoy`.

The mapping is propagated through the entire subtree, so setting it once on the root reaches every leaf. It lives on `Envoy.__init__` and on `NNsight` as a class attribute — subclasses (see Recipe 3) can set `envoys = {…}` as a default for all instances, and users can still override per-instance with `envoys=…` at construction time. Pass `envoys=None` to opt out.

### Where this replaces boilerplate

Any library that currently proxies module access through a hand-rolled accessor class — `LayerAccessor`, `AttnAccessor`, `HeadAccessor` — is a candidate for this pattern. Instead of maintaining a parallel object hierarchy that does manual `.output[0]` unpacking and tuple-return detection, you define a custom `Envoy` per module type, attach the eproperties you want, and wire them in with a single `envoys={…}`. Users get autocomplete, `help()` works, and the proxy layer goes away.

---

## Recipe 3 — Custom root `NNsight` subclass: wrap a whole model family

### The motivation

You have a model that isn't on HuggingFace — say, a custom speech encoder, an in-house code model with a bespoke tokenizer, or a research prototype. `NNsight(model)` gets you tracing immediately, but every downstream script has to handle tokenization, batching, and generation by hand. That's the boundary a custom root subclass crosses.

Every higher-level wrapper that ships with NNsight — `LanguageModel`, `VisionLanguageModel`, `DiffusionModel`, `VLLM` — is a subclass of `NNsight`. They all share the same shape: override some loading hooks, supply a `_prepare_input` / `_batch` contract so multi-prompt invokes work, and optionally publish a convenience `trace` or `generate` that sets sensible defaults.

### The recipe

A minimal skeleton:

```python
import torch
from nnsight import NNsight
from nnsight.intervention.envoy import Envoy


class MyModel(NNsight):
    """Root envoy for the MyModel family.

    Subclasses NNsight to add model-specific loading, batching, and a
    pre-wired `envoys` mapping so every attention block is wrapped with
    `AttnEnvoy` (from Recipe 2) automatically.
    """

    # Applied to every descendant in the tree by default.
    envoys = {MyAttention: AttnEnvoy, MyBlock: BlockEnvoy}

    def __init__(self, name_or_model, *args, **kwargs):
        # Handle the "load from name" vs. "wrap pre-loaded" cases.
        if isinstance(name_or_model, str):
            module = self._load_from_name(name_or_model, **kwargs)
        else:
            module = name_or_model
        super().__init__(module, *args)

    def _load_from_name(self, name, **kwargs):
        # Whatever loading logic your model family needs.
        ...

    def _prepare_input(self, *inputs, **kwargs):
        # Called once per invoke. Convert user-facing inputs (strings, dicts,
        # whatever) into the (args, kwargs, batch_size) triple nnsight uses
        # downstream. See LanguageModel._prepare_input for a worked example.
        ...

    def _batch(self, batched_inputs, *inputs, **kwargs):
        # Called when multiple invokes need to be concatenated into one
        # forward pass. Only needs to exist if your model supports batched
        # invokes with differing inputs.
        ...

    def generate(self, *inputs, **kwargs):
        # Optional model-specific convenience; your model family may not
        # have a "generate" concept at all.
        ...
```

Two things worth calling out:

**`envoys = {…}` as a class attribute.** Because subclasses of `NNsight` pick up the class-level `envoys` automatically (via a `kwargs.setdefault("envoys", type(self).envoys)` in `NNsight.__init__`), you can ship a distribution of useful Envoy subclasses with your model wrapper and users never have to think about the mapping. They just get `model.layers[5].self_attn.heads` for free. They can still override per-instance with `envoys=…`, or opt out with `envoys=None`.

**`_prepare_input` / `_batch` only matter if you support multi-invoke batching.** Base `NNsight` works fine for single-invoke traces without these. You only need them when users are expected to do `with tracer.invoke("a"):` `with tracer.invoke("b"):` against the same session. Look at [`LanguageModel._prepare_input`](https://github.com/ndif-team/nnsight/blob/main/src/nnsight/modeling/language.py) for the canonical tokenization-and-packing example; [`DiffusionModel._batch`](https://github.com/ndif-team/nnsight/blob/main/src/nnsight/modeling/diffusion.py) for a non-LM case.

### Wiring cross-tree references

The one place a root subclass shines is *wiring cross-tree references into child envoys*. The logit-lens pattern from the opening is the canonical case: every `BlockEnvoy` needs access to `self.lm_head` and `self.transformer.ln_f`, which live elsewhere in the tree.

```python
class LensLanguageModel(LanguageModel):
    envoys = {GPT2Block: LensBlockEnvoy}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # After the tree is built, wire the lens components into each block.
        for block in self.transformer.h:
            block.__dict__["_lens_norm"] = self._module.transformer.ln_f
            block.__dict__["_lens_head"] = self._module.lm_head
```

Inside `LensBlockEnvoy.hidden.preprocess` you then reach for `self._lens_norm` and `self._lens_head` directly, and the `block.lens` eproperty can wrap it all up:

```python
class LensBlockEnvoy(Envoy):
    @eproperty(key="output")
    @requires_output
    def lens(self):
        """Logit-lens projection of this block's hidden state."""

    @lens.preprocess
    def lens(self, value):
        hidden = value[0] if isinstance(value, tuple) else value
        return self._lens_head(self._lens_norm(hidden))

with model.trace(prompt):
    per_layer_tokens = [b.lens.argmax(-1).save() for b in model.transformer.h]
```

The five-line loop from the opening becomes a comprehension.

### In the wild

For a production subclass of `LanguageModel` that re-hosts every major open-weights transformer under a single naming convention, see [nnterp](https://github.com/ndif-team/nnterp). It's a good reference for the loading-and-renaming side of the problem — architecture dispatch, tokenizer quirks, validation. It currently builds its per-layer and per-head accessors as hand-rolled proxy classes (`LayerAccessor`, `AttentionProbabilitiesAccessor`); the `envoys={…}` mapping plus custom eproperties of the kind we showed in Recipes 1 and 2 would let that whole proxy layer fold into declarative `Envoy` subclasses.

---

## Recipe 4 — Custom `Tracer`: a new execution mode

### The motivation

The context manager you open with `with model.trace(...)` (or `.generate`, `.edit`, `.scan`) is a `Tracer`. It captures your intervention code via AST extraction, compiles it into a function, hands it to a `Backend`, and coordinates mediator threads during execution. If the execution mode you want isn't any of the built-ins, you need a new tracer.

The existing ones are a good catalog of what "new execution mode" means in practice:

| Tracer | File | What's unusual about it |
| --- | --- | --- |
| `InterleavingTracer` | `tracing/tracer.py` | The baseline — real forward pass with thread-based intervention. |
| `ScanningTracer` | `tracing/tracer.py` | Uses fake tensors for shape validation without running the model. |
| `EditingTracer` | `tracing/editing.py` | Skips execution entirely; records interventions as persistent default mediators on the Envoy. |
| `IteratorTracer` | `tracing/iterator.py` | Drives generation-step iteration (`tracer.iter[:]`). |
| `BackwardsTracer` | `tracing/backwards.py` | A separate interleaving session for `.grad` access inside `with tensor.backward():`. |
| `Invoker` | `tracing/invoker.py` | Each `tracer.invoke(...)` is one of these — a child tracer scoped to a single input. |
| `StreamTracer` / `RemoteInterleavingTracer` | `modeling/mixins/remoteable.py` | NDIF remote execution, blocking and streaming variants. |

If none of those fit — if you want, say, a tracer that runs the intervention twice for differential analysis, or one that forks its workers across multiple CUDA streams — you subclass `Tracer` or `InterleavingTracer` and override the few methods that matter for your case.

### The contract

A tracer has three responsibilities:

1. **Capture** — extract the source of the `with` body via `Tracer.capture()`, parse it into an `Info` object with filename, line numbers, and frame-local references.
2. **Compile** — in `Tracer.compile()`, wrap the captured source into a function definition that the backend will turn into a code object. This is where you inject setup or teardown (e.g. `ScanningTracer` wraps the body in a fake-tensor context).
3. **Execute** — in `Tracer.execute(fn)`, call the compiled function and return its result. This is where `InterleavingTracer` starts the mediator workers.

The cleanest subclass to read if you want the whole picture in under 80 lines is [`EditingTracer`](https://github.com/ndif-team/nnsight/blob/main/src/nnsight/intervention/tracing/editing.py): it runs the intervention function without actually executing the model, collects the mediators it produces, and parks them on `Envoy._default_mediators` so subsequent traces re-apply them automatically. That's a complete new execution mode in ~40 lines.

### When to reach for it

You probably don't need a new tracer. The existing ones cover "run locally," "shape-check," "generate N tokens," "differentiate," and "persist edits." A custom tracer is for when your execution mode doesn't fit any of those shapes — and it's worth asking first whether a custom *backend* (next recipe) would do instead, since backends and tracers overlap in power.

---

## Recipe 5 — Custom `Backend`: dispatch somewhere new

### The motivation

The backend is what happens to your compiled intervention function once the tracer has it. `ExecutionBackend` runs it locally. `RemoteBackend` serializes it and posts it to NDIF. `AsyncVLLMBackend` folds it into vLLM's sampling parameters so a remote engine can run it asynchronously. If you need your traces to land somewhere *else* — a Modal container, a Ray cluster, an internal inference service, a job queue of your own design — that's a backend.

There are two tiers of "custom," and the tier you want depends on how different your environment is from the ones NNsight already supports.

### Tier 1 — Configure an existing backend

You often don't need to subclass anything. `RemoteBackend` is parameterized in ways that can be re-tuned without touching its source. The pattern Workbench uses for serving NDIF jobs from a FastAPI app is a minimal, idiomatic example:

```python
# workbench/_api/state.py (excerpt)
from nnsight.intervention.backends.remote import RemoteBackend

def make_backend(self, model=None, job_id=None):
    if self.remote:
        return RemoteBackend(
            job_id=job_id,
            blocking=False,
            model_key=model.to_model_key() if model is not None else None,
        )
    return None
```

Three deliberate choices:

- `blocking=False` — the FastAPI handler returns a job ID immediately so the browser can poll for completion, rather than holding the HTTP connection open while NDIF does its thing.
- `model_key=…` — routes the job to a specific deployment on the NDIF side.
- Environment-driven `CONFIG.API.HOST` and `CONFIG.API.KEY` — lets Workbench swap between NDIF production, staging, and a local fake, just by changing env vars before `RemoteBackend` reads the config.

No subclass needed. Before writing one, check whether the behavior you want is reachable through `RemoteBackend`'s constructor kwargs or NNsight's `CONFIG`.

### Tier 2 — Subclass `Backend`

When configuration isn't enough, the contract is small. `Backend.__call__(tracer)` has one job: turn the tracer's captured source into a compiled function, then decide what to do with it. The base class handles the compile step; you override `__call__`, call `super().__call__(tracer)` to get the function back, and dispatch.

The minimal local case — `ExecutionBackend` — is exactly this:

```python
# src/nnsight/intervention/backends/execution.py
class ExecutionBackend(Backend):
    def __call__(self, tracer):
        fn = super().__call__(tracer)     # compile
        try:
            Globals.enter()
            return tracer.execute(fn)     # run locally with interleaving
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()
```

Three lines of interesting logic. The `Globals.enter() / exit()` pair installs the pymount C extension (which is what makes `obj.save()` work on arbitrary objects) and resets it on exit; `tracer.execute(fn)` starts the mediator threads; `wrap_exception` remaps any exceptions back to the user's source lines.

A sketch of a custom dispatching backend follows the same shape:

```python
from nnsight.intervention.backends.base import Backend
from nnsight.intervention.tracing.globals import Globals
from nnsight.intervention.tracing.util import wrap_exception


class MyQueueBackend(Backend):
    """Submits traces to an internal job queue and polls for results."""

    def __init__(self, queue_client, *, blocking=True):
        super().__init__()
        self.queue = queue_client
        self.blocking = blocking
        self.job_id = None

    def __call__(self, tracer):
        fn = super().__call__(tracer)         # get the compiled function
        payload = serialize_for_queue(fn, tracer.info)
        self.job_id = self.queue.submit(payload)

        if not self.blocking:
            return None                       # caller polls via backend()

        try:
            Globals.enter()
            return self.queue.wait(self.job_id)
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()
```

The pattern that's interesting here isn't the queue specifically — it's that a non-blocking backend returns `None` from `__call__`, and the user polls via `tracer.backend()` later. `RemoteBackend` does exactly this for NDIF; `AsyncVLLMBackend` returns a streaming generator so the user can iterate over partial results.

### When the backend doesn't execute the function

`AsyncVLLMBackend` is the most interesting real example because it *doesn't run the compiled function at all.* Instead it compiles the function, walks the resulting mediators, serializes them into vLLM `SamplingParams`, and submits to the async engine — vLLM handles execution, and the backend returns an async generator for the caller to iterate over.

That's the shape of the most ambitious custom backends: compile for the tracer's benefit, then repurpose the intervention graph entirely. Anyone building an inference-engine integration for a new engine (TensorRT-LLM, SGLang, a proprietary runtime) is likely to end up in this territory.

### NDIF's server-side backend

NDIF itself runs a custom backend on the server side to execute submitted jobs. It's not open source, but its shape is what you'd expect: deserialize the incoming `RequestModel`, reconstruct the intervention graph, dispatch `tracer.execute` locally on a GPU it owns, stream results back to the `RemoteBackend` that sent them. The fact that the local `ExecutionBackend` and the remote NDIF backend share the same compile-and-execute contract is why `remote=True` is a single flag instead of a different API.

---

## Closing

The extension surface maps onto the layers of what NNsight actually does:

- **Properties** (eproperty) layer new *views* on existing hook points.
- **Envoys** layer those views onto specific *module types*.
- **Root NNsight subclasses** layer model-family-specific loading, batching, and defaults on top of Envoys.
- **Tracers** control *how the intervention code gets captured and compiled*.
- **Backends** control *where the compiled result runs*.

Pick the layer that matches the level of customization you actually need. Most problems you'll run into are eproperty or Envoy problems — lean into those before reaching for a custom tracer. The tests for the features used in the first two recipes live at [`tests/test_envoys.py`](https://github.com/ndif-team/nnsight/blob/main/tests/test_envoys.py) and [`tests/test_transform.py`](https://github.com/ndif-team/nnsight/blob/main/tests/test_transform.py) and are runnable out of the box if you want to start by tweaking them. For the deeper layers, the production implementations inside `src/nnsight/modeling/` — `LanguageModel`, `DiffusionModel`, `VLLM` — are the best reference.

If you build something interesting on top of these, drop a link in the [NDIF forum](https://discuss.ndif.us/).
