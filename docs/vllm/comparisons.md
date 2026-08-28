# Comparisons

Two other libraries put interpretability on vLLM: [interp-engine](https://interp-engine.org)
(Neuronpedia / Decode Research) and [vLLM-Lens](https://github.com/UKGovernmentBEIS/vllm-lens)
(UK AISI). This page compares nnsight's vLLM integration with each, job by job, and ends with
one throughput grid run over all three on the same machine. Every other page in this section
avoids comparison; this one is nothing else.

---

## interp-engine

[interp-engine](https://interp-engine.org) is the interpretability engine behind Neuronpedia: a
fixed vocabulary of 34 named *points* (`resid_post.10`, `mlp_act.5`, ...) served on a hooked vLLM
backend, a CUDA-graph `vllm-static` backend, and a HuggingFace eager backend, with a validator
that checks its captures against TransformerLens and nnsight across 50+ architectures. It is
good, careful software, and this section deliberately follows the shape of
[its documentation](https://www.interp-engine.org/docs/) page for page so the two can be read
side by side.

This page is the comparison the others avoid. It was written against interp-engine 1.3.4 and
nnsight 0.8 on the same machine, and the throughput table at the bottom is one grid run over
both.

### The one-sentence difference

interp-engine answers a **closed question well**: *give me point P, steered by spec S.* nnsight
runs **your Python inside the engine's forward**: any module, any computation, any write, on the
same request path. Most of what follows is that difference worked through each job.

### Addressing

| | interp-engine | nnsight |
| --- | --- | --- |
| Unit | `Address(name, layer[, stream])`, 34 canonical names | a module on the tree + a side (`.input`, `.output`, `.inputs`, `.source.<op>`) |
| Layer index | flattened forward order | the tree's own indices (`layers[10]`) |
| Discovery | `model.points()`, the visualizer, `SUPPORTED_POINTS.md` | `print(model)` on the meta tree, `.source` for the ops inside a forward |
| Portability across families | the point means the same tensor on every family (`attn_out_post` vs `attn_out` on sandwich-norm models is handled for you) | you address vLLM's module for *this* family; the sum of a layer's tuple is the residual on the standard families, and you read the tree to know |
| Missing on vLLM | `mlp_pre`, `mlp_pre_linear`, `lm_head`, `attn_gate`, `expert_weights`, `expert_indices` (fused away); `attn_scores`/`attn_probs` by recompute | `gate_up_proj.output` split in two is `mlp_pre` and `mlp_pre_linear`; `logits_processor(lm_head, h)` is the unembed; the pattern is a recompute here too; expert *selection* is inside the fused kernel for both |

The table on [Locations](locations.md#qwen3-8b-by-name) is, in effect, interp-engine's point
list rewritten as nnsight locations on Qwen3-8B.

### Loading

| interp-engine | nnsight |
| --- | --- |
| `load_model(id, backend="vllm")` — hooked, eager, every point | `VLLM(id)` — every location |
| `backend="vllm-static", static_points=[...]` — graphs on, declared taps | `VLLM(id, taps=[...])` — graphs on, declared taps |
| `backend="vllm-generate"` — graphs + compile, no capture | plain `vllm.LLM`, or `model.generate(...)` outside a block on any engine |
| `backend="eager"` (HuggingFace; gradients, batches, `attn_probs`) | `TransformersModel(id)` — the same block, with gradients |
| `warmup()` / `shutdown()` | `dispatch=True` / process exit |
| `num_gpus=4` | `tensor_parallel_size=4` |
| `configure_static(points)` after construction | taps are fixed at construction |

Both static/tapped engines sit on the same seam — vLLM's breakable CUDA graphs, a callable
recorded at capture and run on every replay — and both turn `torch.compile` off to get it.
interp-engine's `"auto"` static set is `resid_post` at every layer, read *and* write;
nnsight's taps are whatever module locations you name, read and write.

### Capabilities

interp-engine asks first: `model.hooks_available`, `model.points()`, `model.grad_support`,
`CapabilityUnsupported` naming the capability and an alternative. nnsight has less to ask
because there is no point table to be absent from — the tree is the capability — and refuses
in the same spirit: a non-tap read, a barrier, a chunked prompt, a misspelled sampling keyword
and a typo'd tap each raise with the location and the fix
([Capabilities and limits](capabilities.md)).

### Reading

| Job | interp-engine | nnsight |
| --- | --- | --- |
| one point | `run_with_cache(model, tokens, [point])` | `x = loc.clone().save()` in a trace |
| every layer | a list of 36 addresses | a loop over `model.model.layers` |
| while generating | `capture_generation(...)` | `tracer.all()` |
| a batch | eager only | many invokes, on vLLM, batched by the scheduler |
| MoE routing | `router_logits`; selection eager-only | `mlp.gate.output`; selection fused, same |
| per-head contributions | `head_contributions(model, cache, 10)` | `z[:, h*d:(h+1)*d] @ W_O[:, h*d:(h+1)*d].T` in the block |
| direct logit attribution | client-side, from captured points | every layer and head in one forward, in the worker ([Attribution](attribution.md)) |
| SAE features | capture the point, encode on the client | the same, or a feature's live activation in the worker from two rows ([SAE features](sae.md)) |
| gradients | eager only | HuggingFace path only |

Where the two differ in *kind*: interp-engine's cache is assembled on the client from what the
worker shipped; nnsight's block runs on the worker and ships what you saved. Capturing every layer
of a 70B model at every step costs interp-engine 3.5× its single-layer rate and nnsight nothing
measurable (table below), because the clones never leave the worker until the end.

### Attention

Neither library has the attention pattern as a hook on vLLM — the paged kernel never forms it.
interp-engine rebuilds it inside `capture_attention` from captured q/k; on nnsight you rebuild it
in the block from `self_attn.attn.inputs`, in a dozen lines, and both match HuggingFace's eager
attention to bf16 noise. interp-engine additionally ships `per_head_value`, `attn_out_gate` and
`split_fused_qkv` for the family-specific layouts; on nnsight those are slices you write, and the
[Attention](attention.md) page gives them for Qwen3. Under tensor parallelism the two part ways:
interp-engine's per-head `z` and direct attribution are single-GPU only (`num_gpus > 1` shards the
heads across ranks and its off-kernel recompute sees one shard), while nnsight gathers `o_proj.input`
whole before the block reads it, so the per-head material above works at any `tensor_parallel_size`.

### Logit lens

interp-engine's `decode_residuals` is a method that applies the family's post-unembed arithmetic
and runs on the client from captured residuals (a `topk` variant runs on the worker). nnsight
calls the model's own `norm` and `logits_processor` in the block, on the worker, so the vocab-wide
tensor never travels unless you save it — and it works at every layer in one forward on vLLM,
where interp-engine's `layer_logits` is eager-only.

### Writing

| | interp-engine | nnsight |
| --- | --- | --- |
| Operations | `AddSpec`, `OrthogonalDecompSpec`, `ProjectionCapSpec`, in order per layer | any expression |
| Where | `resid_post` by default; `point="z"` and the stream points | any location — a head's slice of `o_proj.input`, a router logit, the logits, the sampled id |
| Positions | `position_mask`, `SteerMask.SPECIAL_TOKENS` | a boolean mask on the rows |
| During generation | the spec is applied every step | put the edit under `tracer.iter[:N]` |
| Conditional / stateful | no | yes: `if step == 3`, a running estimate, a probe's output deciding the write ([Conditional interventions](conditional.md)) |
| Activation patching | no write of another run's activation | a saved tensor written at a position, per layer or per head, batched as invokes ([Activation patching](patching.md)) |
| Ablate a component | no write off the residual points | `mlp.output[:] = 0`, a head's slice of `o_proj.input`, a neuron, a router logit ([Ablation](ablation.md)) |
| SAE feature clamp | add a fixed decoder direction | scale the feature's *live* activation through generation ([SAE features](sae.md)) |

The `ablate` and `force` rows of the grid — zeroing a head's slice of `o_proj.input`, and
overriding the sampled token — are the two the interp-engine harness could not express on any
backend.

### Generating

`generate_stream` yields a `GenStep` per token with `n_logprobs`; on vLLM its `.logits` is
`None` because the sampler never ships the tensor out of the worker. nnsight's `model.logits` is
that tensor, on the worker, readable *and writable* every step, and `logprobs=k` rides
`tracer.result` for the portable case. Streaming text is `generate_stream` on both backends in
interp-engine and `mode="async"` in nnsight; `generate_full` and `tracer.result` are the same
vLLM `RequestOutput`.

### Chat and tokens

interp-engine's `Tokenize` helper is the richer one: `message_partition`, `message_spans`,
`GeneratedTurnSpans`, `compose_assistant_turns`, with the DeepSeek-V4 template quirks handled.
nnsight hands you the HuggingFace tokenizer and the [Chat and tokens](chat.md) page shows
offset-mapping spans in a few lines.

### Serving

| | interp-engine | nnsight |
| --- | --- | --- |
| In-process async | every method is `async`; `sync_model` facade | `mode="async"`; `tracer.backend` streams |
| Concurrency | `asyncio.gather` over `capture(...)` | `asyncio.gather` over traces |
| A server | your FastAPI app, model built in `lifespan` | `nnsight-serve`, or your app around `VLLM` |
| Clients without a GPU | your API | `VLLM(id)` meta tree + `trace(..., serve=url)` |
| Instrument every request | a spec in a `steer()` context | `model.edit()` — one block, every request, any tenant |

### Correctness

interp-engine's validator compares its points against TransformerLens and nnsight/nnterp across
50+ architectures at early, middle and late layers, with the results checked into the
repository; that is a level of cross-engine validation nnsight does not have. nnsight's vLLM path
is covered by ~195 tests on two GPUs, and every value shown in this section was checked against
a HuggingFace forward of the same checkpoint. Its sharded path is checked value by value against
a one-rank engine of the same checkpoint — Qwen2.5 at `tp=2`, DeepSeek-V2-Lite at `tp=2` and at
`tp=4, dcp=2` — and the request accounting under preemption, `n > 1`, aborted streams and foreign
tenants sharing the batch has a test each. Both libraries note the same trap: a value that is the
right shape from the wrong place raises nothing.

### Throughput {#throughput}

See [Throughput, measured](#throughput-measured) at the end of the page — one grid over all
three libraries. The interp-engine columns there are `vllm` (hooked, eager) and `vllm-static`
(CUDA graphs, declared taps); `vllm-generate` is left out because it is vanilla vLLM under
another name (its numbers matched the vanilla column to within 1%), and the eager HuggingFace
backends are compared in the text below rather than plotted.

What the grid says about interp-engine:

- **Under graphs the two are the same engine.** `nnsight taps` and `IE vllm-static` are within
  noise of each other and of vanilla vLLM on plain generation and single-layer capture, at every
  size and parallelism — as they should be, sitting on the same vLLM seam.
- **Both eager engines pay the same tax**, and it is the driver's: 86 vs 86 tok/s on one GPU,
  ~70 tok/s flat as GPUs are added.
- **Where the computation happens is the difference that scales.** Every-layer capture and the
  logit lens ship tensors to interp-engine's client per request; nnsight keeps them on the
  worker. At 70B: 35 vs 10 tok/s for every-layer capture, 35 vs 22 for the lens.
- **Gradients are the one row interp-engine leads**, on its eager HuggingFace backend: one
  forward+backward at 70B takes 531 ms there against 1,216 ms on nnsight's `TransformersModel`.

---

## vLLM-Lens

[vLLM-Lens](https://github.com/UKGovernmentBEIS/vllm-lens) (UK AISI, MIT) is a vLLM *plugin*: it
registers through vLLM's `general_plugins` entry point, so an unmodified `vllm serve` or offline
`LLM` gains activation capture, steering vectors and Garçon-style hooks the moment the package is
installed, driven by `SamplingParams.extra_args` (offline) or `vllm_xargs` (over the OpenAI API).
It also ships an Inspect AI model provider and a set of examples — causal tracing, logit and
Jacobian lens, a deception probe, an emotion tracker, an activation oracle. This section was
written against vLLM-Lens 1.2.1 on vLLM 0.27.1, the same engine version as the rest of this site.

The [Examples](examples/causal-tracing.md) group of this section is the vLLM-Lens example set
redone in nnsight, so each can be read against its original.

### The one-sentence difference

vLLM-Lens exposes **one seam** — a decoder layer's residual stream, on the way in (pre-hook) or
out (post-hook) — and lets you capture it, add to it, or run a pickled function on it. nnsight
exposes **every module** and runs your block interleaved with the forward: the attention
projections, the per-head outputs, the router, the logits, the sampled id, and the residual
stream are all locations, and the same block reads and writes any of them.

### Where a hook can fire

| | vLLM-Lens | nnsight |
| --- | --- | --- |
| Residual stream leaving a block | `output_residual_stream=[l]`, `Hook(layer_indices=[l])` | `sum(layers[l].output)` |
| Residual stream entering a block | `Hook(..., pre=True)` | `layers[l].input_layernorm.output[1]`, `embed_tokens.output` |
| q / k / v, per-head `z`, `o_proj` input, MLP neurons, the router | — | any of them ([Locations](locations.md)) |
| Pre-sampling logits, the sampled id | — (logprobs through the API) | `model.logits`, `model.samples`, both writable |
| Inside a module's forward | — | `.source` ops |

Both libraries take the residual stream to be vLLM's `(hidden, residual)` summed, and both
clone before handing it to user code (vLLM-Lens clones for you; on nnsight a kept reference
must be cloned — [Locations](locations.md#clone-what-you-keep)).

### Reading

| Job | vLLM-Lens | nnsight |
| --- | --- | --- |
| Capture layers | `extra_args={"output_residual_stream": [15, 20]}` → `out.activations["residual_stream"]`, `(layers, pos, d)` | `.save()` on the location; `tracer.cache()` |
| Every step of a generation | captured per forward pass, stacked | `tracer.all()` / `tracer.iter` |
| Compute on the worker | `Hook(fn)`; results in `ctx.saved`, returned as `hook_results` | the block itself |
| Parameters under TP / PP | `ctx.get_parameter(name)` gathers; `prefetch_params` for PP | activations are gathered; `logits_processor(lm_head, h)` for the unembed ([Tensor parallelism](tensor-parallel.md)) |
| A sweep of many prompts | `register_hooks` once, `generate` per prompt, `collect_hook_results` | `model.edit()` once, `generate(prompts)`, values on each output |
| Batch of prompts | `llm.generate(prompts, params)` | `generate(prompts)` (plain) or one invoke per prompt (traced) |

### Writing

| | vLLM-Lens | nnsight |
| --- | --- | --- |
| Additive steering | `SteeringVector(activations, layer_indices, scale, norm_match, position_indices)` | `layers[l].output[0][:] += scale * v` under `tracer.iter` |
| Norm-matched | `norm_match=True`: `h += scale · ‖h‖ · v/‖v‖` | `h += scale * h.norm(dim=-1, keepdim=True) * v / v.norm()` |
| Position-specific | 3-D activations + `position_indices` | index the rows |
| Anything else | return a tensor from a `Hook` — at a layer boundary | any expression at any location, including the sampler |
| Persistent | `register_hooks` | `model.edit()` |

### Serving

This is where vLLM-Lens is strongest. It lives *inside* `vllm serve`: the OpenAI-compatible
completions and chat endpoints accept `vllm_xargs` for capture, steering and hooks, the server
gains `/v1/hooks/*` for persistent hooks and parameter prefetch, activations come back base64-
encoded in the response, and any OpenAI client — or Inspect, through the bundled provider — can
drive it. nnsight's equivalent is [`nnsight-serve`](serving.md#nnsight-serve), a single-model
server that runs nnsight traces submitted by GPU-less clients and installs engine-wide edits;
it does not speak the OpenAI API. An edit installed on an nnsight engine does run on every
request the engine serves, whoever sent it, which is the persistent-hook pattern; but the
front door for OpenAI-style traffic is vLLM-Lens's.

| | vLLM-Lens | nnsight |
| --- | --- | --- |
| Server | `vllm serve` + plugin | `nnsight-serve` |
| Client protocol | OpenAI API + `vllm_xargs`; `VLLMLensClient` | nnsight traces over HTTP (`serve=url`) |
| Code on the server | cloudpickled hook functions (arbitrary code; trusted clients only) | serialized trace blocks (likewise) |
| Persistent instrumentation | `/v1/hooks/register` | `model.edit(serve=url)` |
| Inspect AI | provider built in | — |
| Streaming | the API's | `mode="async"` |

### Parallelism and engines

| | vLLM-Lens | nnsight |
| --- | --- | --- |
| Tensor parallel | steering and hooks on every rank; capture on rank 0; a hook that saves Python lists sees them `tp_size`× | every rank runs the block; reads are gathered whole; rank 0 reports |
| Pipeline parallel | yes (`prefetch_params` for cross-stage weights) | no |
| Expert parallel / MoE | yes | yes, incl. MoE partial-sum gather |
| CUDA graphs | never — the plugin forces `enforce_eager` for every engine in the process | `taps=` keeps replay ([Performance](performance.md)) |
| LoRA | yes (`lora_request`; the activation-oracle example) | yes (`lora_request` in the sampling kwargs) |
| Installed alongside other engines | `VLLM_LENS_DISABLE=1` to make it a no-op | nothing is patched until `VLLM(...)` is built |

The last row matters operationally: vLLM-Lens patches `EngineArgs.create_engine_config` and
`LLM.generate` at import, so every vLLM engine in a process that has it installed runs eager
with the worker extension attached — including one you did not mean to instrument.

### The examples

| vLLM-Lens example | Mechanism there | On nnsight |
| --- | --- | --- |
| `causal_tracing.py` | pre-hook noise on the subject embeddings, post-hook restore, one HTTP request per `(layer, position)` | [Causal tracing](examples/causal-tracing.md) — one trace per layer, one invoke per position, batched by the scheduler |
| `logit_lens.py` | hook with `ctx.get_parameter("lm_head.weight")`, manual RMSNorm | [Logit lens](logit-lens.md) — the model's own `norm` and `logits_processor` |
| `jacobian_lens.py`, `jacobian_lens_chat.py` | hook applying a fitted `J_l`, prefetched weights; lens fit separately on prime-rl | [Jacobian lens](examples/jacobian-lens.md) — Neuronpedia's fitted lens, read out per step; `edit()` for the chat pattern |
| `deception_probe.py` | persistent hooks over contrastive prompts, LBFGS probe | [A linear probe](examples/probe.md) — `edit()` over the prompts; then the probe runs *inside* the model every step |
| `emotion_tracker.py` | persistent hooks for direction vectors, per-token projections via chat | [Concept directions](examples/emotion-tracker.md) |
| `activation_oracle.py` | capture, then norm-matched positional steering under a LoRA oracle | expressible (norm-matched positional write + `lora_request`); not reproduced here — it needs the 70B oracle adapter |
| `extract_residual_stream.ipynb` | per-request and persistent capture, offline and HTTP | [Capture](capture.md), [Async and servers](serving.md) |

### Throughput {#vllm-lens-throughput}

vLLM-Lens is the fifth series in [Throughput, measured](#throughput-measured); its column ran
in its own environment (the plugin forces eager mode on every engine in a process) on the same
cards, and the grid gained one row for it: a **sweep** of 1024 short prompts at one token each,
capturing one layer — the activation-extraction workload vLLM-Lens is built for.

- **Plain generation and single-layer capture are a wash.** Both libraries hook the same seam
  and both run the engine eagerly by default: 86–87 tok/s plain, 77–79 capturing one layer.
  nnsight's taps column is the only one that keeps CUDA graphs, and the only one within a few
  percent of vanilla.
- **vLLM-Lens pays per layer and per hook.** Its hooks are installed on every decoder layer and
  each does its bookkeeping for every in-flight request on every step, and a steering vector or
  hook clones the layer's output: capturing all 32 layers halves throughput (37 vs nnsight's 68
  tok/s), and a steering vector, a probe or a lens each cost about 40% (48–49 vs 78). nnsight's
  block visits only the locations it names.
- **The sweep is close.** vLLM-Lens's capture rides the request with little per-request setup
  (1.32 s over 1024 prompts, 1.8× vanilla). An nnsight trace serializes a block per invoke and
  collects per step (1.6 s, 2.1×); the intended shape for a sweep is
  [`model.edit()`](serving.md#edit-the-engine), which installs the block once — 1.1 s eager,
  1.0 s under taps, the fastest capture of the three.
- **Two rows vLLM-Lens cannot express**: an ablation inside the attention block and an override
  of the sampled token, because its hook points are layer boundaries and it has no hook on the
  sampler.

A trap found while measuring, worth knowing on the nnsight side: writing
`model.model.layers[16].output` *inside* the block references the model, and each of the 1024
invokes then serializes it — 8.6 s for the sweep. Binding the layer envoy before the trace and
using it inside is what the grid shows. Invisible in a single trace, decisive in a sweep
([Performance](performance.md#what-else-moves-the-number)).

- **Tensor parallelism.** vLLM-Lens installs its hooks on every rank; at tp=4 plain generation
  matches the other eager engines (76 tok/s vs 69–72), single-layer capture too (69), but a
  steering vector or probe costs a quarter (58, 57) where the eager nnsight engine and
  interp-engine lose nothing, and every-layer capture drops to 37 (nnsight eager 58, taps 208).
  At 70B/tp=4 the same shape: plain generation and one-layer capture on par or slightly ahead,
  steering, probe and lens at 18–19 tok/s against the eager engines' 27–29.

---

## Throughput, measured {#throughput-measured}

One harness, all three libraries, the same machine: bf16, A100-80GB, vLLM 0.27.1,
transformers 5.15, 512-token prompt, 128 new tokens, greedy, prefix caching off on every
engine, 3 processes × 3 timed runs per cell (mean; std ≤ 2% except the HuggingFace-eager rows,
which are not plotted). Each dot is a library's throughput on a workload as a **share of plain
vLLM** doing the same generation with nothing attached — vanilla `generate` for single-stream
rows, vanilla 8-concurrent `generate` for the ×8 rows, and vanilla's own sweep time for the
sweep rows (where less time is more). Hollow dots are eager engines, filled dots keep CUDA
graphs; hue is the library. The whisker through each dot is the min–max over its repeated runs
(nine for most cells: three processes × three timed runs; the 70B and tp=8 panels have six).
A cell that is statistically significantly faster than its nnsight counterpart — the eager
engines and vLLM-Lens against `nnsight eager`, `vllm-static` against `nnsight taps` — by an
exact two-sided Mann-Whitney U test at p < 0.05 and at least 3% apart is **bold** in the tables;
a bold nnsight cell beat every counterpart in its row by the same test. Hover a dot for the
number, the run count and the range.

### Other models and scenarios

The same grid on more models and two more workload shapes (three processes each; the
DeepSeek panel has no interp-engine column, whose runner was not pointed at it):

- **Qwen3-8B** (36 layers with QK-norm, the model this section's examples use). The eager
  engines are all at ~74% of vanilla on plain generation (67–69 tok/s vs 91) where on
  Llama-8B they were at 93%: the eager tax scales with the number of module calls per layer, and
  Qwen3 has more of them. The graph engines are unaffected (nnsight taps 88, interp-engine
  static 88). Everything else has the Llama shape — nnsight leads on every-layer capture and the
  lens, interp-engine's hooked engine leads by ~10% on single-layer capture, steering and the
  probe, vLLM-Lens pays ~35% for a hook or steering vector.
- **Qwen1.5-MoE-A2.7B** (60 experts, 4 active). The eager engines collapse to 20% of vanilla
  (40 tok/s vs 210): an MoE layer is many small kernels, and each is a Python round trip on an
  eager path. nnsight's taps keep 98% (205); interp-engine's static engine 86% (181); vLLM-Lens
  is an eager engine and sits with the eager nnsight and interp-engine at 40–42. At 8 concurrent
  requests nnsight taps (1,091) and interp-engine static (874) both *exceed* vanilla (671),
  which turns out to be vLLM's `torch.compile` path losing on this model — see the note under
  the MoE chart.
- **DeepSeek-V2-Lite** (MLA attention + MoE, 27 layers, 16B). Same picture, stronger:
  eager engines at 19% of vanilla (31 vs 164), nnsight taps at 98% (161). The MLA path
  (`kv_a_proj`, `kv_b_proj`, the absorbed decode kernel) adds module calls that only a
  graph engine hides.
- **Llama-3.2-1B.** With a small model the fixed per-step Python cost dominates: the eager
  engines are at 37–39% of vanilla, the graph engines at 94–95%, and vLLM-Lens's hook costs
  show most clearly (steering 116 vs interp-engine 145, every-layer capture 96 vs nnsight 119).
- **Long context** (2048-token prompt, 512 new tokens, Llama-8B). Longer decode amortizes
  per-step overheads: every eager engine moves up a few points relative to the 512/128 grid, the
  ordering is unchanged, and vLLM-Lens's per-hook cost persists (steering 50, probe 49 vs the
  eager engines' 75–84).
- **32 concurrent requests** (Llama-8B). Plain generation is a wash for every engine
  (1,430–1,470 tok/s vs vanilla 1,523). Capturing one layer on all 32 streams costs nnsight
  6–10% (1,330 eager, 1,367 taps), interp-engine 4–8% (1,408 static, 1,371 hooked) and
  vLLM-Lens 25% (1,100): at this batch size the per-request bookkeeping that all three do on
  every step is what shows, and the graph engines' decode advantage is mostly spent.

--8<-- "docs/vllm/assets/throughput-llama-8b.svg"

??? note "The numbers — Llama-3.1-8B, one GPU (tok/s; sweeps in seconds)"

    --8<-- "docs/vllm/assets/throughput-llama-8b.md"

--8<-- "docs/vllm/assets/throughput-llama-8b-tp2.svg"

??? note "The numbers — Llama-3.1-8B, tp=2"

    --8<-- "docs/vllm/assets/throughput-llama-8b-tp2.md"

--8<-- "docs/vllm/assets/throughput-llama-8b-tp4.svg"

??? note "The numbers — Llama-3.1-8B, tp=4"

    --8<-- "docs/vllm/assets/throughput-llama-8b-tp4.md"

--8<-- "docs/vllm/assets/throughput-llama-70b.svg"

??? note "The numbers — Llama-3.1-70B, tp=4"

    --8<-- "docs/vllm/assets/throughput-llama-70b.md"

--8<-- "docs/vllm/assets/throughput-qwen3-8b.svg"

??? note "The numbers — Qwen3-8B, one GPU"

    --8<-- "docs/vllm/assets/throughput-qwen3-8b.md"

--8<-- "docs/vllm/assets/throughput-qwen-moe.svg"

The vanilla column on this model is vLLM's default engine: CUDA graphs *and* `torch.compile`.
On this MoE the compiled path is the slow one at 8 concurrent requests. Measured directly on
plain vLLM, same prompts and settings: eager 315 tok/s; `torch.compile` alone 245 (the
compiled forward is *slower* than eager on this model); compile + CUDA graphs, the default,
669; CUDA graphs with compilation off 1,095 (1,099 in the breakable-graph mode nnsight's taps
use). Enabling vLLM's own custom kernels under compile (`custom_ops=all`) changes nothing
(668), so it is the inductor-compiled forward itself, not a missing fused op. nnsight taps
(1,091) and interp-engine static (874) are therefore not beating vLLM; they run the un-compiled
graph path that vLLM's default loses to on this model. Single-request generation is unaffected
(210 default vs 205 taps).

??? note "The numbers — Qwen1.5-MoE-A2.7B, one GPU"

    --8<-- "docs/vllm/assets/throughput-qwen-moe.md"

--8<-- "docs/vllm/assets/throughput-llama-1b.svg"

??? note "The numbers — Llama-3.2-1B, one GPU"

    --8<-- "docs/vllm/assets/throughput-llama-1b.md"

--8<-- "docs/vllm/assets/throughput-llama-8b-long.svg"

??? note "The numbers — Llama-3.1-8B, 2048-token prompt, 512 new tokens"

    --8<-- "docs/vllm/assets/throughput-llama-8b-long.md"

--8<-- "docs/vllm/assets/throughput-llama-8b-x32.svg"

??? note "The numbers — Llama-3.1-8B, 32 concurrent"

    --8<-- "docs/vllm/assets/throughput-llama-8b-x32.md"

--8<-- "docs/vllm/assets/throughput-deepseek-v2-lite.svg"

??? note "The numbers — DeepSeek-V2-Lite, one GPU"

    --8<-- "docs/vllm/assets/throughput-deepseek-v2-lite.md"

--8<-- "docs/vllm/assets/throughput-qwen35-08b.svg"

??? note "The numbers — Qwen3.5-0.8B, one GPU"

    --8<-- "docs/vllm/assets/throughput-qwen35-08b.md"

--8<-- "docs/vllm/assets/throughput-qwen35-4b.svg"

??? note "The numbers — Qwen3.5-4B, one GPU"

    --8<-- "docs/vllm/assets/throughput-qwen35-4b.md"

--8<-- "docs/vllm/assets/throughput-qwen36-moe.svg"

??? note "The numbers — Qwen3.6-35B-A3B (MoE), tp=2"

    --8<-- "docs/vllm/assets/throughput-qwen36-moe.md"

The three panels above are **hybrid gated-delta-net trunks** (Qwen3.5-0.8B: 18 of 24 layers are
recurrent), and they needed a correctness fix before any taps number could be published: a full
CUDA graph captured over such a trunk silently miscomputes prefill (plain vLLM does, with
compilation off — the recurrent layers branch on the batch's prefill/decode composition), so a
tapped nnsight engine now pins `cudagraph_mode="FULL_DECODE_ONLY"` on any model vLLM reports as
hybrid or attention-free: prefill runs eagerly, decode keeps replay, and tapped generation
matches eager exactly. Two consequences show in the numbers. The taps-to-vanilla gap is wider
here (60–76%) than on standard trunks (93–97%), because vanilla's `torch.compile` genuinely pays
on these new architectures and taps run without it — yet taps still lead every graph-mode
alternative on these models. And the eager engines collapse hardest of any model measured
(8–25% of vanilla, identically for all three libraries): a recurrent layer is many small kernels,
each a Python round trip, and vanilla is very fast. The head-ablation row is ✗ under taps here
because only layer outputs are tapped on these trunks.

--8<-- "docs/vllm/assets/throughput-llama-8b-tp8.svg"

??? note "The numbers — Llama-3.1-8B, tp=8"

    --8<-- "docs/vllm/assets/throughput-llama-8b-tp8.md"

--8<-- "docs/vllm/assets/throughput-llama-70b-tp8.svg"

??? note "The numbers — Llama-3.1-70B, tp=8"

    --8<-- "docs/vllm/assets/throughput-llama-70b-tp8.md"

The two panels above extend the tensor-parallel series to **all eight cards** (three trials for
8B, two for 70B, on a shared node — runs that overlapped another user's job were dropped, and
the trials that remain agree within 3%). The trend from tp=2 and tp=4 simply continues. Plain
vLLM keeps scaling (8B: 92 → 148 → 229 → 313 tok/s from one to eight cards; 70B: 37 → 61 from
four to eight), and the graph engines follow it: capturing a layer every step, nnsight taps
hold 91% of vanilla on 8B and 95% on 70B, interp-engine static 80% and 90%. The eager engines
do not move at all — nnsight eager, interp-engine's hooked engine and vLLM-Lens sit at 64–74
tok/s on 8B at *every* card count, which is now 20–24% of vanilla, and at 28–32 on 70B (46–53%;
the heavier step hides more of the per-module handoff). Where the libraries differ is what they
serve under graphs: every-layer capture on 70B is 57 tok/s under taps against 5–10 for the
three engines that gather the tensors out of the worker. One number goes the other way: the
`edit()`-once sweep, the fastest capture on one GPU, gets slower with every rank added while the
per-request trace does not — on 8B, 1.05 s at tp=2, 1.35–1.46 s at tp=4, 2.2 s at tp=8, against
a per-request trace steady at 1.2–1.4 s (70B/tp=8: 5.7 against 5.1). An installed block runs
its saves on every rank; a per-request one collects once. Past four cards, trace the sweep.

The harness (`ie-bench/`: one runner per library, `common.py` for the rows, `report.py` and the
chart script), the raw `results-*.jsonl` and every log live alongside nnsight's tests; every
number above was produced by it.
