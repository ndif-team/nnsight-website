# Comparisons

Three other libraries put interpretability on vLLM:
[interp-engine](https://interp-engine.org) (Neuronpedia),
[vLLM-Lens](https://github.com/UKGovernmentBEIS/vllm-lens) (UK AISI) and
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), whose bridge can run
on a vLLM engine. This page says where each one's approach differs from nnsight's, and what
that lets you do or not do.

The main difference is the same in all three. nnsight runs a Python block that you write
inside the engine, at any module. The others offer a fixed set of places and operations, and
in exchange give you things a general block does not.

The capability comparison was made from the source of interp-engine 1.14.3, vLLM-Lens 1.3.0
and TransformerLens 4.4.0.

## At a glance

| | nnsight | interp-engine | vLLM-Lens | TransformerLens |
| --- | --- | --- | --- | --- |
| You write | A Python block that runs in the worker | Requests for named points, and steering specs | Capture and steering arguments on a request, or a pickled Python hook | Hook names to cache, and edit specs |
| It reaches | Any module, operations inside a forward, the logits, the sampled token | 28 named points: residual stream, attention, MLP, router | The residual stream at each layer. Queries and keys, read-only | The outputs of the embedding, each layer, attention block and MLP, and the final norm |
| A write can be | Any expression | One of six fixed operations | A steering vector, or any expression in a hook | One of four fixed operations, with values set before the forward |
| Copy an activation from another run | Yes | No | Yes, in a hook | One vector per hook point |
| Logits and the sampled token | Read and write | No | No | Logits, read-only, rebuilt on the client |
| During generation | Yes | Yes | Yes | No, one forward pass per call |
| Several requests in one batch | Yes | Yes | Yes | Only in eager mode |
| CUDA graphs | With taps | With its `vllm-static` backend | No, always eager | Yes, with `torch.compile` |
| Tensor parallelism | Yes | Yes | Yes | Yes, one prompt per forward |
| A server | `nnsight-serve`, its own protocol | Its own HTTP server | Stock `vllm serve`, through the OpenAI API | None |
| Something that runs on every request | `model.edit()`, any block | `set_steering`, steering only | Persistent hooks | No |
| vLLM version | 0.33 | 0.28.0 | 0.30.0 | 0.20 |

None of the four takes gradients through a vLLM forward pass.

## interp-engine

interp-engine exposes a fixed vocabulary of named points, such as `resid_post.10` and
`mlp_out.5`, and a fixed set of steering operations. It has a hooked eager backend and a
`vllm-static` backend that keeps CUDA graphs, built on the same vLLM mechanism as nnsight's
taps.

What it gives you that nnsight does not:

- **Point names that mean the same tensor on every model family.** With nnsight you address
  vLLM's module tree for the model you loaded, and read the tree to know what a module returns.
- **An answer, before any forward pass, about which points a checkpoint serves.**
- **Steering inside full CUDA graphs**, through a compiled kernel. nnsight's taps need piecewise
  graphs.
- **Built-in tools:** a Jacobian lens that runs on the worker, probe and SAE projections,
  helpers for chat-message token spans, and a validator that checks its points against
  TransformerLens and nnsight.
- **A server that takes JSON**, so clients do not send code.

What nnsight gives you that it does not:

- **Any computation at any module.** interp-engine's six operations add, scale, project out,
  cap or swap along fixed directions. A write that depends on a condition, on state from an
  earlier step, or on another run's activation is not expressible.
- **Component ablation.** There is no operation that zeroes or replaces a head or an MLP.
- **The logits and the sampled token.** On vLLM, interp-engine returns top log-probabilities
  only.

## vLLM-Lens

vLLM-Lens is a vLLM plugin. Once it is installed, a stock `vllm serve` or `LLM` accepts
capture, steering-vector and hook arguments on each request. Its hooks are on the residual
stream at each decoder layer, and a hook can be a pickled Python function.

What it gives you that nnsight does not:

- **The OpenAI-compatible API.** Any OpenAI client can ask for activations or steering, with
  chat templates and tool calling, and it ships a provider for Inspect AI evals.
  `nnsight-serve` speaks its own protocol.
- **Steering without code on the server.** A steering vector is sent as JSON.
- **Pipeline parallelism.**
- **Attention patterns built in**, from captured queries and keys.

What nnsight gives you that it does not:

- **Locations other than the residual stream.** vLLM-Lens has no hook on an MLP, a head's
  output, the router, the logits or the sampled token.
- **CUDA graphs.** vLLM-Lens sets `enforce_eager` on every engine in a process where it is
  loaded.
- **Errors returned to the caller.** vLLM-Lens logs an exception in a hook and skips the hook.
- **Cost that follows what you use.** vLLM-Lens installs its hooks on every layer and runs
  them for every request in the batch. An nnsight block visits only the modules it names.

## TransformerLens

`RemoteBridge.boot_vllm` gives you TransformerLens's `run_with_cache` over a vLLM engine. Its
hooks are installed before vLLM compiles the model, so they become part of the compiled graph.
Each hook copies its output into a buffer and applies `output * scale + bias`. An edit changes
`scale` and `bias` between forward passes.

What it gives you that nnsight does not:

- **`torch.compile` and full CUDA graphs, with reads and writes inside them.** nnsight's taps
  turn `torch.compile` off.
- **TransformerLens's hook names and API.** The same code runs on its HuggingFace backend.
- **Logits for the whole sequence**, and so a per-token loss, from one forward pass.
- **Pipeline parallelism.**

What nnsight gives you that it does not:

- **Generation.** The bridge runs one forward pass per call.
- **Python in the forward pass.** A write is `suppress`, `scale`, `add` or `set` with a value
  fixed in advance, so it cannot depend on the activation.
- **Module inputs, queries, keys and values, each head's output, and the sampled token.**
- **Batching with compiled graphs.** The compiled path takes one prompt per forward. Its
  batched mode is eager, and it is refused under tensor parallelism.
- **A persistent edit, and a server.**
- **Recent vLLM.** TransformerLens pins `vllm<0.21`.

## Throughput

One harness ran the same workloads on every library, on the same machine: bf16, A100-80GB,
a 512-token prompt, 128 new tokens, greedy decoding, prefix caching off.

These measurements are older than the capability comparison above. They used nnsight 0.8,
interp-engine 1.3.4 and vLLM-Lens 1.2.1 on vLLM 0.27.1, and TransformerLens 4.0.0 on vLLM
0.20.2. interp-engine has since moved more work onto the worker, which these numbers do not
show.

Each dot is a library's throughput as a share of plain vLLM doing the same generation.
Hollow dots are eager engines and filled dots keep CUDA graphs. TransformerLens is read
against plain vLLM 0.20.2.

--8<-- "docs/vllm/assets/throughput-llama-8b.svg"

??? note "The numbers: Llama-3.1-8B, one GPU (tokens per second, sweeps in seconds)"

    --8<-- "docs/vllm/assets/throughput-llama-8b.md"

In the tables, a bold cell is significantly faster than its nnsight counterpart, `·` means not
measured, and `✗` means the library cannot express the workload.

What the measurements showed:

- **With CUDA graphs, nnsight's taps and interp-engine's static backend ran at the same
  speed**, within a few percent of plain vLLM. They use the same vLLM mechanism.
- **Every eager engine paid the same cost for eager mode**: 86 tokens per second against plain
  vLLM's 92 on one GPU, and no gain from more GPUs.
- **Work kept on the worker was faster.** Reading every layer on every step ran at 68 tokens
  per second on nnsight's default engine and 88 with taps, against 48 and 71 for
  interp-engine's two backends and 37 for vLLM-Lens.
- **A vLLM-Lens hook or steering vector cost about 40%** (49 tokens per second against 78),
  because its hooks run on every layer for every request.
- **TransformerLens ran the single-forward rows only.** One forward over 512 tokens took 62 ms
  against nnsight's 53 ms. A sweep of 1,024 prompts took 25.9 s on its compiled path, which
  takes one prompt at a time, and 7.2 s on its batched path, against nnsight's 1.6 s.

### Other models and settings

??? note "Llama-3.1-8B, 2 GPUs"

    --8<-- "docs/vllm/assets/throughput-llama-8b-tp2.svg"

    --8<-- "docs/vllm/assets/throughput-llama-8b-tp2.md"

??? note "Llama-3.1-8B, 4 GPUs"

    --8<-- "docs/vllm/assets/throughput-llama-8b-tp4.svg"

    --8<-- "docs/vllm/assets/throughput-llama-8b-tp4.md"

??? note "Llama-3.1-8B, 8 GPUs"

    --8<-- "docs/vllm/assets/throughput-llama-8b-tp8.svg"

    --8<-- "docs/vllm/assets/throughput-llama-8b-tp8.md"

??? note "Llama-3.1-70B, 4 GPUs"

    --8<-- "docs/vllm/assets/throughput-llama-70b.svg"

    --8<-- "docs/vllm/assets/throughput-llama-70b.md"

??? note "Llama-3.1-70B, 8 GPUs"

    --8<-- "docs/vllm/assets/throughput-llama-70b-tp8.svg"

    --8<-- "docs/vllm/assets/throughput-llama-70b-tp8.md"

??? note "Llama-3.1-8B, 2,048-token prompt and 512 new tokens"

    --8<-- "docs/vllm/assets/throughput-llama-8b-long.svg"

    --8<-- "docs/vllm/assets/throughput-llama-8b-long.md"

??? note "Llama-3.1-8B, 32 concurrent requests"

    --8<-- "docs/vllm/assets/throughput-llama-8b-x32.svg"

    --8<-- "docs/vllm/assets/throughput-llama-8b-x32.md"

??? note "Llama-3.2-1B, one GPU"

    --8<-- "docs/vllm/assets/throughput-llama-1b.svg"

    --8<-- "docs/vllm/assets/throughput-llama-1b.md"

??? note "Qwen3-8B, one GPU"

    --8<-- "docs/vllm/assets/throughput-qwen3-8b.svg"

    --8<-- "docs/vllm/assets/throughput-qwen3-8b.md"

??? note "Qwen1.5-MoE-A2.7B (mixture of experts), one GPU"

    --8<-- "docs/vllm/assets/throughput-qwen-moe.svg"

    At 8 concurrent requests the engines that keep CUDA graphs exceed plain vLLM. On this
    model vLLM's compiled path is slower than its graph path without compilation (669 against
    1,095 tokens per second), and plain vLLM uses the compiled path.

    --8<-- "docs/vllm/assets/throughput-qwen-moe.md"

??? note "DeepSeek-V2-Lite (mixture of experts), one GPU"

    --8<-- "docs/vllm/assets/throughput-deepseek-v2-lite.svg"

    --8<-- "docs/vllm/assets/throughput-deepseek-v2-lite.md"

??? note "Qwen3.5-0.8B, one GPU"

    --8<-- "docs/vllm/assets/throughput-qwen35-08b.svg"

    Qwen3.5 and Qwen3.6 mix recurrent layers with attention layers. On vLLM 0.27.1 a tapped
    engine replayed CUDA graphs on these models for generation steps only.

    --8<-- "docs/vllm/assets/throughput-qwen35-08b.md"

??? note "Qwen3.5-4B, one GPU"

    --8<-- "docs/vllm/assets/throughput-qwen35-4b.svg"

    --8<-- "docs/vllm/assets/throughput-qwen35-4b.md"

??? note "Qwen3.6-35B-A3B (mixture of experts), 2 GPUs"

    --8<-- "docs/vllm/assets/throughput-qwen36-moe.svg"

    --8<-- "docs/vllm/assets/throughput-qwen36-moe.md"
