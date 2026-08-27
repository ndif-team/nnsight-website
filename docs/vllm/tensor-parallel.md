# Tensor parallelism

`tensor_parallel_size=N` shards the model across `N` GPUs. Your block does not change: every
value it reads is the **whole** tensor, and what it writes is re-sharded before vLLM continues.

## The same block, sharded

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True, tensor_parallel_size=2)
attn = model.model.layers[10].self_attn

with model.trace("The capital of France is", temperature=0.0):
    qkv = attn.qkv_proj.output[0].clone().save()        # column-parallel: gathered, [pos, 6144]
    z = attn.o_proj.input.clone().save()                # row-parallel input: gathered, [pos, 4096]
    resid = sum(model.model.layers[10].output).save()   # already whole: [pos, 4096]
    logits = model.logits.save()

print(qkv.shape, z.shape, resid.shape, model.tokenizer.decode(logits.argmax(-1)))
# torch.Size([5, 6144]) torch.Size([5, 4096]) torch.Size([5, 4096])  Paris
```

On one rank `qkv_proj` produces half the columns and `o_proj` consumes half; nnsight all-gathers
(or all-reduces, for a deferred partial sum) at the location your block is waiting on, once per
visit, and only there. Untouched layers pay nothing.

## Writes land on every rank

Zero one head, whichever rank holds it:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True, tensor_parallel_size=2)
head, d = 7, 128

with model.trace("The capital of France is", temperature=0.0, max_tokens=8) as tracer:
    for _ in tracer.iter[:8]:
        model.model.layers[10].self_attn.o_proj.input[:, head * d:(head + 1) * d] = 0
    out = tracer.result.save()

print(repr(out.outputs[0].text))
# ' Paris. The capital of France is Paris'
```

The gathered tensor is edited on every rank identically, then each rank takes its own shard
back.

## Rank order in fused tensors

A fused column-parallel layer (`qkv_proj`, `gate_up_proj`) gathers in rank order, so the whole
holds every value of the single-GPU tensor but grouped by rank: at `tp=2`,
`[q₀ k₀ v₀ | q₁ k₁ v₁]` rather than `[q | k | v]`. Slice by head — head `h`'s query is
`rank = h // (32 // tp)`, and inside that rank's block the same offsets as on one GPU — rather
than `[:q_size]`.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True, tensor_parallel_size=2)
attn = model.model.layers[10].self_attn
tp, n_heads, n_kv, d = 2, 32, 8, 128
per_rank = (n_heads + 2 * n_kv) // tp * d          # 3072 columns per rank: 16 q, 4 k, 4 v heads


def q_of_head(qkv, h):
    rank, local = divmod(h, n_heads // tp)
    start = rank * per_rank + local * d
    return qkv[:, start:start + d]


with model.trace("The capital of France is", temperature=0.0):
    qkv = attn.qkv_proj.output[0]
    q7 = q_of_head(qkv, 7).clone().save()
    q20 = q_of_head(qkv, 20).clone().save()          # lives on rank 1

print(q7.shape, q20.shape)
# torch.Size([5, 128]) torch.Size([5, 128])
```

`o_proj.input` (`z`) is gathered the same way, and since each rank holds a contiguous run of heads
the head slice `h*d:(h+1)*d` is the same as on one GPU.

## Parameters are shards

Only activations are gathered. `attn.o_proj.weight` inside a trace is this rank's piece.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True, tensor_parallel_size=2)

with model.trace("The capital of France is", temperature=0.0):
    w = model.model.layers[10].self_attn.o_proj.weight.shape
    lm = model.lm_head.weight.shape
    shapes = (tuple(w), tuple(lm)).save()

print(shapes)
# ((4096, 2048), (75968, 4096))
```

For a logit lens under TP use the model's own logits processor, which gathers the vocabulary —
`model.logits_processor(model.lm_head, h)` — rather than `h @ lm_head.weight.T`
([Logit lens](logit-lens.md)).

## Rules

- **Every rank runs the block.** No rank-dependent control flow, no reading `torch.distributed`
  rank inside it; the collectives would deadlock.
- **Sampling must agree across ranks** — greedy, or seeded — since every rank runs the block
  against its own sampled ids.
- Rank 0 reports the saves.
- vLLM spawns its workers once CUDA is initialised in the parent; set
  `VLLM_WORKER_MULTIPROC_METHOD=spawn` only if it complains about forking after CUDA
  initialisation. Not nnsight's requirement.

## Mixture of experts

The router (`mlp.gate`) is replicated, so its logits need no gathering on any rank. The fused
experts module is the one MoE-specific case: where vLLM leaves it a per-rank partial sum, nnsight
all-reduces on read and divides a write back by the group size so the block's own reduce sums it
exactly once. Both expert layouts work — the default (every rank holds a slice of every expert)
and `enable_expert_parallel=True` (each rank holds whole experts). Verified against a HuggingFace
reference in `tests/vllm/test_moe_batching.py`.

## Decode-context parallelism

`decode_context_parallel_size > 1` on an MLA model (DeepSeek) with `VLLM_DCP_Q_REPLICATE=1`
builds `q_proj` as a group-sharded layer; nnsight drops the group replicas after the gather so the
value reads as whole. Verified on DeepSeek-V2-Lite at `tp=4, dcp=2`.

## Multi-node with Ray

vLLM's stock Ray executor is a plain engine kwarg; nnsight adds nothing to it — the block rides the
request into whichever node's worker runs it.

<!-- norun -->
```python
from nnsight.modeling.vllm import VLLM

model = VLLM("meta-llama/Llama-3.1-70B", tensor_parallel_size=8,
             distributed_executor_backend="ray", dispatch=True)
```

Cluster setup (`ray start`, `RAY_ADDRESS`, placement) is as vLLM documents for
[multi-node serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html). Ray's
**v1** executor is not supported (`tracer.result` and dangling-read errors are dropped there);
the default v2 executor is.

## Not covered

Pipeline parallelism (`pipeline_parallel_size > 1`): the first stage never sees the logits, and
a block spanning stages is not shipped between them. Shard with tensor parallelism instead.
