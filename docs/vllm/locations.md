# Locations

A **location** is a module on vLLM's model tree plus a side: `.input`, `.output`, `.inputs` (all
positional and keyword arguments), or one of its `.source` ops. There is no fixed vocabulary of
named points; `print(model)` is the vocabulary, and any module on it is a location.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B")        # meta tree only; no GPU needed to look
print(model.model.layers[10])
```

Two more locations sit after the model and belong to the engine rather than a module:
`model.logits` (this step's pre-sampling logits) and `model.samples` (the ids the sampler drew).
`tracer.result` is the finished request. All three are on [Generation](generation.md).

## The shape of a value

vLLM concatenates every in-flight request's tokens into one `[total_tokens, hidden]` slab; nnsight
narrows what your block sees to its own request's rows. So a value is `[pos, width]` on the prefill
step and `[1, width]` on each decode step — there is no batch axis, and `[-1]` is always the newest
token.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0, max_tokens=3) as tracer:
    shapes = list().save()
    for _ in tracer.all():
        shapes.append((tuple(model.model.layers[10].output[0].shape), tuple(model.logits.shape)))

print(shapes)
# [((5, 4096), (1, 151936)), ((1, 4096), (1, 151936)), ((1, 4096), (1, 151936))]
```

## Tuples, and the residual stream

vLLM fuses the residual add into the norm kernel, so a decoder layer returns two tensors rather
than one, and so do the two norms inside it. The residual stream is the **sum** of a layer's
outputs; the second element alone is the stream *before* the MLP was added.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
layer = model.model.layers[10]

with model.trace("The capital of France is", temperature=0.0):
    normed, resid_pre = layer.input_layernorm.output                # entering block 10
    resid_pre = resid_pre.clone().save()
    attn_out = layer.self_attn.output.clone().save()
    mlp_out, resid_mid = layer.output                                # leaving block 10
    resid_mid = resid_mid.clone().save()
    resid_post = (mlp_out + resid_mid).save()

print((resid_pre + attn_out - resid_mid).abs().max().item())        # resid_mid = resid_pre + attn
# 0.0
```

Checked against a HuggingFace forward on the same prompt: `resid_post` at every layer, `resid_pre`,
`resid_mid`, the final norm and the logits all agree to bf16 noise (cosine ≥ 0.9999).

## Clone what you keep

The residual tensor is rewritten **in place** by the next block's fused norm, and vLLM recycles
activation buffers between layers. A saved reference therefore points at memory a later layer
overwrites; a fresh tensor (a sum, a slice copy, a `.clone()`) does not.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
layer = model.model.layers[10]

with model.trace("The capital of France is", temperature=0.0):
    aliased = layer.output[1].save()             # a reference into vLLM's buffer
    cloned = layer.output[1].clone().save()      # a copy, taken before layer 11 runs

print((aliased - cloned).abs().max().item())    # not zero: `aliased` was overwritten
# 3840.0
```

This holds on the eager engine and is *guaranteed* under [graph taps](performance.md), where the
value served is the graph's own memory. `tracer.cache()` clones for you.

Buffer recycling is not the only way a saved reference goes stale: some layers keep writing into
the tensor they returned. DeepSeek's MLA attention rotates the `q_proj` output in place after the
module returns, so `q_proj.output[0].save()` on one rank comes back rotated while the same read
under tensor parallelism (a fresh gather) does not — `.clone()` makes both the linear's output.

## Qwen3-8B, by name

Every location below was read on `Qwen/Qwen3-8B` (36 layers, `d_model` 4096, 32 query heads, 8
KV heads, `head_dim` 128, `d_mlp` 12288) for a prompt of `T` tokens. `L = model.model.layers[i]`.

| What | Location | Shape |
| --- | --- | --- |
| token embeddings | `model.model.embed_tokens.output` | `[T, 4096]` |
| residual entering block *i* | `L.input_layernorm.output[1]` (block 0: `L.input_layernorm.input`) | `[T, 4096]` |
| attention input (normed) | `L.input_layernorm.output[0]` = `L.self_attn.inputs[1]["hidden_states"]` | `[T, 4096]` |
| fused q ‖ k ‖ v | `L.self_attn.qkv_proj.output[0]` | `[T, 6144]` = 4096 ‖ 1024 ‖ 1024 |
| q, k after QK-norm | `L.self_attn.q_norm.output`, `.k_norm.output` | `[T, 32, 128]`, `[T, 8, 128]` |
| q, k, v after RoPE | `L.self_attn.attn.inputs[0]` | `[T, 4096]`, `[T, 1024]`, `[T, 1024]` |
| per-head attention output, before `W_O` | `L.self_attn.attn.output` = `L.self_attn.o_proj.input` | `[T, 4096]` (head *h* at `h*128:(h+1)*128`) |
| attention output | `L.self_attn.output` = `L.self_attn.o_proj.output[0]` | `[T, 4096]` |
| residual after attention | `L.post_attention_layernorm.output[1]` | `[T, 4096]` |
| MLP input (normed) | `L.post_attention_layernorm.output[0]` = `L.mlp.input` | `[T, 4096]` |
| gate ‖ up, pre-activation | `L.mlp.gate_up_proj.output[0]` | `[T, 24576]` = 12288 ‖ 12288 |
| post-activation neurons | `L.mlp.act_fn.output` | `[T, 12288]` |
| MLP output | `L.mlp.output` = `L.mlp.down_proj.output[0]` | `[T, 4096]` |
| residual leaving block *i* | `L.output[0] + L.output[1]` | `[T, 4096]` |
| final norm | `model.model.norm.output[0]` (`[1]` is the un-normed residual) | `[T, 4096]` |
| logits, this step | `model.logits` | `[1, 151936]` |
| sampled id, this step | `model.samples` | `[1, 1]` |

Positional `.input` on a block is its **first** positional argument, which for a vLLM decoder layer
is the position ids, not the hidden states — hence `.inputs[1]["hidden_states"]` above, or the norm's
output. Linear layers return `(tensor, bias)`, so their `.output[0]` is the tensor. The attention
pattern is not a location on any vLLM model (the paged kernel never forms it); [Attention](attention.md)
rebuilds it from q and k.

## Inside a forward

`.source` lists the operations of a module's `forward`, each of them a location too.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
print(model.model.layers[10].self_attn.source)

with model.trace("The capital of France is", temperature=0.0):
    q = model.model.layers[10].self_attn.source.qkv_split_0.output[0].clone().save()

print(q.shape)
# torch.Size([5, 4096])
```

## Shared modules

vLLM builds one `RotaryEmbedding` and hands it to every layer, so `layers[10].self_attn.rotary_emb`
is the same module — and the same location — as `layers[0].self_attn.rotary_emb`. Reading it "at
layer 10" is out of order by the time layer 10 runs. Read the per-layer module that consumes its
result instead (`self_attn.attn.inputs`), or the source op `self_attn.source.self_rotary_emb_0`.

## Other architectures

The table is Qwen3's; Llama, Mistral, Gemma and friends have the same `model.layers[i].self_attn` /
`.mlp` layout with the same fused projections, GPT-2 has `transformer.h[i].attn` / `.mlp`, and a
mixture-of-experts block adds `mlp.gate` (the router, a plain linear) and `mlp.experts` (one fused
kernel — individual experts are not addressable). DeepSeek's multi-head latent attention has no
`qkv_proj`: `self_attn.q_proj` (or `q_a_proj` → `q_b_proj` when the checkpoint sets `q_lora_rank`,
fused with the KV down-projection as `fused_qkv_a_proj`), the replicated `kv_a_proj_with_mqa`,
`kv_b_proj` (column-parallel, `[T, heads * (qk_nope_head_dim + v_head_dim)]`) and `o_proj`; its
MoE blocks add `mlp.shared_experts` beside `mlp.experts`. `print(model)` on the meta tree costs
nothing and settles it.
