# Writing a block

A block on vLLM is written like any nnsight trace. This page covers what is specific to vLLM:
the shape of a value, the values the engine adds, generation steps, and several prompts.

## Locations

A location is a module on vLLM's model tree plus a side: `.output`, `.input`, `.inputs` (all
arguments), or an operation inside its forward through `.source`. Print the model to see the
tree. This needs no GPU.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B")           # the module tree only
print(model.model.layers[10].mlp)
# Qwen2MLP(
#   (gate_up_proj): MergedColumnParallelLinear(in_features=4096, output_features=24576, bias=False, tp_size=1, gather_output=False)
#   (down_proj): RowParallelLinear(in_features=12288, output_features=4096, bias=False, tp_size=1, reduce_results=True)
#   (act_fn): SiluAndMul()
# )
```

The engine adds three values that are not modules:

| Value | What it is | Shape |
| --- | --- | --- |
| `model.logits` | The logits the sampler is about to draw from, this step. Writable. | `[1, vocab]` |
| `model.samples` | The token the sampler drew, this step. Writable. | `[1, 1]` |
| `tracer.result` | The finished request, vLLM's `RequestOutput`. Read it last. | |

## Shapes

vLLM packs the tokens of every running request into one tensor, and nnsight gives your block
its own request's rows. A value is `[tokens, width]`: every prompt token on the first step,
then one row per generated token. The newest token is always `[-1]`.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0, max_tokens=3) as tracer:
    shapes = list().save()
    for _ in tracer.iter[:3]:
        shapes.append((tuple(model.model.layers[10].output[0].shape), tuple(model.logits.shape)))

print(shapes)
# [((5, 4096), (1, 151936)), ((1, 4096), (1, 151936)), ((1, 4096), (1, 151936))]
```

## The residual stream

vLLM fuses the residual add into the next layer's norm. A decoder layer therefore returns a
pair, `(hidden, residual)`, and the residual stream after the layer is their sum.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)

with model.trace("The capital of France is", temperature=0.0):
    hidden, residual = model.model.layers[10].output
    norms = (hidden.norm(dim=-1)[-1].item(), residual.norm(dim=-1)[-1].item()).save()
    stream = (hidden + residual).save()                 # the residual stream after layer 10

print([round(n) for n in norms], stream.shape)
# [31, 61] torch.Size([5, 4096])
```

Linear layers return `(tensor, bias)`, so their value is `.output[0]`. A decoder layer's first
positional argument is the position ids, so read its hidden state from `.inputs`, not `.input`.

## Clone what you keep

A value is a view of the engine's memory, and a later layer overwrites it. Clone, or compute
something new from it, before you save.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
layer = model.model.layers[10]

with model.trace("The capital of France is", temperature=0.0):
    view = layer.output[1].save()                # a view of vLLM's buffer
    copy = layer.output[1].clone().save()        # a copy, taken before layer 11 runs

print((view - copy).abs().max().item())          # not zero: the view was overwritten
# 3840.0
```

Setting `NNSIGHT_VLLM_CLONE_READS=1` in the environment before the engine is built makes every
read a copy. In-place edits then write to the copy and are lost, so assign the edited value
back (`layer.output = out`).

## Writing

Edit in place: `x[:] += v`, `x[-1] = 0`. The next module reads the edited memory. A replacement
(`layer.output = t`) must have the same rows as the value it replaces. Nothing checks this, and
a wrong size can stop the engine ([Limits and errors](limits.md#nothing-checks-these)).

A write in the block body runs on the first step only. To keep it on while the model
generates, put it under `tracer.iter`.

## Generation steps

`max_tokens` sets the number of steps. `tracer.iter[:n]` runs its body once per step, and
`tracer.all()` runs it on every step.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
berlin = model.tokenizer.encode(" Berlin")[0]

with model.trace("The capital of France is", temperature=0.0, max_tokens=6) as tracer:
    tokens = list().save()
    for step in tracer.iter[:6]:
        if step == 0:
            model.samples[:] = berlin                  # force the first token
        tokens.append(model.samples.item())
    out = tracer.result.save()

print([model.tokenizer.decode(t) for t in tokens], repr(out.outputs[0].text))
# [' Berlin', '.', ' Is', ' this', ' statement', ' true'] ' Berlin. Is this statement true'
```

- Sampling arguments go on `trace` or `invoke`: `temperature`, `top_p`, `max_tokens`, `seed`,
  `stop`, `logprobs` and anything else vLLM's `SamplingParams` takes. vLLM's default temperature
  is `1.0`, so pass `temperature=0.0` for greedy decoding.
- A request can end before `max_tokens`, on an end-of-sequence token or a `stop` string. A
  loop that asks for more steps is cut short, and the statements after it do not run. The
  warning is printed by the engine process, so check the length of what you collected, or pass
  `ignore_eos=True`.
- `tracer.result` is served when the request finishes, so read it last.
- `model.generate(prompts, ...)` called without a `with` block runs the engine and returns its
  outputs.

## Several prompts

Each invoke is one request, and vLLM batches them. A name saved in every invoke comes back as
a list, one entry per invoke.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
prompts = ["Paris is in", "The capital of Japan is", "Berlin is the capital of"]

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for prompt in prompts:
        with tracer.invoke(prompt):
            stream = sum(model.model.layers[20].output).save()
            token = model.logits.argmax(-1).save()

for prompt, s, t in zip(prompts, stream, token):
    print(f"{prompt!r:28} {tuple(s.shape)}  -> {model.tokenizer.decode(t)!r}")
# 'Paris is in'                (3, 4096)  -> ' the'
# 'The capital of Japan is'    (5, 4096)  -> ' Tokyo'
# 'Berlin is the capital of'   (5, 4096)  -> ' Germany'
```

Invokes do not share a forward pass, so one invoke cannot read another's values and
`tracer.barrier` is refused. To move a value between prompts, save it in one trace and use it
in the next ([Activation patching](examples/patching.md)).

An invoke takes one prompt: a string, a list of token ids, or a tokenizer's output. For a chat
model, apply the template yourself:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
messages = [{"role": "user", "content": "In one word, what is the capital of France?"}]
ids = model.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False)      # a Qwen3 option

with model.trace(ids, temperature=0.0, max_tokens=8) as tracer:
    out = tracer.result.save()

print(repr(out.outputs[0].text))
# 'Paris.'
```

## Qwen3-8B by name

`L` is `model.model.layers[i]`, and `T` is the number of tokens. Llama, Mistral and Gemma have
the same layout.

| What | Location | Shape |
| --- | --- | --- |
| Token embeddings | `model.model.embed_tokens.output` | `[T, 4096]` |
| Residual stream before layer *i* | `L.input_layernorm.output[1]` | `[T, 4096]` |
| Residual stream after layer *i* | `L.output[0] + L.output[1]` | `[T, 4096]` |
| Query, key and value, fused | `L.self_attn.qkv_proj.output[0]` | `[T, 6144]` |
| Query, key and value, as the attention kernel gets them | `L.self_attn.attn.inputs[0]` | `[T, 4096]`, `[T, 1024]`, `[T, 1024]` |
| Each head's output, before the output projection | `L.self_attn.o_proj.input` | `[T, 4096]`, head *h* at `h*128:(h+1)*128` |
| Attention output | `L.self_attn.output` | `[T, 4096]` |
| MLP neurons, after the activation | `L.mlp.act_fn.output` | `[T, 12288]` |
| MLP output | `L.mlp.output` | `[T, 4096]` |
| Final norm | `model.model.norm.output[0]` | `[T, 4096]` |

Three things are not locations on any vLLM model:

- **The attention pattern.** The attention kernel never forms it. Rebuild it from the queries
  and keys ([Attention patterns](examples/attention.md)).
- **One expert of a mixture-of-experts layer.** The experts are one fused kernel. The router is
  `mlp.gate`, and masking a router logit removes an expert.
- **The inside of a fused kernel.** Its inputs and outputs are locations.

vLLM builds one rotary-embedding module and shares it across layers, so read it through the
layer that uses it (`self_attn.attn.inputs`).
