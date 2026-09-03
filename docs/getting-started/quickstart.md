# Quick Start

Your first nnsight intervention, end to end. For the longer version, see the
[Walkthrough](../tutorials/tutorials/get_started/walkthrough.ipynb).

## Loading a Model

nnsight wraps PyTorch models to enable tracing and intervention. For language models, use `TransformersModel`:

```python
from nnsight import TransformersModel

model = TransformersModel('gpt2', device_map='auto', dispatch=True)
```

!!! info "Model Dispatching"
    Setting `dispatch=True` loads the model weights immediately. Otherwise, the model is loaded on a [meta device](https://docs.pytorch.org/docs/stable/meta.html) for faster initialization.

## Your First Trace

The `.trace()` context manager runs a forward pass while giving you access to internal activations:

```python
with model.trace('The Eiffel Tower is in the city of'):
    # Access hidden states from the last layer
    hidden_states = model.transformer.h[-1].output.save()
    
    # Get the model's output
    output = model.output.save()

# After exiting the context, saved values are available
print(hidden_states.shape)  # torch.Size([1, 10, 768])
print(model.tokenizer.decode(output.logits.argmax(dim=-1)[0]))
```

!!! warning "Always use `.save()`"
    The body of a trace does not run where you wrote it. nnsight compiles it and runs it in a
    worker alongside the forward pass, then copies back only the values you marked. So a value
    you want after the block **must** be `.save()`d, and it comes back under the name you bound
    it to. Reading an unsaved name afterwards is a `NameError` in a script, an
    `UnboundLocalError` inside a function.

## Accessing Activations

Access any module's input or output during the forward pass. A module's `.output` **is the object that module really returns** — sometimes a tensor, sometimes a tuple — so it varies per module, not per model.

!!! warning "Tuple or tensor? Check, don't assume"
    In [:hugging_face: `transformers`](https://github.com/huggingface/transformers) **5.x**, a decoder block (`GPT2Block`, `LlamaDecoderLayer`) returns a **bare tensor**, while an attention submodule still returns a **tuple**. In **4.x** blocks returned tuples too, which is why a great deal of nnsight code you will find online indexes them with `[0]`.

    That matters because indexing a tensor with `[0]` **does not raise** — it silently gives you the first element of the *batch*, shape `[seq, hidden]` instead of `[batch, seq, hidden]`, and everything downstream is quietly wrong.

    One line settles it for any module:

    ```python
    with model.trace("hello"):
        print(type(model.transformer.h[0].attn.output))  # <class 'tuple'>
        print(type(model.transformer.h[0].output))       # <class 'torch.Tensor'>
    ```

    (Attention first: it runs inside the block, so it produces its output before the block
    produces its own.)

```python
with model.trace("The Eiffel Tower is in the city of"):
    attn_output = model.transformer.h[0].attn.output[0].save() # (1)!
    
    mlp_output = model.transformer.h[0].mlp.output.save() # (2)!

    # Access the full layer output (a tensor on transformers 5.x -- no [0])
    layer_output = model.transformer.h[5].output.save()
    
    # Access the final logits
    logits = model.lm_head.output.save()
```

1. The output of the attention module is a tuple
2. The MLP output is a single tensor, so we can save it directly without indexing

## Modifying Activations

Intervene on the model by modifying activations in-place:

```python
with model.trace("Hello"):
    # Zero out all activations at layer 0
    model.transformer.h[0].output[:] = 0
    
    # Modify only the last token position
    model.transformer.h[1].output[:, -1, :] = 0
    
    output = model.output.save()
```

Or replace activations entirely:

```python
import torch

with model.trace("Hello"):
    # Add noise to MLP output
    hs = model.transformer.h[-1].mlp.output.clone()
    noise = 0.01 * torch.randn(hs.shape, device=hs.device, dtype=hs.dtype)
    model.transformer.h[-1].mlp.output = hs + noise

    result = model.transformer.h[-1].mlp.output.save()
```

Every tensor you build inside a trace has to land where the activation already is. `torch.randn`
gives you a CPU tensor, and adding one to a CUDA activation is a
`RuntimeError: Expected all tensors to be on the same device`. Reading `device=` and `dtype=`
off the activation itself, as above, also stays correct when `device_map` has sharded the model
across devices or the layers are in mixed precision.

The two forms differ in more than style. `output[:] = v` writes through the tensor the model is
holding; `output = v` hands the model a different one. Both take effect, but a replacement built
from scratch, such as `torch.zeros_like(...)` or a fresh `torch.randn(...)`, is a tensor autograd
has never seen, so it cuts the graph and any later gradient read fails. Derive the new value from
the old one, as `hs + noise` does, or write in place.

## Understanding Module Hierarchy

Print the model to see its structure and available modules:

```python
print(model)
```

```
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm(...)
        (attn): GPT2Attention(...)
        (ln_2): LayerNorm(...)
        (mlp): GPT2MLP(...)
      )
    )
    (ln_f): LayerNorm(...)
  )
  (lm_head): Linear(...)
)
```

Access any module using the same dotted path notation:

- `model.transformer.h[0]` — First transformer block
- `model.transformer.h[0].attn` — Attention module in first block
- `model.transformer.h[-1].mlp` — MLP in last block
- `model.lm_head` — Final language modeling head

## The three properties

Everything you read or write on a module goes through one of these:

| Property | Is | Assignable |
|----------|----|----|
| `.output` | the module's forward-pass return value | yes |
| `.input` | its first positional argument (or first keyword one) | yes |
| `.inputs` | `(args, kwargs)` — everything it was called with | yes |

Within one trace you have to touch modules in the order the model runs them. Your code is a
worker that parks until the model produces each value, so reading layer 11 and then writing
layer 0 raises `OutOfOrderError`: layer 0 has already gone by. Put reads and writes in forward
order, or give each one its own `tracer.invoke(...)`.

## Using with Any PyTorch Model

For arbitrary PyTorch models (not just language models), use the base `NNsight` wrapper:

```python
from nnsight import NNsight
import torch

net = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.Linear(10, 2)
)

model = NNsight(net)

with model.trace(torch.rand(1, 5)):
    layer1_out = model[0].output.save()
    output = model.output.save()

print(layer1_out.shape)  # torch.Size([1, 10])
```

## Next Steps

- **[Walkthrough](../tutorials/tutorials/get_started/walkthrough.ipynb)** — the full guided introduction
- **[Features](../features/index.md)** — one page per capability
- **[Tutorials](../tutorials/index.md)** — worked interpretability experiments
- **[Documentation](../documentation/index.md)** — the reference
