# Quick Start

This guide walks you through your first nnsight intervention in just a few minutes.

## Loading a Model

nnsight wraps PyTorch models to enable tracing and intervention. For language models, use `LanguageModel`:

```python
from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='auto', dispatch=True)
```

!!! info "Model Dispatching"
    Setting `dispatch=True` loads the model weights immediately. Otherwise, the model is loaded on a [meta device](https://docs.pytorch.org/docs/stable/meta.html) for faster initialization.

## Your First Trace

The `.trace()` context manager runs a forward pass while giving you access to internal activations:

```python
with model.trace('The Eiffel Tower is in the city of'):
    # Access hidden states from the last layer
    hidden_states = model.transformer.h[-1].output[0].save()
    
    # Get the model's output
    output = model.output.save()

# After exiting the context, saved values are available
print(hidden_states.shape)  # torch.Size([1, 10, 768])
print(model.tokenizer.decode(output.logits.argmax(dim=-1)[0]))
```

!!! warning "Always use `.save()`"
    Values you want to access after the trace exits **must** be saved with `.save()`. Without it, tensors are garbage collected at the end of the trace context.

## Accessing Activations

Access any module's input or output during the forward pass. Check your model's architecture to understand its output structure. For example, layers in [:hugging_face: `transformers`](https://github.com/huggingface/transformers) models typically return tuples, where the first element contains the hidden states.

```python
with model.trace("The Eiffel Tower is in the city of"):
    attn_output = model.transformer.h[0].attn.output[0].save() # (1)!
    
    mlp_output = model.transformer.h[0].mlp.output.save() # (2)!

    # Access the full layer output
    layer_output = model.transformer.h[5].output[0].save()
    
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
    model.transformer.h[0].output[0][:] = 0
    
    # Modify only the last token position
    model.transformer.h[1].output[0][:, -1, :] = 0
    
    output = model.output.save()
```

Or replace activations entirely:

```python
import torch

with model.trace("Hello"):
    # Add noise to MLP output
    hs = model.transformer.h[-1].mlp.output.clone()
    noise = 0.01 * torch.randn(hs.shape)
    model.transformer.h[-1].mlp.output = hs + noise
    
    result = model.transformer.h[-1].mlp.output.save()
```

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

## Key Properties

Every module has these special properties for accessing values:

| Property | Description |
|----------|-------------|
| `.output` | The module's forward pass output |
| `.input` | First positional argument to the module |
| `.inputs` | All inputs as `(args_tuple, kwargs_dict)` |

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

You've learned the basics of nnsight! Continue exploring:

- **[Features](../features/index.md)** — Deep dives into specific capabilities
- **[Tutorials](../tutorials/index.md)** — Step-by-step guides for common tasks
- **[Documentation](../documentation/index.md)** — Comprehensive reference material
