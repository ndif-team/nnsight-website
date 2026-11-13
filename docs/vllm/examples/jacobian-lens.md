# Jacobian lens

The [logit lens](logit-lens.md) reads a layer's residual stream as if the later layers did
nothing. The Jacobian lens (Anthropic, *Verbalizable Representations Form a Global Workspace*,
2026) first maps the residual stream into the final layer's basis with a fitted matrix
`J_l`, the average Jacobian of the final hidden state with respect to layer `l`, and then
unembeds.

Fitting `J_l` needs gradients, which vLLM does not have. Neuronpedia publishes fitted lenses,
including one for `Qwen/Qwen3-8B`. Reading with a lens is a matrix product, so it runs in the
worker.

## Load the lens

Each `J_l` is a `[4096, 4096]` matrix. Whatever a block uses is sent to the worker with it, so
load only the layers you read.

```python
import torch
import nnsight
from huggingface_hub import hf_hub_download
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen3-8B", dispatch=True)
tok = model.tokenizer

path = hf_hub_download("neuronpedia/jacobian-lens",
                       "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt")
lens = torch.load(path, map_location="cpu")
print(lens["source_layers"][:3], "...", lens["source_layers"][-1], "fit on", lens["n_prompts"], "prompts")
# [0, 1, 2] ... 34 fit on 461 prompts

LAYERS = [20, 26, 32]
J = {l: lens["J"][l].to("cuda", torch.bfloat16) for l in LAYERS}    # [4096, 4096] each
```

## Read it out

`h @ J_l.T` maps the residual stream into the final basis. The model's own `norm` and
`logits_processor` do the rest, as in the logit lens.

```python
prompt = "The Eiffel Tower is located in the city of"
K = 5

with model.trace(prompt, temperature=0.0):
    jlens, plain = {}, {}
    for l in LAYERS:
        h = sum(model.model.layers[l].output)                           # [T, d_model]
        for name, x in (("jlens", h @ J[l].T), ("plain", h)):
            logits = model.logits_processor(model.lm_head, model.model.norm(x))
            top = logits.float().softmax(-1).topk(K, dim=-1)
            (jlens if name == "jlens" else plain)[l] = (top.indices, top.values)
    jlens, plain = nnsight.save(jlens), nnsight.save(plain)

ids = tok(prompt)["input_ids"]
for l in LAYERS:
    print(f"\nlayer {l}   (top-1 at every position)")
    for name, res in (("J-lens", jlens), ("logit lens", plain)):
        print(f"  {name:>10}: " + " ".join(f"{tok.decode(t)!r:>10}" for t in res[l][0][:, 0].tolist()))
# layer 20   (top-1 at every position)
#       J-lens: '/Internal'   ' Space'      '建筑师'       '巴黎'  ' French'     '____'     '____'     '____'    ' city'    ' city'     '____'
#   logit lens:        '玿'        '‐'      'ian'   ' Tower' ' famously' ' famously'    ' ____'  ' ______'  ' center'   ' _____'   ' _____'
# layer 26   (top-1 at every position)
#       J-lens:    'gMaps'    'lixir'    'stadt'   ' Tower'      '是一座' ' tallest'     '____'     '____'       '法国'    '(city'     '____'
#   logit lens:        '玿'    'ureka'   'icient'   ' Tower' ' famously'  ' taller'    ' ____'    ' ____'  ' famous'      ' of'  ' ______'
# layer 32   (top-1 at every position)
#       J-lens:      ' ",'    'ulers'       'el'   ' Tower'      ' is' ' located'     '____'   ' Paris'    ' city'      ' of'   ' Paris'
#   logit lens:        '玿'    'ureka'       'el'   ' Tower'      ' is'    ' made'      ' in'   ' Paris'    ' city'      ' of'   ' Paris'
```

At layer 20 the Jacobian lens already reads `巴黎` (Paris) at the subject and `French` after
it, where the logit lens reads `Tower` and `famously`. By layer 32 both read `Paris`.

## Every step of a generation

Under `tracer.iter` the lens follows each generated token.

```python
with model.trace(prompt, temperature=0.0, max_tokens=8) as tracer:
    disposed = list().save()
    for _ in tracer.iter[:8]:
        h = sum(model.model.layers[26].output)[-1:]
        logits = model.logits_processor(model.lm_head, model.model.norm(h @ J[26].T))
        disposed.append(logits.argmax(-1).item())
    out = tracer.result.save()

print(repr(out.outputs[0].text))
# ' Paris, France. It is one of'
print([tok.decode(t) for t in disposed])
# ['____', ' France', ' France', '.', ' Its', '是一座', ' famous', ' famous']
```

## Install it for every request

Put the same block in [`model.edit()`](../editing.md) and every request returns its own
readout. The lens is sent once, with the edit.

```python
with model.edit() as (tracer, edit):
    readout = nnsight.save([])
    for _ in tracer.all():
        h = sum(model.model.layers[26].output)[-1:]
        logits = model.logits_processor(model.lm_head, model.model.norm(h @ J[26].T))
        readout.append(logits.float().softmax(-1).topk(3, dim=-1).indices[0].tolist())

outputs = model.generate(["The capital of Japan is", "Water boils at"], temperature=0.0, max_tokens=4)
for o in outputs:
    print(repr(o.outputs[0].text), [[tok.decode(t) for t in step] for step in o.saves["readout"][:2]])
# ' Tokyo. The capital' [['____', ' ____', ' ______'], ['.', '。', '.",']]
# ' 100' [['温度', ' temperatures', 'Temperature'], ['温度', ' boiling', '高温']]
edit.clear()
```
