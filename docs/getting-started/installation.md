# Installation

## Requirements

- Python 3.10 or higher
- PyTorch 2.4 or higher
- `transformers` 5.x, if you are working with HuggingFace models

`pip install nnsight` pulls in a `transformers` but does not pin its major version, and the two
majors differ in what a module returns. On 5.x a decoder block hands back a bare tensor; on 4.x
it hands back a one-element tuple, which is why so much nnsight code you will find online writes
`.output[0]`. These docs are written for 5.x throughout. Check what you have:

```python
import transformers
print(transformers.__version__)
```

## Install from PyPI

The simplest way to install NNsight is via pip:

```bash
pip install nnsight
```

## Install from Source

For the latest development version, install directly from GitHub:

```bash
pip install git+https://github.com/nnsight/nnsight.git
```

Or clone the repository and install in editable mode:

```bash
git clone https://github.com/nnsight/nnsight.git
cd nnsight
pip install -e .
```

## Optional Dependencies

### vLLM Support

For high-performance inference with [vLLM](https://github.com/vllm-project/vllm), install the `vllm` extra:

```bash
pip install nnsight[vllm]
```

### Quantized models

Loading a checkpoint in 4 or 8 bits — `TransformersModel(..., dtype="nf4")` and friends — goes
through `bitsandbytes` and `accelerate`. Neither is a dependency of nnsight, so a plain
`pip install nnsight` leaves you without them:

```bash
pip install bitsandbytes accelerate
```

## Verify Installation

```python
import nnsight
print(nnsight.__version__)      # 0.8.0
```

## Next Steps

Head to the [Quick Start](quickstart.md) and run your first intervention.
