# Installation

## Requirements

- Python 3.10 or higher
- PyTorch (installed automatically)
- `transformers` 5.x, if you are working with Hugging Face models

Model implementations and architectures change from one `transformers` version to the next, and
`pip install nnsight` does not pin the major. These docs are written against 5.x, so check what
you have:

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
pip install git+https://github.com/ndif-team/nnsight.git
```

Or clone the repository and install in editable mode:

```bash
git clone https://github.com/ndif-team/nnsight.git
cd nnsight
pip install -e .
```

## Optional Dependencies

### vLLM Support

For high-performance inference with [vLLM](https://github.com/vllm-project/vllm), install the `vllm` extra:

```bash
pip install nnsight[vllm]
```

### Serving

`nnsight-serve` puts one vLLM engine behind HTTP so that clients without a GPU can trace it. The
extra pulls in vLLM plus the server dependencies:

```bash
pip install nnsight[serve]
```

### Running the tests

The `dev` extra is everything the test suite needs on top of the core install:

```bash
pip install nnsight[dev]
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
