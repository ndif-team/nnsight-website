# Installation

## Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher

## Install from PyPI

The simplest way to install nnsight is via pip:

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

For high-performance inference with vLLM:

```bash
pip install nnsight[vllm]
```

## Verify Installation

Verify your installation by running:

```python
import nnsight
print(nnsight.__version__)
```

## Next Steps

Once installed, head to the [Quick Start](quickstart.md) guide to run your first intervention!
