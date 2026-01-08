# Installation

## Requirements

- Python 3.8 or higher
- pip package manager

## Install from PyPI

The simplest way to install nnsight is via pip:

```bash
pip install nnsight
```

## Install from Source

To install the latest development version:

```bash
git clone https://github.com/nnsight/nnsight.git
cd nnsight
pip install -e .
```

## Verify Installation

To verify that nnsight is installed correctly:

```python
import nnsight
print(nnsight.__version__)
```

## Optional Dependencies

For additional features, you can install optional dependencies:

```bash
pip install nnsight[all]
```
