# Getting Started

Start here. nnsight lets you look inside a model's forward pass and change what happens in it.
You wrap a model you already have, open a `with model.trace(...)` block, and read or edit any
internal value as ordinary Python. The model itself doesn't change, and there are no hooks to
register.

Two pages get you running. [Installation](installation.md) is `pip install nnsight` and the
optional extras. [Quick Start](quickstart.md) is your first trace end to end: load a model, read
a hidden state, edit one, and get the values back out. It uses GPT-2 and runs on CPU, so you can
follow along without a GPU.

After that, [Features](../features/index.md) has a page per capability,
[Tutorials](../tutorials/index.md) has worked interpretability experiments, and the
[Walkthrough](../tutorials/tutorials/get_started/walkthrough.ipynb) is the long-form guided tour.

## :material-download: [Installation](installation.md)

`pip install nnsight`, the optional extras, and what to install alongside it.

## :material-rocket-launch: [Quick Start](quickstart.md)

Load a model, read a hidden state, change one, and get the values back out.
