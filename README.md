# nnsight-website

Documentation site for [nnsight](https://github.com/ndif-team/nnsight), built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

Live at [nnsight.net](https://nnsight.net).

## Prerequisites

- Python 3.11+
- [Pandoc](https://pandoc.org/installing.html) (required for notebook rendering)

## Setup

```bash
pip install -r requirements.txt
```

## Development

Start a local dev server with hot reload:

```bash
mkdocs serve
```

The site will be available at `http://127.0.0.1:8000`.

## Production Build

Build the static site into the `site/` directory:

```bash
mkdocs build
```

Output is written to `site/`. This is what gets deployed.
