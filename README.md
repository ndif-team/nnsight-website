# nnsight-website

Documentation site for [nnsight](https://github.com/ndif-team/nnsight), built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

Live at [nnsight.net](https://nnsight.net).

## Prerequisites

- Python 3.12+

## Setup

```bash
pip install .
```

## Development

Start a local dev server with hot reload:

```bash
# Rebuild everything at every change (slow)
mkdocs serve --livereload

# Only rebuild changed files (fast but may miss some changes, e.g. in notebooks)
mkdocs serve --livereload --dirtyreload
```

The site will be available at `http://127.0.0.1:8000`.

## Production Build

Build the static site into the `site/` directory:

```bash
mkdocs build
```

Output is written to `site/`. This is what gets deployed.

## Auto-Testing Notebooks

`run_notebooks.py` executes Jupyter notebooks with [papermill](https://papermill.readthedocs.io/), reports pass/fail results, and optionally updates source files with executed outputs. Each notebook gets a version info cell (nnsight, Python, torch, transformers) prepended on execution.

### Setup

```bash
pip install ".[test]"
```

### Usage

```bash
python run_notebooks.py [options]
```

### Options

| Flag | Description |
|------|-------------|
| `-f`, `--folders` | Folders to run from (default: `features`). Available: `features`, `tutorials`, `mini-papers` |
| `-o`, `--only` | Run only notebooks matching these names |
| `-s`, `--skip` | Skip notebooks matching these names |
| `-c`, `--clean` | Delete output notebooks after execution |
| `-u`, `--update` | Update source notebooks only if package versions differ (requires all to pass) |
| `--force-update` | Always update source notebooks with executed outputs (requires all to pass) |

### Examples

```bash
python run_notebooks.py                                          # Run default (features)
python run_notebooks.py -f features tutorials/tutorials          # Run multiple folders
python run_notebooks.py --only 7_cross_prompt 10_early_stopping  # Run specific notebooks
python run_notebooks.py --skip 15_vllm_support                   # Skip specific notebooks
python run_notebooks.py -c -u                                    # Clean outputs, update source if versions changed
python run_notebooks.py -c --force-update                        # Clean outputs, always update source
```
