# NNsight Docs

Check the [wiki](https://github.com/ndif-team/nnsight-website/wiki) for all the info on how to update the docs using Sphinx.

In short, this build is managed with Github Actions. Any changes you make to the source files will be built and automatically published to Github pages and served.

## Keeping in sync with NNsight

In order to build and publish this site, the correct NNsight version needs to be released. If that version is not published, things will not work properly.

## Getting Started

This project uses `uv` as a default python environment. You can get started [here](https://docs.astral.sh/uv/getting-started/installation/).

From root, run `uv sync` to set up the required packages. This will install nnsight according to whatever is in `pyproject.toml` and `uv.lock`, so if you need to upgrade it to a specific package, you can do so by editing `pyproject.toml` to reflect the correct version and running `uv sync` again.

Then, start a hotreloading webserver by running `bash run.sh`.

## Auto-Testing Notebooks

`run_notebooks.py` executes Jupyter notebooks with [papermill](https://papermill.readthedocs.io/), reports pass/fail results, and optionally updates source files with executed outputs. Each notebook gets a version info cell (nnsight, Python, torch, transformers) prepended on execution.

### Setup

```bash
uv sync --extra tutorials
```

### Usage

```bash
uv run python run_notebooks.py [options]
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
uv run python run_notebooks.py                                     # Run default (features)
uv run python run_notebooks.py -f features tutorials               # Run multiple folders
uv run python run_notebooks.py --only cross_prompt early_stopping  # Run specific notebooks
uv run python run_notebooks.py --skip vllm_support                 # Skip specific notebooks
uv run python run_notebooks.py -c -u                               # Clean outputs, update source if versions changed
uv run python run_notebooks.py -c --force-update                   # Clean outputs, always update source
```
