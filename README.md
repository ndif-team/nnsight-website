# NNsight Docs

Check the [wiki](https://github.com/ndif-team/nnsight-website/wiki) for all the info on how to update the docs using Sphinx.

In short, this build is managed with Github Actions. Any changes you make to the source files will be built and automatically published to Github pages and served.

## Keeping in sync with NNsight

In order to build and publish this site, the correct NNsight version needs to be released. If that version is not published, things will not work properly.

## Getting Started

This project uses `uv` as a default python environment. You can get started [here](https://docs.astral.sh/uv/getting-started/installation/).

From root, run `uv sync` to set up the required packages. This will install nnsight according to whatever is in `pyproject.toml` and `uv.lock`, so if you need to upgrade it to a specific package, you can do so by editing `pyproject.toml` to reflect the correct version and running `uv sync` again.

Then, start a hotreloading webserver by running `bash run.sh`.
