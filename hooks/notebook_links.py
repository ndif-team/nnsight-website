"""Rewrite in-notebook `.ipynb` cross-links to their published URLs.

Notebooks are rendered by mkdocs-jupyter, which does not resolve markdown links
the way mkdocs resolves them in `.md` pages. A link written relative to the
`docs/` tree, e.g.

    [Logit Lens](../tutorials/probing/logit_lens.ipynb)

therefore survives verbatim into the built HTML and 404s, because every notebook
is published at `<name>/index.html` rather than next to its siblings.

This hook resolves each such link against the mkdocs file tree and replaces it
with a URL relative to the page it appears on. Links are left as `.ipynb` paths
in the source so they keep working when the notebook is opened directly on
GitHub or in Colab.

Unresolvable links are reported as build warnings rather than silently passed
through — a genuinely wrong path should be visible.
"""

from __future__ import annotations

import posixpath
import re

from mkdocs.plugins import get_plugin_logger
from mkdocs.utils import get_relative_url

log = get_plugin_logger(__name__)

# href="<path>.ipynb" and href="<path>.ipynb#anchor"
_LINK = re.compile(r'(href=")([^"#]+?\.ipynb)((?:#[^"]*)?")')

# src_uri -> published url, populated once per build
_URLS: dict[str, str] = {}


def on_files(files, config):
    _URLS.clear()
    _URLS.update({file.src_uri: file.url for file in files})
    return files


def on_post_page(output: str, page, config) -> str:
    if not _URLS:
        return output

    page_dir = posixpath.dirname(page.file.src_uri)

    def replace(match: re.Match) -> str:
        target = match.group(2)

        # Absolute URLs (http://, https://, //cdn, /root) are somebody else's problem.
        if "://" in target or target.startswith("/"):
            return match.group(0)

        # The regex sees the whole rendered page, including code samples, so a page that
        # *documents* a link (`href="*.ipynb"` in a code span) would otherwise be rewritten
        # and warned about. Nothing with a glob or angle bracket is a real path.
        if any(ch in target for ch in '*?<>| \t'):
            return match.group(0)

        src_uri = posixpath.normpath(posixpath.join(page_dir, target))
        url = _URLS.get(src_uri)
        if url is None:
            log.warning(
                "%s: link to '%s' does not resolve to a page in the docs tree "
                "(looked for '%s')",
                page.file.src_uri,
                target,
                src_uri,
            )
            return match.group(0)

        return match.group(1) + get_relative_url(url, page.file.url) + match.group(3)

    return _LINK.sub(replace, output)
