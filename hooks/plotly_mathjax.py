"""Drop the MathJax loader that Plotly embeds beside every figure.

A Plotly figure shown in a notebook is saved with a `text/html` payload, and that
payload starts with

    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script>

so that LaTeX in axis titles renders in a bare notebook. On this site every
notebook page already loads MathJax 2.7.7 with the CommonHTML output through
mkdocs-jupyter, and the site's math is typeset by that instance. Loading a second
copy, once per figure, with the SVG output jax on top of it, replaces the page's
inline math with mis-sized SVG glyphs and breaks lines around them. The
`text/html` payload itself has to stay: the site does not render the
`application/vnd.plotly.v1+json` mime type on its own, so a figure saved without
it does not appear at all.

None of the figures here use LaTeX, and the page's own MathJax would handle it
if they did, so the loader is removed from the rendered page.
"""

from __future__ import annotations

import re

from mkdocs.plugins import get_plugin_logger

log = get_plugin_logger(__name__)

_PLOTLY_MATHJAX = re.compile(
    r'<script[^>]+src="[^"]*/mathjax/[^"]*MathJax\.js\?config=TeX-AMS-MML_SVG"[^>]*>\s*</script>'
)


def on_post_page(output: str, page, config) -> str:
    output, count = _PLOTLY_MATHJAX.subn("", output)
    if count:
        log.debug("%s: removed %d Plotly MathJax loader(s)", page.file.src_uri, count)
    return output
