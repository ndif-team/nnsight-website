"""Tag each page with metadata used by the custom social card layout.

  * ``page.meta["card_type"]`` — short content-type label ("Guide", "API",
    "Tutorial", ...) shown in the card.
  * ``page.meta["api_path"]`` — qualified Python path (e.g.
    ``nnsight.intervention.envoy``) for pages under ``documentation/``.
  * ``page.meta["author_name"]`` / ``page.meta["author_avatar"]`` — first
    author's display name and a circular-masked PNG (cached locally) for
    blog posts.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import requests
import yaml
from PIL import Image, ImageDraw

# Map URL prefixes to a human content-type label. Order matters; the first
# matching prefix wins.
_TYPE_BY_PREFIX = (
    ("tutorials/", "Tutorial"),
    ("features/", "Feature"),
    ("documentation/", "API"),
    ("getting-started/", "Guide"),
    ("about/", "About"),
    ("status/", "Status"),
)

_API_PREFIX = "documentation/"
_API_ROOT_MODULE = "nnsight"

_AVATAR_CACHE_DIR = Path(".cache/plugin/social/authors")
_AVATAR_SIZE = 240
_AVATAR_TIMEOUT = 10

_authors_cache: dict | None = None


def _derive_card_type(src_uri: str, meta: dict) -> str | None:
    if src_uri.startswith("blog/"):
        return "Blog"
    for prefix, label in _TYPE_BY_PREFIX:
        if src_uri.startswith(prefix):
            return label
    return None


def _derive_api_path(src_uri: str) -> str | None:
    if not src_uri.startswith(_API_PREFIX):
        return None
    path = src_uri[len(_API_PREFIX):].removesuffix(".md")
    if path.endswith("/index"):
        path = path[: -len("/index")]
    if not path:
        return None
    return f"{_API_ROOT_MODULE}." + path.replace("/", ".")


def _load_authors(docs_dir: Path) -> dict:
    global _authors_cache
    if _authors_cache is not None:
        return _authors_cache
    authors_file = docs_dir / "blog" / ".authors.yml"
    if not authors_file.exists():
        _authors_cache = {}
        return _authors_cache
    with open(authors_file) as f:
        data = yaml.safe_load(f) or {}
    _authors_cache = data.get("authors", {})
    return _authors_cache


def _resolve_avatar(author_id: str, info: dict, docs_dir: Path) -> str | None:
    avatar_src = info.get("avatar")
    if not avatar_src:
        return None

    _AVATAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{author_id}|{avatar_src}".encode()).hexdigest()[:12]
    cache_path = _AVATAR_CACHE_DIR / f"{author_id}-{key}.png"
    if cache_path.exists():
        return str(cache_path)

    try:
        if avatar_src.startswith(("http://", "https://")):
            resp = requests.get(
                avatar_src,
                timeout=_AVATAR_TIMEOUT,
                headers={"User-Agent": "nnsight-docs-build"},
            )
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
        else:
            local = (docs_dir / "blog" / avatar_src).resolve()
            if not local.exists():
                return None
            img = Image.open(local)
    except Exception:
        return None

    img = img.convert("RGBA")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w + side) // 2, (h + side) // 2))
    img = img.resize((_AVATAR_SIZE, _AVATAR_SIZE), Image.LANCZOS)

    mask = Image.new("L", (_AVATAR_SIZE, _AVATAR_SIZE), 0)
    ImageDraw.Draw(mask).ellipse(
        (0, 0, _AVATAR_SIZE, _AVATAR_SIZE), fill=255
    )

    out = Image.new("RGBA", (_AVATAR_SIZE, _AVATAR_SIZE), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    out.save(cache_path)
    return str(cache_path)


def on_page_markdown(markdown, *, page, config, files):
    src = page.file.src_uri

    if "card_type" not in page.meta:
        ct = _derive_card_type(src, page.meta)
        if ct:
            page.meta["card_type"] = ct

    if "api_path" not in page.meta:
        ap = _derive_api_path(src)
        if ap:
            page.meta["api_path"] = ap

    if src.startswith("blog/posts/"):
        author_ids = page.meta.get("authors") or []
        if author_ids:
            authors = _load_authors(Path(config.docs_dir))
            info = authors.get(author_ids[0])
            if info:
                page.meta.setdefault("author_name", info.get("name", ""))
                avatar = _resolve_avatar(
                    author_ids[0], info, Path(config.docs_dir)
                )
                if avatar:
                    page.meta.setdefault("author_avatar", avatar)

    return None
