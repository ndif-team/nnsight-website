"""Per-page titles, meta descriptions, and paper citations for search engines.

Without this every page inherits `site_description` ("Documentation for the nnsight
Python library"), so ~100 pages ship an identical snippet and no page carries the
name of the technique or paper it is about. Search engines lean on `<title>` and
`<meta name="description">`, and people search for the *paper* ("Patchscopes",
"progress measures for grokking") rather than for us.

Notebooks have no YAML front matter, so the metadata is injected here instead:

  * ``on_page_markdown`` sets ``page.meta["title"]`` / ``["description"]``, which
    Material renders into ``<title>`` and the description/OpenGraph tags. Nav labels
    come from ``mkdocs.yml`` and are unaffected.
  * ``on_post_page`` injects a schema.org ``TechArticle`` block citing the paper the
    page reproduces, so the page can be associated with that work.

Titles are kept short — Material appends " - nnsight", and search results truncate
around 60 characters.
"""

from __future__ import annotations

import html
import json

# filename -> (title, description, paper title, paper url)
# Paper references are taken from each notebook's own citations.
PAGES: dict[str, tuple[str, str, str | None, str | None]] = {
    # -- mini papers ---------------------------------------------------------
    "jacobian-lens.ipynb": (
        "Jacobian Lens (Gurnee et al.)",
        "Reproducing the Jacobian lens in nnsight: reading a language model's residual "
        "stream through its causal effect on the output, steering with lens vectors, and "
        "fitting a lens from scratch on gemma-2-2b and GPT-2.",
        "Verbalizable Representations Form a Global Workspace in Language Models",
        "https://transformer-circuits.pub/2026/workspace/index.html",
    ),
    "patchscopes.ipynb": (
        "Patchscopes (Ghandeharioun et al.)",
        "Patchscopes in nnsight: decoding a language model's hidden states by patching them "
        "into a different prompt and letting the model explain itself. Entity description and "
        "zero-shot feature extraction on gemma-2-2b.",
        "Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models",
        "https://arxiv.org/abs/2401.06102",
    ),
    "ioi-path-patching.ipynb": (
        "IOI Path Patching (Wang et al.)",
        "Path patching the indirect object identification circuit in GPT-2 with nnsight: "
        "isolating single head-to-head edges, sweeping every upstream head into a name mover's "
        "query, and separating the q, k and v pathways.",
        "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small",
        "https://arxiv.org/abs/2211.00593",
    ),
    "grokking-progress-measures.ipynb": (
        "Progress Measures for Grokking (Nanda et al.)",
        "Grokking reproduced end to end in nnsight: train a one-layer transformer on modular "
        "addition, find the Fourier features it learns, and watch restricted and excluded loss "
        "reveal the circuit forming long before the test loss moves.",
        "Progress measures for grokking via mechanistic interpretability",
        "https://arxiv.org/abs/2301.05217",
    ),
    "othello-world-models.ipynb": (
        "Emergent World Models in Othello-GPT (Li et al.)",
        "Probing Othello-GPT's internal board with nnsight: a linear probe reads the board far "
        "better in mine/theirs coordinates than black/white, and editing along the probe's "
        "directions changes which moves the model plays.",
        "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task",
        "https://arxiv.org/abs/2210.13382",
    ),
    "concept-attention.ipynb": (
        "Concept Attention (Helbling et al.)",
        "ConceptAttention in nnsight: per-concept spatial heatmaps from a diffusion "
        "transformer's own attention, by splicing concept tokens into FLUX's encoder stream "
        "behind a one-way attention mask.",
        "ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features",
        "https://arxiv.org/abs/2502.04320",
    ),
    "vlm-logit-lens.ipynb": (
        "The Logit Lens Over Image Tokens (Neo et al.)",
        "Applying the logit lens to a vision-language model's image patches with nnsight: "
        "LLaVA-1.5's 576 image tokens read as language, producing a crude segmentation map from "
        "a model never trained to segment.",
        "Towards Interpreting Visual Information Processing in Vision-Language Models",
        "https://arxiv.org/abs/2410.07149",
    ),
    "cyclic-arithmetic.ipynb": (
        "Base-10 Arithmetic Behind Cyclic Reasoning (Feucht et al.)",
        "Llama-3.1-8B answers month and weekday arithmetic in base 10, not modular arithmetic. "
        "Fourier probes in nnsight find periods 2, 5 and 10 at layer 18 - and no period 7, even "
        "on a task built from a 7-cycle.",
        "Arithmetic in the Wild: Llama uses Base-10 Addition to Reason About Cyclic Concepts",
        "https://arxiv.org/abs/2605.01148",
    ),
    "gaze-heads.ipynb": (
        "Gaze Heads (Gandikota & Bau)",
        "Finding the attention heads a vision-language model looks through, with nnsight: score "
        "every head in one forward pass per query, then steer what the model describes with a "
        "per-head pre-softmax attention bias.",
        "Gaze Heads: How VLMs Look at What They Describe",
        "https://arxiv.org/abs/2606.14703",
    ),
    "todd_function_vectors.ipynb": (
        "Function Vectors (Todd et al.)",
        "Function vectors in nnsight: extracting the compact task representation that in-context "
        "learning builds inside a language model, and transplanting it to trigger the task "
        "without any examples.",
        "Function Vectors in Large Language Models",
        "https://arxiv.org/abs/2310.15213",
    ),
    "marks_geometry_of_truth.ipynb": (
        "The Geometry of Truth (Marks & Tegmark)",
        "The geometry of truth in nnsight: finding the linear direction that separates true from "
        "false statements in a language model's activations, and steering along it to flip the "
        "model's verdict.",
        "The Geometry of Truth: Emergent Linear Structure in Large Language Model "
        "Representations of True/False Datasets",
        "https://arxiv.org/abs/2310.06824",
    ),
    "csordas_llm_depth.ipynb": (
        "Do Language Models Use Their Depth Efficiently? (Csordás et al.)",
        "Measuring how much each layer of Llama-3.1-8B actually contributes, with nnsight: "
        "per-layer residual contributions, skip-a-layer effects on later computation, and "
        "integrated gradients over depth.",
        "Do Language Models Use Their Depth Efficiently?",
        "https://arxiv.org/abs/2505.13898",
    ),
    "feucht_dual_route_induction.ipynb": (
        "The Dual-Route Model of Induction (Feucht et al.)",
        "The dual-route model of induction in nnsight: separating token-level induction heads "
        "from concept-level ones in Llama-2-7B, and ablating each route to see what copying "
        "behaviour survives.",
        "The Dual-Route Model of Induction",
        "https://arxiv.org/abs/2504.03022",
    ),
    "huang_demystifying_memorization.ipynb": (
        "Demystifying Verbatim Memorization (Huang et al.)",
        "Probing verbatim memorization in nnsight: activation patching over memorized passages "
        "in Pythia and Llama to find where a recited continuation is actually decided.",
        "Demystifying Verbatim Memorization in Large Language Models",
        "https://arxiv.org/abs/2407.17817",
    ),
    # -- tutorials -----------------------------------------------------------
    "cross-attention-ablation.ipynb": (
        "Cross-Attention Ablation in Stable Diffusion",
        "Ablating individual cross-attention layers in Stable Diffusion with nnsight: of the 16 "
        "layers that carry the prompt into the image, exactly one turns \"Starry Night\" from a "
        "Van Gogh into a photograph.",
        None,
        None,
    ),
}

# Short descriptions for the feature guides - these are the pages people reach from
# queries like "nnsight gradients" or "nnsight batching".
FEATURES: dict[str, str] = {
    "1_getting.ipynb": "Reading activations out of any module in a model with nnsight's trace API.",
    "2_setting.ipynb": "Editing activations mid-forward with nnsight: in-place writes, replacement, and where each applies.",
    "3_gradients.ipynb": "Backward passes in nnsight: reading and editing gradients inside an interleaved trace.",
    "4_multiple_token.ipynb": "Intervening during multi-token generation with nnsight's tracer.iter.",
    "5_loading.ipynb": "Loading models in nnsight: TransformersModel, dispatch, device maps, and remote meta models.",
    "6_modules.ipynb": "Addressing modules in nnsight: envoy paths, per-architecture layouts, and calling modules ad hoc.",
    "7_model_editing.ipynb": "Persistent model edits with nnsight's edit API, applied on every later run.",
    "8_batching.ipynb": "Batching many prompts into one forward pass with nnsight's tracer.invoke.",
    "9_empty_invokers.ipynb": "Empty invokes in nnsight: batch-wide operations without an extra input.",
    "10_cache.ipynb": "Caching activations from many modules at once with nnsight's tracer.cache.",
    "11_source.ipynb": "Reaching intermediate operations inside a module's forward with nnsight's .source.",
    "12_early_stopping.ipynb": "Stopping a forward pass early in nnsight once you have the values you need.",
    "13_skip.ipynb": "Skipping a module's computation in nnsight and substituting your own value.",
    "14_scan.ipynb": "Checking shapes without running the model, using nnsight's scan and fake tensors.",
    "15_remote_execution.ipynb": "Running nnsight traces remotely on NDIF against models too large to host locally.",
    "16_vllm_support.ipynb": "Using nnsight with vLLM for high-throughput interpretability workloads.",
}

_SITE = "https://nnsight.net"


def _entry(src_uri: str):
    name = src_uri.rsplit("/", 1)[-1]
    if name in PAGES:
        return PAGES[name]
    if name in FEATURES:
        return None, FEATURES[name], None, None
    return None


def on_page_markdown(markdown, *, page, config, files):
    found = _entry(page.file.src_uri)
    if not found:
        return None
    title, description, _, _ = found
    if title and "title" not in page.meta:
        page.meta["title"] = title
    if description and "description" not in page.meta:
        page.meta["description"] = description
    return None


def on_post_page(output: str, page, config) -> str:
    """Add a schema.org TechArticle citing the paper this page reproduces."""
    found = _entry(page.file.src_uri)
    if not found:
        return output
    title, description, paper, paper_url = found
    if not paper_url:
        return output

    data = {
        "@context": "https://schema.org",
        "@type": "TechArticle",
        "headline": title or page.title,
        "description": description,
        "url": f"{_SITE}/{page.file.url}",
        "isPartOf": {"@type": "WebSite", "name": "nnsight", "url": _SITE},
        "citation": {"@type": "ScholarlyArticle", "name": paper, "url": paper_url},
    }
    block = (
        '<script type="application/ld+json">'
        + html.escape(json.dumps(data, ensure_ascii=False), quote=False)
        + "</script>"
    )
    return output.replace("</head>", block + "</head>", 1)
