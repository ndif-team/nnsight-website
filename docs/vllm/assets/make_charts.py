"""Throughput charts + tables for comparisons.md, from the ie-bench jsonl files.

Each hardware config becomes one inline SVG (dot plot, one row per workload, x = share of
plain vLLM doing the same generation with nothing attached) and one markdown table.
"""
import json, statistics, sys, pathlib
from collections import defaultdict

RESULTS = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [  # slug, model key, file, title, row-label substitutions
    ("llama-8b", "llama-8b", "results-v2.jsonl", "Llama-3.1-8B, one GPU", {}),
    ("llama-8b-tp2", "llama-8b-tp2", "results-tp.jsonl", "Llama-3.1-8B, tensor-parallel 2", {}),
    ("llama-8b-tp4", "llama-8b-tp4", "results-tp.jsonl", "Llama-3.1-8B, tensor-parallel 4", {}),
    ("llama-70b", "llama-70b", "results-70b.jsonl", "Llama-3.1-70B, tensor-parallel 4", {}),
    ("qwen3-8b", "qwen3-8b", "results-qwen3.jsonl", "Qwen3-8B, one GPU", {}),
    ("qwen-moe", "qwen-moe", "results-moe.jsonl", "Qwen1.5-MoE-A2.7B (mixture of experts), one GPU", {}),
    ("llama-1b", "llama-1b", "results-v2.jsonl", "Llama-3.2-1B, one GPU", {}),
    ("llama-8b-long", "llama-8b", "results-long.jsonl", "Llama-3.1-8B, 2048-token prompt, 512 new tokens", {}),
    ("llama-8b-x32", "llama-8b", "results-x32.jsonl", "Llama-3.1-8B, 32 concurrent requests", {"8 concurrent": "32 concurrent"}),
    ("deepseek-v2-lite", "deepseek-v2-lite", "results-dsv2.jsonl", "DeepSeek-V2-Lite (MLA + MoE), one GPU", {}),
    ("qwen35-08b", "qwen35-08b", "results-qwen35.jsonl", "Qwen3.5-0.8B, one GPU", {}),
    ("qwen35-4b", "qwen35-4b", "results-qwen35.jsonl", "Qwen3.5-4B, one GPU", {}),
    ("qwen36-moe", "qwen36-moe", "results-qwen36moe.jsonl", "Qwen3.6-35B-A3B (mixture of experts), tensor-parallel 2", {}),
    ("llama-8b-tp8", "llama-8b-tp8", "results-tp8.jsonl", "Llama-3.1-8B, tensor-parallel 8", {}),
    ("llama-70b-tp8", "llama-70b-tp8", "results-tp8.jsonl", "Llama-3.1-70B, tensor-parallel 8", {}),
]
# series: key, label, hue slot, filled?
SERIES = [
    ("ns_vllm", "nnsight eager", "blue", False),
    ("ns_taps", "nnsight taps (CUDA graphs)", "blue", True),
    ("ie_vllm", "interp-engine vllm", "orange", False),
    ("ie_static", "interp-engine vllm-static", "orange", True),
    ("lens_vllm", "vLLM-Lens", "aqua", True),
]
HUE = {"blue": ("#2a78d6", "#3987e5"), "orange": ("#eb6834", "#d95926"), "aqua": ("#1baf7a", "#199e70")}
# rows: key, label, reference row on the vanilla column, lower-is-better?
ROWS = [
    ("gen", "generate", "gen", False),
    ("gen_x8", "generate, 8 concurrent", "gen_x8", False),
    ("cap1", "capture 1 layer, every step", "gen", False),
    ("capall", "capture every layer, every step", "gen", False),
    ("cap1_x8", "capture 1 layer, 8 concurrent", "gen_x8", False),
    ("steer", "additive steering, 1 layer", "gen", False),
    ("lens", "logit lens every step", "gen", False),
    ("probe", "linear probe every step", "gen", False),
    ("ablate", "zero one attention head every step", "gen", False),
    ("force", "override the sampled token every step", "gen", False),
    ("sweep_cap", "sweep: 1024 × 1 token, capture 1 layer, per request", "sweep_cap", True),
    ("sweep_cap_edit", "sweep: 1024 × 1 token, capture 1 layer, edit() once", "sweep_cap", True),
]
TABLE_COLS = [("vanilla", "vanilla vLLM")] + [(k, l) for k, l, _, _ in SERIES]


def load(path):
    res = defaultdict(list)
    for line in open(path):
        e = json.loads(line)
        res[(e["column"], e["row"])].append(e)
    return res


def value(entries):
    """(mean, status, note) — pooling only runs consistent with the latest (drops stale reruns)."""
    if not entries:
        return None, "missing", "", []
    last = entries[-1]
    if last["status"] != "ok":
        return None, last["status"], last.get("note", ""), []
    def m(e):
        s = e.get("samples") or [e.get("tok_s", e.get("ms"))]
        return statistics.mean(s)
    ref = m(last)
    pooled = [s for e in entries[-3:] if e["status"] == "ok" and abs(m(e) - ref) / ref < 0.15
              for s in (e.get("samples") or [m(e)])]
    return statistics.mean(pooled), "ok", last.get("note", ""), pooled


def mann_whitney_p(a, b):
    """Exact two-sided Mann-Whitney U p-value (permutation over all splits; fine for n <= 12)."""
    from itertools import combinations
    a, b = list(a), list(b)
    if len(a) < 2 or len(b) < 2:
        return None
    def u_stat(x, y):
        return sum((xi > yj) + 0.5 * (xi == yj) for xi in x for yj in y)
    pooled = a + b
    n = len(a)
    u_obs = u_stat(a, b)
    mean_u = len(a) * len(b) / 2
    extreme = 0
    total = 0
    for idx in combinations(range(len(pooled)), n):
        x = [pooled[i] for i in idx]
        y = [pooled[i] for i in range(len(pooled)) if i not in idx]
        total += 1
        if abs(u_stat(x, y) - mean_u) >= abs(u_obs - mean_u) - 1e-9:
            extreme += 1
    return extreme / total


COUNTERPART = {"ie_vllm": "ns_vllm", "lens_vllm": "ns_vllm", "ie_static": "ns_taps"}
ALPHA = 0.05
MIN_EFFECT = 0.03   # a significant difference under 3% is real but not worth a highlight


def fmt(v, row):
    if v is None:
        return "·"
    if row.startswith("sweep"):
        return f"{v / 1000:.2f} s"
    return f"{v:,.0f}"


def svg(config, title, res, relabel=None):
    W, LEFT, RIGHT, TOP, ROWH = 860, 370, 40, 120, 26
    rows = [r for r in ROWS if res.get(("vanilla", r[2])) and any(res.get((c, r[0])) for c, *_ in SERIES + [("vanilla",)])]
    H = TOP + len(rows) * ROWH + 46
    plot_w = W - LEFT - RIGHT
    xmax = 1.12
    for row, _, refrow, lower in rows:
        van = value(res.get(("vanilla", refrow)))[0]
        for key, *_ in SERIES:
            v = value(res.get((key, row)))[0]
            if v and van:
                xmax = max(xmax, ((van / v) if lower else (v / van)) + 0.08)
    gridfracs = [f / 100 for f in range(25, int(xmax * 100) + 1, 25)]

    def X(frac):
        return LEFT + frac / xmax * plot_w

    light = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
    dark = {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"}
    css = [".viz{--surface:%s;--ink:%s;--ink2:%s;--muted:%s;--grid:%s;--axis:%s;--blue:%s;--orange:%s;--aqua:%s;font-family:system-ui,-apple-system,'Segoe UI',sans-serif}" % (
        light["surface"], light["ink"], light["ink2"], light["muted"], light["grid"], light["axis"], HUE["blue"][0], HUE["orange"][0], HUE["aqua"][0])]
    darkvars = "--surface:%s;--ink:%s;--ink2:%s;--muted:%s;--grid:%s;--axis:%s;--blue:%s;--orange:%s;--aqua:%s" % (
        dark["surface"], dark["ink"], dark["ink2"], dark["muted"], dark["grid"], dark["axis"], HUE["blue"][1], HUE["orange"][1], HUE["aqua"][1])
    css.append("@media (prefers-color-scheme: dark){:root:not([data-md-color-scheme=default]) .viz{%s}}" % darkvars)
    css.append("[data-md-color-scheme=slate] .viz{%s}" % darkvars)
    css.append(".viz text{fill:var(--ink2);font-size:12px}.viz .t{fill:var(--ink);font-size:14px;font-weight:600}.viz .m{fill:var(--muted);font-size:11px}"
               ".viz .g{stroke:var(--grid);stroke-width:1}.viz .ref{stroke:var(--axis);stroke-width:1.5}.viz .hit{fill:transparent}"
               ".viz .hit:hover + circle{r:7}")
    out = [f'<svg class="viz" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{title}: throughput of each library as a share of plain vLLM">',
           "<style>" + "".join(css) + "</style>",
           f'<rect width="{W}" height="{H}" fill="var(--surface)" rx="6"/>',
           f'<text class="t" x="16" y="26">{title}</text>',
           f'<text class="m" x="16" y="44">Throughput as a share of plain vLLM doing the same generation with nothing attached (100% = vanilla).</text>',
           f'<text class="m" x="16" y="60">Hollow = eager, filled = CUDA graphs. Whisker = min–max over runs. nnsight dots are smaller and in front, so a tie shows both. ✗ = cannot express.</text>',
           f'<text class="m" x="16" y="76">Hover a dot for the number, its run count and range, and whether it is significantly faster than its nnsight counterpart (exact Mann-Whitney U, p&lt;0.05, ≥3% apart).</text>']
    # legend
    lx, ly = 16, TOP - 20
    for key, label, hue, filled in SERIES:
        if not any(res.get((key, r[0])) for r in rows):
            continue
        w = 16 + 6.6 * len(label) + 20
        if lx + w > W - 16:
            lx, ly = 16, ly + 18
        fill = f"var(--{hue})" if filled else "var(--surface)"
        out.append(f'<circle cx="{lx + 6}" cy="{ly - 4}" r="{5 if key.startswith("ns_") else 5.8}" fill="{fill}" stroke="var(--{hue})" stroke-width="2"/>')
        out.append(f'<text x="{lx + 16}" y="{ly}">{label}</text>')
        lx += w
    # grid + axis
    for frac in gridfracs:
        x = X(frac)
        cls = "ref" if frac == 1.0 else "g"
        out.append(f'<line class="{cls}" x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP + len(rows) * ROWH}"/>')
        out.append(f'<text class="m" x="{x:.1f}" y="{TOP + len(rows) * ROWH + 16}" text-anchor="middle">{int(frac * 100)}%</text>')
    out.append(f'<text class="m" x="{X(1.0):.1f}" y="{TOP + len(rows) * ROWH + 32}" text-anchor="middle">vanilla vLLM</text>')
    # rows
    table = []
    for i, (row, label, refrow, lower) in enumerate(rows):
        for a, b in (relabel or {}).items():
            label = label.replace(a, b)
        y = TOP + i * ROWH + ROWH / 2
        out.append(f'<text x="{LEFT - 12}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
        van, vstat, _, _ = value(res.get(("vanilla", refrow)))
        cells = {"vanilla": fmt(value(res.get(("vanilla", row)))[0], row)}
        vals = {key: value(res.get((key, row))) for key, *_ in SERIES}
        # significance against the nnsight counterpart (two-sided exact Mann-Whitney U)
        sig = {}
        better = lambda x, y: (x < y) if lower else (x > y)
        for key, cp in COUNTERPART.items():
            vk, vc = vals.get(key), vals.get(cp)
            if vk and vc and vk[1] == "ok" and vc[1] == "ok":
                pval = mann_whitney_p(vk[3], vc[3])
                if pval is not None and pval < ALPHA and abs(vk[0] - vc[0]) / vc[0] >= MIN_EFFECT:
                    sig[key] = "faster" if better(vk[0], vc[0]) else "slower"
        for ns in ("ns_vllm", "ns_taps"):
            rivals = [k for k, cp in COUNTERPART.items() if cp == ns and vals.get(k) and vals[k][1] == "ok"]
            if rivals and all(sig.get(k) == "slower" for k in rivals):
                sig[ns] = "faster"
        nx = 0
        draw_order = sorted(SERIES, key=lambda t: t[0].startswith("ns_"))   # nnsight painted last = on top
        for key, slabel, hue, filled in draw_order:
            v, stat, note, samples = vals[key]
            r_dot = 5 if key.startswith("ns_") else 5.8
            cells[key] = fmt(v, row) if stat == "ok" else ("✗" if stat == "unsupported" else "·")
            if sig.get(key) == "faster":
                cells[key] = f"**{cells[key]}**"
            if stat == "unsupported":
                fill = f"var(--{hue})" if filled else "var(--surface)"
                out.append(f'<circle cx="{LEFT + 10 + nx * 16}" cy="{y:.1f}" r="3.5" fill="{fill}" stroke="var(--{hue})" stroke-width="1.5" opacity="0.7"><title>{slabel}: {note}</title></circle>')
                out.append(f'<text class="m" x="{LEFT + 10 + nx * 16}" y="{y + 4:.1f}" text-anchor="middle" font-size="9">✗</text>')
                nx += 1
                continue
            if v is None or van is None:
                continue
            frac = (van / v) if lower else (v / van)
            x = X(min(frac, xmax))
            fill = f"var(--{hue})" if filled else "var(--surface)"
            lo_s, hi_s = min(samples), max(samples)
            f_lo, f_hi = sorted(((van / lo_s) if lower else (lo_s / van), (van / hi_s) if lower else (hi_s / van)))
            tip = f"{slabel}: {fmt(v, row)} ({frac * 100:.0f}% of vanilla; {len(samples)} runs, {fmt(lo_s, row)}–{fmt(hi_s, row)})"
            if key in sig:
                tip += f"; significantly {sig[key]} than {'the counterpart' if key != 'ns_vllm' and key != 'ns_taps' else 'every counterpart'} (p&lt;{ALPHA})"
            out.append(f'<line x1="{X(min(f_lo, xmax)):.1f}" y1="{y:.1f}" x2="{X(min(f_hi, xmax)):.1f}" y2="{y:.1f}" stroke="var(--{hue})" stroke-width="2" opacity="0.45"/>')
            out.append(f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12"><title>{tip}</title></circle>')
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r_dot}" fill="{fill}" stroke="var(--{hue})" stroke-width="2" style="paint-order:stroke" pointer-events="none"/>')
        table.append((label, cells))
    out.append("</svg>")
    return '<div class="viz-wrap">\n' + "\n".join(out) + "\n</div>", table


for slug, key, fname, title, relabel in CONFIGS:
    if not (RESULTS / fname).exists():
        continue
    res = load(RESULTS / fname)
    alias = {"llama-70b-lens": "llama-70b"}          # the vLLM-Lens 70B job ran under its own key
    res = {k: [e for e in v if alias.get(e["model"], e["model"]) == key] for k, v in res.items()}
    res = {k: v for k, v in res.items() if v}
    if not res.get(("vanilla", "gen")) and not res.get(("vanilla", "gen_x8")):
        continue
    s, table = svg(slug, title, res, relabel)
    (OUT / f"throughput-{slug}.svg").write_text(s)
    cols = [c for c in TABLE_COLS if any(res.get((c[0], r[0])) for r in ROWS)]
    md = [f"| workload | " + " | ".join(l for _, l in cols) + " |", "| --- |" + " ---: |" * len(cols)]
    for label, cells in table:
        md.append(f"| {label} | " + " | ".join(cells.get(c, "·") for c, _ in cols) + " |")
    (OUT / f"throughput-{slug}.md").write_text("\n".join(md) + "\n")
    print(slug, "rows", len(table), "cols", [c for c, _ in cols])
