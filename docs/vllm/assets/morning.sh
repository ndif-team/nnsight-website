#!/bin/bash
# Pull the overnight benchmark results from hakone, regenerate the throughput charts/tables,
# add panels to comparisons.md for any new configs that now have data, and rebuild the site.
set -e
SITE=/home/localjadenfk/wd/nnsight-website
RES=/tmp/ie-bench-results
mkdir -p $RES
rsync -q hakone:/disk/u/localjadenfk/ie-bench/results-*.jsonl $RES/
echo "--- overnight queue log ---"
ssh hakone 'cat /disk/u/localjadenfk/ie-bench/logs/overnight-w*.log | sort | grep -v "^$"' | tail -60
cd $SITE
.venv/bin/python docs/vllm/assets/make_charts.py $RES docs/vllm/assets
python3 - <<'EOF'
import pathlib, re
site = pathlib.Path("/home/localjadenfk/wd/nnsight-website")
page = site / "docs/vllm/comparisons.md"; s = page.read_text()
panels = [("qwen3-8b", "Qwen3-8B, one GPU"), ("qwen-moe", "Qwen1.5-MoE-A2.7B, one GPU"),
          ("llama-1b", "Llama-3.2-1B, one GPU"), ("llama-8b-long", "Llama-3.1-8B, 2048-token prompt, 512 new tokens"),
          ("llama-8b-x32", "Llama-3.1-8B, 32 concurrent"), ("deepseek-v2-lite", "DeepSeek-V2-Lite, one GPU"),
          ("qwen35-08b", "Qwen3.5-0.8B, one GPU"), ("qwen35-4b", "Qwen3.5-4B, one GPU"),
          ("qwen36-moe", "Qwen3.6-35B-A3B (MoE), tp=2"),
          ("llama-8b-tp8", "Llama-3.1-8B, tp=8"), ("llama-70b-tp8", "Llama-3.1-70B, tp=8")]
added = []
for slug, title in panels:
    if not (site / f"docs/vllm/assets/throughput-{slug}.svg").exists() or f"throughput-{slug}.svg" in s:
        continue
    block = f'''
--8<-- "docs/vllm/assets/throughput-{slug}.svg"

??? note "The numbers — {title}"

    --8<-- "docs/vllm/assets/throughput-{slug}.md"
'''
    marker = "The harness (`ie-bench/`"
    s = s.replace(marker, block.strip("\n") + "\n\n" + marker, 1)
    added.append(slug)
page.write_text(s)
print("panels added:", added or "none new")
EOF
timeout 900 .venv/bin/mkdocs build -q 2>&1 | grep -i "vllm" | head -5 || true
echo "built; charts in docs/vllm/assets, page docs/vllm/comparisons.md"
