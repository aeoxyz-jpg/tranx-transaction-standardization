"""Measure the local SLM's throughput at several request concurrencies.

Sends the same slm_fewshot prompt the eval uses, for a fixed sample of model-view
descriptors, to the local Ollama server with 1, 2, 4 and 8 requests in flight, and
records wall-clock throughput plus the token counts Ollama reports per request.
This is a laptop measurement: it shows how much concurrent requests help on this
machine, not what a cloud GPU would do. Whether Ollama runs requests in parallel
depends on its OLLAMA_NUM_PARALLEL setting, which is recorded as found.

Usage: python3.11 scripts/slm_throughput.py [--n 64] [--out reports/slm_throughput.json]
"""
import argparse
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from tranx import config
from tranx.pipeline.clean import strip_processor_prefix
from tranx.routes.slm_fewshot import build_prompt

CONCURRENCY = [1, 2, 4, 8]


def sample_descriptions(n: int) -> list[str]:
    """The first n descriptors of the unseen model view (sorted by txn_id), so the
    sample is fixed for a given feed."""
    from tranx.cli import eval_rows
    er = eval_rows("unseen", "model")
    return er.eval_feed.sort("txn_id")["description"].to_list()[:n]


def one_call(description: str, categories: list[str], url: str = config.OLLAMA_URL) -> dict:
    prompt = build_prompt(strip_processor_prefix(description), categories)
    r = requests.post(f"{url}/api/generate",
                      json={"model": config.SLM_MODEL, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.0}}, timeout=300)
    r.raise_for_status()
    body = r.json()
    return {"prompt_tokens": body.get("prompt_eval_count"), "output_tokens": body.get("eval_count")}


def machine() -> dict:
    info = {"platform": platform.platform(), "ollama_num_parallel": os.environ.get("OLLAMA_NUM_PARALLEL")}
    try:
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True).stdout.strip()
        mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
        info["memory_gb"] = round(int(mem) / 2**30)
    except (OSError, ValueError):
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR / "slm_throughput.json")
    ap.add_argument("--url", default=config.OLLAMA_URL)
    ap.add_argument("--num-parallel", type=int, default=None,
                    help="the server's OLLAMA_NUM_PARALLEL, recorded in the output")
    args = ap.parse_args()
    descs = sample_descriptions(args.n)
    categories = config.CATEGORIES
    call = lambda d: one_call(d, categories, args.url)
    call(descs[0])  # warm-up: load the model before timing
    runs, tokens = [], []
    for c in CONCURRENCY:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            out = list(ex.map(call, descs))
        secs = time.perf_counter() - start
        tokens = out
        runs.append({"concurrency": c, "requests": len(descs), "seconds": round(secs, 2),
                     "requests_per_second": round(len(descs) / secs, 2),
                     "ms_per_request": round(1000 * secs / len(descs), 1)})
        print(runs[-1], flush=True)
    pt = [t["prompt_tokens"] for t in tokens if t["prompt_tokens"]]
    ot = [t["output_tokens"] for t in tokens if t["output_tokens"]]
    info = machine()
    if args.num_parallel is not None:
        info["ollama_num_parallel"] = args.num_parallel
    res = {"model": config.SLM_MODEL, "machine": info, "runs": runs,
           "mean_prompt_tokens": round(sum(pt) / len(pt), 1) if pt else None,
           "mean_output_tokens": round(sum(ot) / len(ot), 1) if ot else None,
           "note": "laptop measurement; not a cloud GPU throughput"}
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
