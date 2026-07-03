"""Generate the figures for the write-up from the recorded evaluation results
(100k feed, eval-cap 1000, Qwen2.5-3B). Outputs PNGs to reports/figures/."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ROUTES = ["rules", "embedding", "slm_fewshot", "slm_lora"]
COLORS = {"random": "#4C72B0", "unseen": "#DD8452"}

# Hard-feed results (Merchant Norm, Category Acc, ms/txn) per route per split.
HARD = {
    "rules":       {"random": {"norm": 0.74, "cat": 0.78, "ms": 0.5},
                    "unseen": {"norm": 0.03, "cat": 0.57, "ms": 0.8}},
    "embedding":   {"random": {"norm": 0.90, "cat": 0.92, "ms": 8.2},
                    "unseen": {"norm": 0.00, "cat": 0.68, "ms": 8.1}},
    "slm_fewshot": {"random": {"norm": 0.81, "cat": 0.71, "ms": 569.0},
                    "unseen": {"norm": 0.78, "cat": 0.72, "ms": 582.0}},
    "slm_lora":    {"random": {"norm": 0.87, "cat": 0.82, "ms": 712.0},
                    "unseen": {"norm": 0.69, "cat": 0.76, "ms": 711.0}},
}


def _grouped(metric, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(ROUTES))
    w = 0.38
    for i, split in enumerate(["random", "unseen"]):
        vals = [HARD[r][split][metric] for r in ROUTES]
        bars = ax.bar([p + (i - 0.5) * w for p in x], vals, width=w,
                      label=split, color=COLORS[split])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ROUTES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="split")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)
    print("wrote", OUT / fname)


def fig_latency():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ms = [HARD[r]["random"]["ms"] for r in ROUTES]
    bars = ax.bar(ROUTES, ms, color=["#55A868", "#55A868", "#C44E52", "#C44E52"])
    for b, v in zip(bars, ms):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:g} ms",
                ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("ms / transaction (log scale)")
    ax.set_title("Cost per transaction — ~3 orders of magnitude apart (~1,400×)")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=130)
    plt.close(fig)
    print("wrote", OUT / "latency.png")


def fig_lora_vs_fewshot():
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = ["random\nCategory", "random\nMerchant", "unseen\nCategory", "unseen\nMerchant"]
    few = [0.71, 0.81, 0.72, 0.78]
    lora = [0.82, 0.87, 0.76, 0.69]
    x = range(len(groups))
    w = 0.38
    b1 = ax.bar([p - w / 2 for p in x], few, width=w, label="few-shot", color="#8172B3")
    b2 = ax.bar([p + w / 2 for p in x], lora, width=w, label="LoRA fine-tune", color="#CCB974")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Fine-tune vs few-shot (Qwen2.5-3B, hard feed)\n"
                 "LoRA wins everywhere except unseen-merchant generalization")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "lora_vs_fewshot.png", dpi=130)
    plt.close(fig)
    print("wrote", OUT / "lora_vs_fewshot.png")


def _heat(v):
    """Green/yellow/red shading for a 0-1 accuracy cell."""
    if v >= 0.70:
        return "#C8E6C9"
    if v >= 0.40:
        return "#FFF3C4"
    return "#F8C9C4"


def _table(headers, rows, title, fname, col_widths, heat_cols=(), figsize=(11, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center",
                   colWidths=col_widths, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    n_cols = len(headers)
    # rows with multi-line cell text need extra height so they don't overlap
    # the row below them.
    row_lines = {0: 1}
    for ridx, row in enumerate(rows, start=1):
        row_lines[ridx] = max(str(cell_text).count("\n") + 1 for cell_text in row)
    base_height = next(iter(tbl.get_celld().values())).get_height()
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        lines = row_lines.get(r, 1)
        if lines > 1:
            cell.set_height(base_height * (lines * 0.65 + 0.35))
        if r == 0:
            cell.set_facecolor("#34495E")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if c in heat_cols:
                try:
                    cell.set_facecolor(_heat(float(rows[r - 1][c])))
                except ValueError:
                    pass
            elif r % 2 == 0:
                cell.set_facecolor("#F7F7F7")
        if c == 0 and r > 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / fname)


def fig_engines_table():
    headers = ["Route", "Merchant normalization", "Category", "Cost / txn"]
    rows = [
        ["rules", "regex clean + rapidfuzz token-set\nvs a learned merchant list",
         "learned MCC & merchant priors", "<1 ms"],
        ["embedding", "nearest canonical in\nMiniLM embedding space",
         "logistic regression", "~8 ms"],
        ["slm_fewshot", "Qwen2.5-3B few-shot (Ollama)", "the LLM", "~570 ms"],
        ["slm_lora", "Qwen2.5-3B LoRA fine-tuned (MLX)", "the LLM", "~710 ms"],
    ]
    _table(headers, rows, "Four standardization engines", "table_engines.png",
           col_widths=[0.16, 0.42, 0.27, 0.15], figsize=(11, 3.6))


def fig_results_table():
    headers = ["Route", "Split", "Category", "Merchant\n(exact)",
               "Merchant\n(norm)", "ms / txn"]
    rows = [
        ["rules", "random", "0.78", "0.72", "0.74", "0.5"],
        ["embedding", "random", "0.92", "0.90", "0.90", "8.2"],
        ["slm_fewshot", "random", "0.71", "0.70", "0.81", "569"],
        ["slm_lora", "random", "0.82", "0.85", "0.87", "712"],
        ["rules", "unseen", "0.57", "0.00", "0.03", "0.8"],
        ["embedding", "unseen", "0.68", "0.00", "0.00", "8.1"],
        ["slm_fewshot", "unseen", "0.72", "0.67", "0.78", "582"],
        ["slm_lora", "unseen", "0.76", "0.60", "0.69", "711"],
    ]
    _table(headers, rows,
           "Results on the hard feed (Qwen2.5-3B, eval-cap 1000)\n"
           "merchant-norm column shaded green/yellow/red",
           "table_results.png",
           col_widths=[0.18, 0.13, 0.16, 0.16, 0.16, 0.13],
           heat_cols=(4,), figsize=(11, 4.4))


if __name__ == "__main__":
    _grouped("norm",
             "Merchant normalization on dirty descriptors\n"
             "Only the few-shot SLM survives unseen merchants",
             "Merchant Norm (case/punctuation-insensitive)", "merchant_norm.png")
    _grouped("cat", "Category accuracy on dirty descriptors",
             "Category Acc", "category.png")
    fig_lora_vs_fewshot()
    fig_latency()
    fig_engines_table()
    fig_results_table()
