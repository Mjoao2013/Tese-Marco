"""
Phase 3 Visualization — generates all PNG result plots.
Run from the Tese root directory: python 04_Phase3/generate_phase3_plots.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
OUT       = ROOT / "Results" / "Images"
OUT.mkdir(parents=True, exist_ok=True)

BASE_DIR = ROOT / "Results" / "Baseline"
OPT_DIR  = ROOT / "Results" / "Optimized"
ENH_DIR  = ROOT / "Results" / "Enhanced"

base_data = json.loads((BASE_DIR / "phase3_categories_results.json").read_text())
opt_data  = json.loads((OPT_DIR  / "phase3_opt_results.json").read_text())
enh_data  = json.loads((ENH_DIR  / "phase3_enhanced_results.json").read_text())

STYLE_COLORS = ["#6c757d", "#4dabf7", "#ff8c00", "#f03e3e", "#2f9e44"]
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── 1. Comprehensive Model Comparison ─────────────────────────────────────────
print("Generating: phase3_model_comparison.png")
models = [
    "TF-IDF BR",
    "DistilBERT\nBaseline",
    "BERT-base\nOptimized",
    "BERT-base\nEnhanced\n(global)",
    "BERT-base\nEnhanced\n(per-label)",
]
metrics_data = {
    "Micro-F1": [
        base_data["tfidf"]["micro_f1"],
        base_data["bert_br"]["micro_f1"],
        opt_data["bert_br_global_thresh"]["micro_f1"],
        enh_data["bert_br_global_thresh"]["micro_f1"],
        enh_data["bert_br_perlabel_thresh"]["micro_f1"],
    ],
    "Macro-F1": [
        base_data["tfidf"]["macro_f1"],
        base_data["bert_br"]["macro_f1"],
        opt_data["bert_br_perlabel_thresh"]["macro_f1"],
        enh_data["bert_br_global_thresh"]["macro_f1"],
        enh_data["bert_br_perlabel_thresh"]["macro_f1"],
    ],
    "LRAP": [
        base_data["tfidf"]["lrap"],
        base_data["bert_br"]["lrap"],
        opt_data["bert_br_global_thresh"]["lrap"],
        enh_data["bert_br_global_thresh"]["lrap"],
        enh_data["bert_br_perlabel_thresh"]["lrap"],
    ],
    "Hamming Loss": [
        base_data["tfidf"]["hamming"],
        base_data["bert_br"]["hamming"],
        opt_data["bert_br_global_thresh"]["hamming"],
        enh_data["bert_br_global_thresh"]["hamming"],
        enh_data["bert_br_perlabel_thresh"]["hamming"],
    ],
}

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.suptitle("Phase 3: Category Classification — Model Comparison (85 labels, 165k games)",
             fontsize=14, fontweight="bold", y=1.02)

for ax, (metric, values) in zip(axes, metrics_data.items()):
    bars = ax.bar(range(len(models)), values, color=STYLE_COLORS, edgecolor="white",
                  linewidth=0.8, width=0.65)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=7.5)
    if metric == "Hamming Loss":
        ax.set_ylabel("Lower is better", fontsize=9, color="#666")
    else:
        ax.set_ylabel("Higher is better", fontsize=9, color="#666")
        ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

legend_patches = [mpatches.Patch(color=c, label=m) for c, m in zip(STYLE_COLORS, models)]
fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
plt.savefig(OUT / "phase3_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  DONE")

# ── 2. Training Curves (Enhanced model) ───────────────────────────────────────
print("Generating: phase3_training_curves.png")
history = enh_data["history"]
epochs  = list(range(1, len(history["train_loss"]) + 1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase 3 Enhanced (BERT-base) — Training Curves", fontsize=13, fontweight="bold")

ax1.plot(epochs, history["train_loss"], "o-", color="#4dabf7", linewidth=2,
         markersize=5, label="Train Loss")
ax1.plot(epochs, history["val_loss"], "s-", color="#f03e3e", linewidth=2,
         markersize=5, label="Val Loss")
best_epoch = np.argmin(history["val_loss"]) + 1
ax1.axvline(best_epoch, color="#666", linestyle="--", alpha=0.7)
ax1.text(best_epoch + 0.1, ax1.get_ylim()[0] + 0.01,
         f"Best\nepoch {best_epoch}", fontsize=8, color="#444")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("BCE Loss", fontsize=11)
ax1.set_title("Loss (early stop on val loss)", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3, linestyle="--")
ax1.set_xticks(epochs)

ax2.plot(epochs, history["val_micro_f1"], "o-", color="#2f9e44", linewidth=2,
         markersize=5, label="Val Micro-F1")
ax2.plot(epochs, history["val_macro_f1"], "s-", color="#ff8c00", linewidth=2,
         markersize=5, label="Val Macro-F1")
ax2.axhline(enh_data["bert_br_perlabel_thresh"]["micro_f1"], color="#2f9e44",
            linestyle=":", alpha=0.6, label=f"Test Micro-F1 = {enh_data['bert_br_perlabel_thresh']['micro_f1']:.4f}")
ax2.axhline(enh_data["bert_br_perlabel_thresh"]["macro_f1"], color="#ff8c00",
            linestyle=":", alpha=0.6, label=f"Test Macro-F1 = {enh_data['bert_br_perlabel_thresh']['macro_f1']:.4f}")
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("F1 Score", fontsize=11)
ax2.set_title("Validation F1 (threshold=0.5 during training)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, linestyle="--")
ax2.set_xticks(epochs)
ax2.set_ylim(0, 0.75)

plt.tight_layout()
plt.savefig(OUT / "phase3_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  DONE")

# ── 3. Per-Label F1 Scatter (support vs F1) ───────────────────────────────────
print("Generating: phase3_per_label_scatter.png")
per_label = enh_data["per_label_f1"]
supports  = [x["support"] for x in per_label]
f1s       = [x["f1"]      for x in per_label]
labels    = [x["label"]   for x in per_label]

fig, ax = plt.subplots(figsize=(12, 7))

sc = ax.scatter(supports, f1s, c=f1s, cmap="RdYlGn", s=60, alpha=0.8,
                edgecolors="white", linewidths=0.5, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label="F1 Score", pad=0.01)

# Annotate extremes
for i, (s, f, lbl) in enumerate(zip(supports, f1s, labels)):
    if f > 0.88 or f < 0.15 or s > 5000:
        ax.annotate(lbl, (s, f), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, alpha=0.9,
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.5))

# Spearman reference line
from scipy.stats import spearmanr
r, p = spearmanr(supports, f1s)
ax.axhline(np.mean(f1s), color="#666", linestyle="--", alpha=0.5, linewidth=1)
ax.text(max(supports)*0.01, np.mean(f1s)+0.01, f"Mean F1 = {np.mean(f1s):.3f}",
        fontsize=9, color="#555")

ax.set_xscale("log")
ax.set_xlabel("Label Support (log scale) — number of test examples", fontsize=12)
ax.set_ylabel("Per-Label F1 Score (per-label threshold)", fontsize=12)
ax.set_title(f"Phase 3 Enhanced: Per-Label F1 vs Support — 85 categories\n"
             f"Spearman r = {r:.3f}  (p = {p:.3f})", fontsize=13, fontweight="bold")
ax.grid(alpha=0.25, linestyle="--")

plt.tight_layout()
plt.savefig(OUT / "phase3_per_label_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  DONE")

# ── 4. Top & Bottom 20 Labels by F1 ──────────────────────────────────────────
print("Generating: phase3_top_bottom_labels.png")
sorted_pl  = sorted(enh_data["per_label_f1"], key=lambda x: x["f1"])
bottom_20  = sorted_pl[:20]
top_20     = sorted_pl[-20:]

fig, (ax_bot, ax_top) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Phase 3 Enhanced: Per-Label F1 — Top 20 & Bottom 20 Categories",
             fontsize=13, fontweight="bold")

# Bottom 20
bot_labels = [x["label"] for x in bottom_20]
bot_f1s    = [x["f1"]    for x in bottom_20]
bot_supp   = [x["support"] for x in bottom_20]
colors_bot = ["#f03e3e" if f < 0.3 else "#ff8c00" for f in bot_f1s]
bars = ax_bot.barh(range(len(bottom_20)), bot_f1s, color=colors_bot,
                   edgecolor="white", linewidth=0.7)
for i, (bar, f, s) in enumerate(zip(bars, bot_f1s, bot_supp)):
    ax_bot.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{f:.3f} (n={s})", va="center", fontsize=8)
ax_bot.set_yticks(range(len(bottom_20)))
ax_bot.set_yticklabels(bot_labels, fontsize=9)
ax_bot.set_xlim(0, 0.65)
ax_bot.set_xlabel("F1 Score", fontsize=11)
ax_bot.set_title("Bottom 20 (worst labels)", fontsize=11, color="#f03e3e")
ax_bot.axvline(0.5, color="#666", linestyle="--", alpha=0.5, linewidth=1, label="F1=0.5")
ax_bot.legend(fontsize=9)
ax_bot.grid(axis="x", alpha=0.3, linestyle="--")

# Top 20
top_labels = [x["label"] for x in reversed(top_20)]
top_f1s    = [x["f1"]    for x in reversed(top_20)]
top_supp   = [x["support"] for x in reversed(top_20)]
colors_top = ["#2f9e44" if f > 0.8 else "#74c69d" for f in top_f1s]
bars = ax_top.barh(range(len(top_20)), top_f1s, color=colors_top,
                   edgecolor="white", linewidth=0.7)
for i, (bar, f, s) in enumerate(zip(bars, top_f1s, top_supp)):
    ax_top.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{f:.3f} (n={s})", va="center", fontsize=8)
ax_top.set_yticks(range(len(top_20)))
ax_top.set_yticklabels(top_labels, fontsize=9)
ax_top.set_xlim(0, 1.1)
ax_top.set_xlabel("F1 Score", fontsize=11)
ax_top.set_title("Top 20 (best labels)", fontsize=11, color="#2f9e44")
ax_top.axvline(0.8, color="#666", linestyle="--", alpha=0.5, linewidth=1, label="F1=0.8")
ax_top.legend(fontsize=9)
ax_top.grid(axis="x", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(OUT / "phase3_top_bottom_labels.png", dpi=150, bbox_inches="tight")
plt.close()
print("  DONE")

# ── 5. F1 Distribution Histogram ─────────────────────────────────────────────
print("Generating: phase3_f1_histogram.png")
all_f1_enh  = [x["f1"] for x in enh_data["per_label_f1"]]
all_f1_base = [x["f1"] for x in base_data["per_label_f1"]]

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, 1, 21)
ax.hist(all_f1_base, bins=bins, alpha=0.6, color="#4dabf7", label="DistilBERT Baseline",
        edgecolor="white", linewidth=0.5)
ax.hist(all_f1_enh,  bins=bins, alpha=0.6, color="#2f9e44", label="BERT-base Enhanced",
        edgecolor="white", linewidth=0.5)
ax.axvline(np.mean(all_f1_base), color="#4dabf7", linestyle="--", linewidth=2,
           label=f"Baseline mean = {np.mean(all_f1_base):.3f}")
ax.axvline(np.mean(all_f1_enh),  color="#2f9e44", linestyle="--", linewidth=2,
           label=f"Enhanced mean = {np.mean(all_f1_enh):.3f}")
ax.set_xlabel("Per-Label F1 Score", fontsize=12)
ax.set_ylabel("Number of Categories", fontsize=12)
ax.set_title("Distribution of Per-Label F1 Across 85 Categories\n"
             "Baseline (DistilBERT) vs Enhanced (BERT-base)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(OUT / "phase3_f1_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("  DONE")

print(f"\nAll plots saved to: {OUT}")
