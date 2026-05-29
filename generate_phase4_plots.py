import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = "05_Phase4/Results/Images/"
os.makedirs(OUT, exist_ok=True)

with open("05_Phase4/Results/phase4_mlgn_results.json") as f:
    d = json.load(f)

labels = [e["label"]   for e in d["per_label_f1"]]
f1s    = [e["f1"]      for e in d["per_label_f1"]]
sups   = [e["support"] for e in d["per_label_f1"]]
hist   = d["history"]
epochs = list(range(1, 16))
p3 = {"micro_f1": 0.6736, "macro_f1": 0.6046, "lrap": 0.7769, "hamming": 0.0244}

# 1. Model Comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Phase 4: MLGN vs Baselines", fontsize=14, fontweight="bold")
models = ["TF-IDF BR", "Phase 3\nEnhanced", "MLGN\nglobal", "MLGN\nper-label"]
colors = ["#888888", "#2196F3", "#FF9800", "#F44336"]
metrics = [
    ("Micro-F1",  [d["tfidf"]["micro_f1"], p3["micro_f1"], d["mlgn_global_thresh"]["micro_f1"],  d["mlgn_perlabel_thresh"]["micro_f1"]]),
    ("Macro-F1",  [d["tfidf"]["macro_f1"], p3["macro_f1"], d["mlgn_global_thresh"]["macro_f1"],  d["mlgn_perlabel_thresh"]["macro_f1"]]),
    ("LRAP",      [d["tfidf"]["lrap"],     p3["lrap"],     d["mlgn_global_thresh"]["lrap"],       d["mlgn_perlabel_thresh"]["lrap"]]),
    ("Hamming\n(lower=better)", [d["tfidf"]["hamming"], p3["hamming"], d["mlgn_global_thresh"]["hamming"], d["mlgn_perlabel_thresh"]["hamming"]]),
]
for ax, (metric, vals) in zip(axes, metrics):
    bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(metric, fontweight="bold")
    ax.set_ylim(0, max(vals)*1.2)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT+"phase4_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("1/5 done")

# 2. Training Curves
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Phase 4 MLGN Training Dynamics (15 Epochs)", fontsize=14, fontweight="bold")
axes[0].plot(epochs, hist["train_bce_loss"], "b-o", ms=4, label="BCE Loss")
axes[0].plot(epochs, hist["train_con_loss"], "r-s", ms=4, label="Contrastive Loss")
axes[0].set_title("BCE vs Contrastive (train)", fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylabel("Loss")
axes[1].plot(epochs, hist["val_loss"], "g-o", ms=4, label="Val Loss")
axes[1].axvline(12, color="orange", linestyle="--", lw=1.5, label="Best epoch (12)")
axes[1].set_title("Validation Loss", fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylabel("Loss")
axes[2].plot(epochs, hist["val_micro_f1"], "b-o", ms=4, label="Val Micro-F1")
axes[2].plot(epochs, hist["val_macro_f1"], "r-s", ms=4, label="Val Macro-F1")
axes[2].axhline(0.6736, color="blue",  linestyle="--", lw=1, alpha=0.5, label="Phase3 Micro target")
axes[2].axhline(0.6046, color="red",   linestyle="--", lw=1, alpha=0.5, label="Phase3 Macro target")
axes[2].set_title("Val F1 vs Phase 3 targets", fontweight="bold")
axes[2].set_xlabel("Epoch"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3); axes[2].set_ylabel("F1")
plt.tight_layout()
plt.savefig(OUT+"phase4_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("2/5 done")

# 3. Contrastive Ratio
ratios = [c/b for c,b in zip(hist["train_con_loss"], hist["train_bce_loss"])]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, ratios, "r-o", ms=6, linewidth=2, label="Contrastive/BCE ratio")
ax.axhspan(0.1, 0.3, alpha=0.15, color="green", label="Healthy range (0.1-0.3)")
ax.axhline(0.1, color="green", linestyle="--", lw=1)
ax.axhline(0.3, color="green", linestyle="--", lw=1)
ax.set_xlabel("Epoch"); ax.set_ylabel("Contrastive / BCE ratio")
ax.set_title("Contrastive Loss Domination - Root Cause Analysis", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(0, 4.0)
ax.annotate("Ratio ~2.5-3.2\n(~25x above healthy)", xy=(1, ratios[0]), xytext=(4, 3.4),
            arrowprops=dict(arrowstyle="->", color="darkred"), fontsize=10, color="darkred")
plt.tight_layout()
plt.savefig(OUT+"phase4_contrastive_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("3/5 done")

# 4. Top/Bottom Labels
sorted_data = sorted(zip(labels, f1s, sups), key=lambda x: x[1])
n = 20
bot_labs, bot_f1s, bot_sup = zip(*sorted_data[:n])
top_labs, top_f1s, top_sup = zip(*sorted_data[-n:])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
cmap_top = plt.cm.Greens(np.linspace(0.4, 0.9, n))
cmap_bot = plt.cm.Reds(np.linspace(0.9, 0.4, n))
bars = ax1.barh(range(n), top_f1s[::-1], color=cmap_top)
ax1.set_yticks(range(n))
ax1.set_yticklabels([f"{l} ({s:,})" for l,s in zip(top_labs[::-1], top_sup[::-1])], fontsize=8)
ax1.set_xlabel("F1 Score"); ax1.set_title("Top 20 Labels by F1", fontweight="bold")
ax1.set_xlim(0, 1.0); ax1.grid(axis="x", alpha=0.3)
for bar, v in zip(bars, top_f1s[::-1]):
    ax1.text(min(bar.get_width()+0.01, 0.96), bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=7)
bars = ax2.barh(range(n), bot_f1s, color=cmap_bot)
ax2.set_yticks(range(n))
ax2.set_yticklabels([f"{l} ({s:,})" for l,s in zip(bot_labs, bot_sup)], fontsize=8)
ax2.set_xlabel("F1 Score"); ax2.set_title("Bottom 20 Labels by F1", fontweight="bold")
ax2.set_xlim(0, 1.0); ax2.grid(axis="x", alpha=0.3)
for bar, v in zip(bars, bot_f1s):
    ax2.text(max(bar.get_width()+0.01, 0.01), bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=7)
plt.suptitle("MLGN Per-Label F1 - Top & Bottom 20 (Categories Task)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT+"phase4_top_bottom_labels.png", dpi=150, bbox_inches="tight")
plt.close()
print("4/5 done")

# 5. Support vs F1 scatter
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(sups, f1s, c=f1s, cmap="RdYlGn", s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
plt.colorbar(sc, ax=ax, label="F1 Score")
ax.set_xscale("log")
ax.set_xlabel("Support (log scale)", fontsize=12); ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("MLGN: Per-Label F1 vs Support  (Spearman r=0.655, p<0.001)", fontsize=12, fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5, label="F1=0.5 threshold")
ax.legend(); ax.grid(alpha=0.2)
highlight = [(l, s, v) for l, v, s in zip(labels, f1s, sups) if v < 0.05 or v > 0.85 or s < 30]
for lab, s, v in highlight:
    ax.annotate(lab, (s, v), fontsize=6, alpha=0.75, xytext=(4,4), textcoords="offset points")
plt.tight_layout()
plt.savefig(OUT+"phase4_per_label_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("5/5 done")
print("ALL DONE")
