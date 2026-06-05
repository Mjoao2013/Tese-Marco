"""
Phase 5B: Focal Loss Ablation for Mechanism Classification
===========================================================
Identical to Phase 5A (BERT-BR) except BCEWithLogitsLoss + pos_weight is
replaced with Focal Loss (Lin et al. 2017, γ=2).

Hypothesis: with 195 labels and ~4,000x imbalance (far worse than categories'
792x), Focal Loss may outperform capped pos_weight BCE by more naturally
suppressing the large volume of easy negatives that dominate the gradient.

Single-variable change vs Phase 5A:
  - REMOVED: BCEWithLogitsLoss(pos_weight=...), pos_weight computation
  - ADDED:   FocalLoss(gamma=2.0)
  - UNCHANGED: everything else (model, LR, dropout, smoothing, splits, ...)

If 5B > 5A: focal loss is preferable for the extreme imbalance regime.
If 5B ≈ 5A: capped BCE is sufficient; focal adds no value.
If 5B < 5A: capped BCE handles mechanisms well despite the higher imbalance.
"""
import os, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, hamming_loss, label_ranking_average_precision_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
WORK_DIR        = Path("/cfs/home/u037341/tese/Tese-Marco/06_Phase5/Results")
DATA_DIR        = Path("/cfs/home/u037341/tese/Tese-Marco/Dataset/XML Dataset")
PROGRESS_LOG    = WORK_DIR / "phase5b_progress.txt"

SEED            = 42
BERT_MODEL      = "bert-base-uncased"
BATCH_SIZE      = 32
MAX_LEN         = 256
NUM_EPOCHS      = 15
PATIENCE        = 5
LR              = 2e-5
WARMUP_RATIO    = 0.1
DROPOUT         = 0.3
LABEL_SMOOTHING = 0.1           # kept from Phase 5A (compatible with focal loss)

# Focal Loss specific (replaces pos_weight in Phase 5A)
FOCAL_GAMMA     = 2.0           # standard; down-weights easy negatives quadratically
# No POS_WEIGHT_CAP needed — focal loss handles imbalance through gamma

# ── Logging ────────────────────────────────────────────────────────────────────
WORK_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts   = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")

open(PROGRESS_LOG, "w").close()

# ── Seed & Device ──────────────────────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")
if device.type == "cuda":
    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ── SECTION 0: Data Loading ────────────────────────────────────────────────────
log("SECTION 0: Data Loading")
df = pd.read_parquet(DATA_DIR / "bgg_mechanism_subset.parquet")
empty_mask = df["description_clean"].str.strip() == ""
df = df[~empty_mask].reset_index(drop=True)
log(f"Clean dataset: {df.shape}")

# ── SECTION 1: Encoding & Stratified Split ─────────────────────────────────────
log("SECTION 1: Multi-Label Encoding & Stratified Split")
from skmultilearn.model_selection import iterative_train_test_split

mlb          = MultiLabelBinarizer()
y_multilabel = mlb.fit_transform(df["mechanisms_list"])
label_names  = mlb.classes_
num_labels   = len(label_names)
log(f"Labels: {num_labels}  |  Avg/game: {y_multilabel.sum(axis=1).mean():.2f}")

label_counts = y_multilabel.sum(axis=0)
imbalance_ratio = label_counts.max() / max(label_counts.min(), 1)
log(f"Imbalance: max_support={int(label_counts.max())}  "
    f"min_support={int(label_counts.min())}  ratio={imbalance_ratio:.0f}x")
log(f"Labels with < 100 examples: {(label_counts < 100).sum()}/{num_labels}")

X_data = df[["id", "name", "description_clean"]].values
y_data = y_multilabel

X_train, y_train, X_temp, y_temp = iterative_train_test_split(X_data, y_data, test_size=0.30)
X_val,   y_val,   X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.50)

X_train_df = pd.DataFrame(X_train, columns=["id", "name", "description_clean"])
X_val_df   = pd.DataFrame(X_val,   columns=["id", "name", "description_clean"])
X_test_df  = pd.DataFrame(X_test,  columns=["id", "name", "description_clean"])

log(f"Train: {len(X_train_df):,}  Val: {len(X_val_df):,}  Test: {len(X_test_df):,}")

max_diff = max(
    np.abs(y_train.mean(axis=0) - y_val.mean(axis=0)).max(),
    np.abs(y_train.mean(axis=0) - y_test.mean(axis=0)).max()
)
log(f"Stratification check — max label freq diff: {max_diff:.4f} "
    f"({'PASSED' if max_diff < 0.02 else 'WARNING'})")

# ── SECTION 2: TF-IDF Baseline ─────────────────────────────────────────────────
log("SECTION 2: TF-IDF Binary Relevance Baseline")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
X_tr_tfidf = vectorizer.fit_transform(X_train_df["description_clean"].values)
X_te_tfidf = vectorizer.transform(X_test_df["description_clean"].values)

ovr = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, C=1.0, random_state=SEED), n_jobs=-1
)
ovr.fit(X_tr_tfidf, y_train)
y_te_pred_tfidf  = ovr.predict(X_te_tfidf)
y_te_proba_tfidf = ovr.predict_proba(X_te_tfidf)

tfidf_results = {
    "micro_f1":   float(f1_score(y_test, y_te_pred_tfidf, average="micro", zero_division=0)),
    "macro_f1":   float(f1_score(y_test, y_te_pred_tfidf, average="macro", zero_division=0)),
    "hamming":    float(hamming_loss(y_test, y_te_pred_tfidf)),
    "subset_acc": float((y_test == y_te_pred_tfidf).all(axis=1).mean()),
    "lrap":       float(label_ranking_average_precision_score(y_test, y_te_proba_tfidf)),
}
log(f"TF-IDF | Micro-F1={tfidf_results['micro_f1']:.4f}  Macro-F1={tfidf_results['macro_f1']:.4f}")

# ── SECTION 3: Focal Loss & Model ─────────────────────────────────────────────
log(f"SECTION 3: {BERT_MODEL} Model + Focal Loss (γ={FOCAL_GAMMA})")

class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for multi-label classification (Lin et al. 2017).
    Reduces the relative loss for well-classified examples, focusing training
    on hard, misclassified examples. Replaces BCE + pos_weight for extreme
    imbalance where most labels are absent for most samples.

    Formula: FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
    With α=None (no class weighting) and γ=2 (standard).
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Numerically stable sigmoid + focal factor
        p      = torch.sigmoid(logits)
        p_t    = torch.where(targets >= 0.5, p, 1.0 - p)   # p if target=1, 1-p if target=0
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal  = (1.0 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = list(texts)
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float32),
        }

train_dataset = MultiLabelDataset(X_train_df["description_clean"].values, y_train, tokenizer)
val_dataset   = MultiLabelDataset(X_val_df["description_clean"].values,   y_val,   tokenizer)
test_dataset  = MultiLabelDataset(X_test_df["description_clean"].values,  y_test,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
log(f"DataLoaders: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)} batches")

class BERTMultiLabel(nn.Module):
    def __init__(self, num_labels, dropout_rate=DROPOUT):
        super().__init__()
        self.bert           = AutoModel.from_pretrained(BERT_MODEL)
        self.pre_classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.dropout        = nn.Dropout(dropout_rate)
        self.classifier     = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.relu           = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.relu(self.pre_classifier(pooled))
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

model     = BERTMultiLabel(num_labels=num_labels).to(device)
criterion = FocalLoss(gamma=FOCAL_GAMMA, reduction="mean")
log(f"Model: {BERT_MODEL} | {sum(p.numel() for p in model.parameters()):,} params | "
    f"FocalLoss(γ={FOCAL_GAMMA})")

# ── SECTION 4: Training ────────────────────────────────────────────────────────
log("SECTION 4: Training — Focal Loss + label smoothing")

optimizer    = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
log(f"Scheduler: linear warmup ({warmup_steps} steps) → linear decay ({total_steps} total)")

# For validation loss we use the same focal criterion so curves are comparable
val_criterion = FocalLoss(gamma=FOCAL_GAMMA, reduction="mean")

best_val_loss    = float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_micro_f1": [], "val_macro_f1": []}
MODEL_SAVE = WORK_DIR.parent / "Models" / "best_focal_mechanisms_5b.pt"
MODEL_SAVE.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    # --- Train ---
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        iids  = batch["input_ids"].to(device)
        amask = batch["attention_mask"].to(device)
        labs  = batch["labels"].to(device)

        labs_smooth = labs * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

        logits = model(iids, amask)
        loss   = criterion(logits, labs_smooth)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    history["train_loss"].append(avg_train_loss)

    # --- Validate (hard labels) ---
    model.eval()
    total_val_loss = 0.0
    all_logits, all_labs = [], []
    with torch.no_grad():
        for batch in val_loader:
            iids  = batch["input_ids"].to(device)
            amask = batch["attention_mask"].to(device)
            labs  = batch["labels"].to(device)
            logits = model(iids, amask)
            total_val_loss += val_criterion(logits, labs).item()
            all_logits.append(logits.cpu())
            all_labs.append(labs.cpu())

    avg_val_loss  = total_val_loss / len(val_loader)
    all_probs     = torch.sigmoid(torch.cat(all_logits)).numpy()
    all_labels_np = torch.cat(all_labs).numpy()
    preds_05      = (all_probs > 0.5).astype(int)

    val_micro_f1 = f1_score(all_labels_np, preds_05, average="micro", zero_division=0)
    val_macro_f1 = f1_score(all_labels_np, preds_05, average="macro", zero_division=0)
    history["val_loss"].append(avg_val_loss)
    history["val_micro_f1"].append(val_micro_f1)
    history["val_macro_f1"].append(val_macro_f1)

    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss    = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE)
    else:
        patience_counter += 1

    log(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | TrainLoss={avg_train_loss:.4f} "
        f"ValLoss={avg_val_loss:.4f} | MicroF1={val_micro_f1:.4f} MacroF1={val_macro_f1:.4f}"
        f"{'  -> BEST' if is_best else f'  ({patience_counter}/{PATIENCE})'}")

    if patience_counter >= PATIENCE:
        log(f"Early stopping at epoch {epoch+1}")
        break

log(f"Training done. Best Val Loss: {best_val_loss:.4f}")
model.load_state_dict(torch.load(MODEL_SAVE))

# ── SECTION 5: Per-Label Threshold Optimisation ────────────────────────────────
log("SECTION 5: Per-Label Threshold Optimisation")

model.eval()
val_logits_list, val_labs_list = [], []
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        val_logits_list.append(logits.cpu())
        val_labs_list.append(batch["labels"])

val_probs_all  = torch.sigmoid(torch.cat(val_logits_list)).numpy()
val_labels_all = torch.cat(val_labs_list).numpy()

best_global_thresh = 0.5
best_global_micro  = 0.0
for t in np.arange(0.10, 0.90, 0.025):
    micro = f1_score(val_labels_all, (val_probs_all > t).astype(int),
                     average="micro", zero_division=0)
    if micro > best_global_micro:
        best_global_micro  = micro
        best_global_thresh = t

log(f"Global threshold: {best_global_thresh:.3f}  (Val Micro-F1={best_global_micro:.4f})")

per_label_thresholds = np.zeros(num_labels)
for i in range(num_labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.90, 0.025):
        f1 = f1_score(val_labels_all[:, i], (val_probs_all[:, i] > t).astype(int),
                      zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    per_label_thresholds[i] = best_t

log(f"Per-label thresh — range: [{per_label_thresholds.min():.3f}, "
    f"{per_label_thresholds.max():.3f}]  mean={per_label_thresholds.mean():.3f}")

# ── SECTION 6: Final Test Evaluation ──────────────────────────────────────────
log("SECTION 6: Final Test Evaluation")

model.eval()
test_logits_list, test_labs_list = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        test_logits_list.append(logits.cpu())
        test_labs_list.append(batch["labels"])

test_probs_all  = torch.sigmoid(torch.cat(test_logits_list)).numpy()
test_labels_all = torch.cat(test_labs_list).numpy()

test_preds_global   = (test_probs_all > best_global_thresh).astype(int)
test_preds_perlabel = (test_probs_all > per_label_thresholds[np.newaxis, :]).astype(int)

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "micro_f1":   float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1":   float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming":    float(hamming_loss(y_true, y_pred)),
        "subset_acc": float((y_true == y_pred).all(axis=1).mean()),
        "lrap":       float(label_ranking_average_precision_score(y_true, y_proba)),
    }

res_global   = compute_metrics(test_labels_all, test_preds_global,   test_probs_all)
res_perlabel = compute_metrics(test_labels_all, test_preds_perlabel, test_probs_all)

log(f"Focal 5B (global thresh={best_global_thresh:.3f}) | "
    f"Micro={res_global['micro_f1']:.4f}  Macro={res_global['macro_f1']:.4f}  "
    f"LRAP={res_global['lrap']:.4f}")
log(f"Focal 5B (per-label thresh) | "
    f"Micro={res_perlabel['micro_f1']:.4f}  Macro={res_perlabel['macro_f1']:.4f}  "
    f"LRAP={res_perlabel['lrap']:.4f}")

# Cross-task reference
PHASE3_MICRO = 0.6736
PHASE3_MACRO = 0.6046
PHASE3_LRAP  = 0.7769
delta_micro  = res_perlabel["micro_f1"] - PHASE3_MICRO
delta_macro  = res_perlabel["macro_f1"] - PHASE3_MACRO
delta_lrap   = res_perlabel["lrap"]     - PHASE3_LRAP
log(f"vs Phase 3 Enhanced (categories cross-task ref): "
    f"ΔMicro={delta_micro:+.4f}  ΔMacro={delta_macro:+.4f}  ΔLRAP={delta_lrap:+.4f}")

# Per-label analysis
per_label_rows = []
for i, label in enumerate(label_names):
    f1_val  = float(f1_score(test_labels_all[:, i], test_preds_perlabel[:, i], zero_division=0))
    support = int(test_labels_all[:, i].sum())
    thresh  = float(per_label_thresholds[i])
    per_label_rows.append({"label": label, "f1": f1_val, "support": support, "threshold": thresh})

per_label_rows.sort(key=lambda x: x["f1"], reverse=True)
labels_gt05 = sum(1 for x in per_label_rows if x["f1"] > 0.5)
labels_eq0  = sum(1 for x in per_label_rows if x["f1"] == 0)

corr, pval = spearmanr(
    [r["support"] for r in per_label_rows],
    [r["f1"]      for r in per_label_rows]
)
log(f"Per-label: F1>0.5={labels_gt05}/{num_labels}  F1=0={labels_eq0}/{num_labels}")
log(f"RQ3 Spearman r(F1, support)={corr:.3f}  p={pval:.4f}")

np.save(WORK_DIR / "phase5b_test_preds_perlabel.npy", test_preds_perlabel)
np.save(WORK_DIR / "phase5b_test_probs.npy",          test_probs_all)
np.save(WORK_DIR / "phase5b_test_labels.npy",         test_labels_all)
log("Saved: phase5b_test_preds_perlabel.npy, phase5b_test_probs.npy, phase5b_test_labels.npy")

# ── SECTION 7: Report & JSON ───────────────────────────────────────────────────
log("SECTION 7: Writing report and results JSON")

report_path = WORK_DIR / "phase5b_focal_report.txt"
with open(report_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("PHASE 5B: MECHANISM CLASSIFICATION — FOCAL LOSS ABLATION\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"MODEL: {BERT_MODEL} (Binary Relevance + FocalLoss γ={FOCAL_GAMMA})\n")
    f.write(f"Change vs Phase 5A: BCEWithLogitsLoss(pos_weight) → FocalLoss(γ={FOCAL_GAMMA})\n\n")
    f.write(f"CONFIG:\n")
    f.write(f"  LR: {LR}  |  Batch: {BATCH_SIZE}  |  MaxLen: {MAX_LEN}\n")
    f.write(f"  Dropout: {DROPOUT}  |  FocalLoss γ: {FOCAL_GAMMA}  |  "
            f"label_smooth: {LABEL_SMOOTHING}\n")
    f.write(f"  Epochs: {NUM_EPOCHS}  |  Patience: {PATIENCE}  |  Early stop: val_loss\n\n")
    f.write(f"DATASET: {len(df):,} games  |  {num_labels} labels  |  "
            f"Avg {y_multilabel.sum(axis=1).mean():.2f} labels/game\n")
    f.write(f"Imbalance: {imbalance_ratio:.0f}x  |  "
            f"Labels < 100 examples: {(label_counts < 100).sum()}/{num_labels}\n")
    f.write(f"Splits: Train={len(X_train_df):,}  Val={len(X_val_df):,}  "
            f"Test={len(X_test_df):,}\n\n")
    f.write("RESULTS (Test Set):\n")
    f.write(f"  {'Model':<40} {'Micro-F1':>10} {'Macro-F1':>10} {'LRAP':>10} {'Hamming':>10}\n")
    f.write(f"  {'-'*75}\n")
    f.write(f"  {'TF-IDF BR':<40} {tfidf_results['micro_f1']:>10.4f} "
            f"{tfidf_results['macro_f1']:>10.4f} {tfidf_results['lrap']:>10.4f} "
            f"{tfidf_results['hamming']:>10.4f}\n")
    f.write(f"  {'Focal 5B (global)':<40} {res_global['micro_f1']:>10.4f} "
            f"{res_global['macro_f1']:>10.4f} {res_global['lrap']:>10.4f} "
            f"{res_global['hamming']:>10.4f}\n")
    f.write(f"  {'Focal 5B (per-label)':<40} {res_perlabel['micro_f1']:>10.4f} "
            f"{res_perlabel['macro_f1']:>10.4f} {res_perlabel['lrap']:>10.4f} "
            f"{res_perlabel['hamming']:>10.4f}\n")
    f.write(f"\n[Cross-task ref] Phase 3 Enhanced (same config, categories):\n")
    f.write(f"  Micro-F1=0.6736  Macro-F1=0.6046  LRAP=0.7769\n")
    f.write(f"  ΔMicro={delta_micro:+.4f}  ΔMacro={delta_macro:+.4f}  "
            f"ΔLRAP={delta_lrap:+.4f}\n\n")
    f.write(f"THRESHOLDS:\n")
    f.write(f"  Global optimal:  {best_global_thresh:.3f}\n")
    f.write(f"  Per-label range: [{per_label_thresholds.min():.3f}, "
            f"{per_label_thresholds.max():.3f}]  mean={per_label_thresholds.mean():.3f}\n\n")
    f.write(f"PER-LABEL ANALYSIS:\n")
    f.write(f"  Labels F1 > 0.5: {labels_gt05}/{num_labels}\n")
    f.write(f"  Labels F1 = 0.0: {labels_eq0}/{num_labels}\n")
    f.write(f"  Spearman r(F1, support) = {corr:.3f}  (p={pval:.4f})\n\n")
    f.write("TOP 10 LABELS:\n")
    for r in per_label_rows[:10]:
        f.write(f"  {r['label']:<40} F1={r['f1']:.4f}  support={r['support']}  "
                f"thresh={r['threshold']:.3f}\n")
    f.write("\nBOTTOM 10 LABELS:\n")
    for r in per_label_rows[-10:]:
        f.write(f"  {r['label']:<40} F1={r['f1']:.4f}  support={r['support']}  "
                f"thresh={r['threshold']:.3f}\n")
    f.write("\nTRAINING HISTORY:\n")
    for i, (tl, vl, mf, maf) in enumerate(zip(
        history["train_loss"], history["val_loss"],
        history["val_micro_f1"], history["val_macro_f1"]
    )):
        f.write(f"  Epoch {i+1:2d}: TrainLoss={tl:.4f}  ValLoss={vl:.4f}  "
                f"MicroF1={mf:.4f}  MacroF1={maf:.4f}\n")
    f.write("\n" + "=" * 80 + "\n")

results_json = {
    "model":   BERT_MODEL,
    "phase":   "5B",
    "task":    "mechanisms",
    "loss":    f"FocalLoss(gamma={FOCAL_GAMMA})",
    "tfidf":   tfidf_results,
    "focal_global_thresh":   {**res_global,   "threshold": float(best_global_thresh)},
    "focal_perlabel_thresh": {**res_perlabel, "thresholds": per_label_thresholds.tolist()},
    "cross_task_ref_phase3": {
        "micro_f1": PHASE3_MICRO, "macro_f1": PHASE3_MACRO, "lrap": PHASE3_LRAP,
        "note": "Phase 3 Enhanced — BCE+pos_weight, categories, 85 labels"
    },
    "delta_vs_phase3": {
        "micro_f1": float(delta_micro), "macro_f1": float(delta_macro), "lrap": float(delta_lrap)
    },
    "per_label_f1":  per_label_rows,
    "rq3_spearman":  {"r": float(corr), "p": float(pval)},
    "history":       history,
    "num_labels":    num_labels,
    "dataset_size":  len(df),
    "imbalance_ratio": float(imbalance_ratio),
    "labels_lt100":  int((label_counts < 100).sum()),
    "config": {
        "bert_model":      BERT_MODEL,
        "batch_size":      BATCH_SIZE,
        "max_len":         MAX_LEN,
        "num_epochs":      NUM_EPOCHS,
        "patience":        PATIENCE,
        "lr":              LR,
        "warmup_ratio":    WARMUP_RATIO,
        "dropout":         DROPOUT,
        "focal_gamma":     FOCAL_GAMMA,
        "label_smoothing": LABEL_SMOOTHING,
        "early_stop_on":   "val_loss",
        "seed":            SEED,
    }
}
with open(WORK_DIR / "phase5b_focal_results.json", "w") as f:
    json.dump(results_json, f, indent=2)

log("Phase 5B Focal Loss COMPLETE")
log(f"Report: {report_path}")
