"""Phase 3: Multi-Label Category Classification (DistilBERT Binary Relevance)"""
import os, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, hamming_loss, label_ranking_average_precision_score
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
WORK_DIR = Path("/cfs/home/u037341/tese/Tese-Marco/Notebooks Tese")
DATA_DIR = Path("/cfs/home/u037341/tese/Tese-Marco/Dataset/XML Dataset")
PROGRESS_LOG = WORK_DIR / "phase3_progress.txt"

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")

# ── Reproducibility & Device ─────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")
if device.type == "cuda":
    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ── SECTION 0: Data Loading ───────────────────────────────────────────────────
log("SECTION 0: Data Loading started")
df = pd.read_parquet(DATA_DIR / "bgg_category_subset.parquet")
log(f"Raw dataset: {df.shape}")

empty_mask = df["description_clean"].str.strip() == ""
log(f"Empty descriptions: {empty_mask.sum()} (dropping)")
df = df[~empty_mask].reset_index(drop=True)
log(f"Clean dataset: {df.shape}")

# ── SECTION 1: Multi-Label Encoding & Split ───────────────────────────────────
log("SECTION 1: Multi-Label Encoding & Splitting started")

mlb = MultiLabelBinarizer()
y_multilabel = mlb.fit_transform(df["categories_list"])
label_names = mlb.classes_
num_labels = len(label_names)

log(f"Multi-hot matrix: {y_multilabel.shape}")
log(f"Num labels: {num_labels}")
log(f"Avg labels/game: {y_multilabel.sum(axis=1).mean():.2f}")

X_data = df[["id", "name", "description_clean"]].values
y_data = y_multilabel

try:
    from skmultilearn.model_selection import iterative_train_test_split
    USE_ITERATIVE = True
    log("Using iterative_train_test_split (skmultilearn)")
except ImportError:
    USE_ITERATIVE = False
    log("WARNING: skmultilearn not available, falling back to sklearn split")

if USE_ITERATIVE:
    X_train, y_train, X_temp, y_temp = iterative_train_test_split(X_data, y_data, test_size=0.30)
    X_val, y_val, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.50)
else:
    indices = np.arange(len(X_data))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED)
    X_train, y_train = X_data[train_idx], y_data[train_idx]
    X_val, y_val = X_data[val_idx], y_data[val_idx]
    X_test, y_test = X_data[test_idx], y_data[test_idx]

X_train_df = pd.DataFrame(X_train, columns=["id", "name", "description_clean"])
X_val_df   = pd.DataFrame(X_val,   columns=["id", "name", "description_clean"])
X_test_df  = pd.DataFrame(X_test,  columns=["id", "name", "description_clean"])

log(f"Train: {len(X_train_df):,}  Val: {len(X_val_df):,}  Test: {len(X_test_df):,}")

# ── SECTION 2: TF-IDF Baseline ────────────────────────────────────────────────
log("SECTION 2: TF-IDF Binary Relevance Baseline started")

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train_df["description_clean"].values)
X_val_tfidf   = vectorizer.transform(X_val_df["description_clean"].values)
X_test_tfidf  = vectorizer.transform(X_test_df["description_clean"].values)

ovr_model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, C=1.0, random_state=SEED), n_jobs=-1
)
ovr_model.fit(X_train_tfidf, y_train)

y_test_pred_tfidf  = ovr_model.predict(X_test_tfidf)
y_test_proba_tfidf = ovr_model.predict_proba(X_test_tfidf)

tfidf_results = {
    "micro_f1":   float(f1_score(y_test, y_test_pred_tfidf, average="micro", zero_division=0)),
    "macro_f1":   float(f1_score(y_test, y_test_pred_tfidf, average="macro", zero_division=0)),
    "hamming":    float(hamming_loss(y_test, y_test_pred_tfidf)),
    "subset_acc": float((y_test == y_test_pred_tfidf).all(axis=1).mean()),
    "lrap":       float(label_ranking_average_precision_score(y_test, y_test_proba_tfidf)),
}
log(f"TF-IDF Test Micro-F1: {tfidf_results['micro_f1']:.4f}  Macro-F1: {tfidf_results['macro_f1']:.4f}")

# ── SECTION 3: BERT Multi-Label Dataset & Model ───────────────────────────────
log("SECTION 3: BERT DataLoaders & Model setup started")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts     = list(texts)
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc  = self.tokenizer(text, max_length=self.max_len, padding="max_length",
                              truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float32),
        }

BATCH_SIZE = 32
train_dataset = MultiLabelDataset(X_train_df["description_clean"].values, y_train, tokenizer)
val_dataset   = MultiLabelDataset(X_val_df["description_clean"].values,   y_val,   tokenizer)
test_dataset  = MultiLabelDataset(X_test_df["description_clean"].values,  y_test,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=(device.type == "cuda"))
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          num_workers=4, pin_memory=(device.type == "cuda"))
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          num_workers=4, pin_memory=(device.type == "cuda"))

log(f"DataLoaders: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)} batches")

class DistilBERTMultiLabel(nn.Module):
    def __init__(self, num_labels, dropout_rate=0.2):
        super().__init__()
        self.distilbert     = AutoModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(self.distilbert.config.hidden_size,
                                        self.distilbert.config.hidden_size)
        self.dropout        = nn.Dropout(dropout_rate)
        self.classifier     = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        self.relu           = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        out    = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.relu(self.pre_classifier(pooled))
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

model = DistilBERTMultiLabel(num_labels=num_labels).to(device)
log(f"Model: DistilBERT Multi-Label | {sum(p.numel() for p in model.parameters()):,} params | {num_labels} labels")

# ── SECTION 4: Training ───────────────────────────────────────────────────────
log("SECTION 4: Training started")

NUM_EPOCHS = 10
PATIENCE   = 3

criterion  = nn.BCEWithLogitsLoss(reduction="mean")
optimizer  = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler  = CosineAnnealingLR(optimizer, T_max=len(train_loader) * NUM_EPOCHS, eta_min=1e-6)

best_val_micro_f1 = 0
patience_counter  = 0
history = {"train_loss": [], "val_loss": [], "val_micro_f1": [], "val_macro_f1": []}
THRESHOLD = 0.5

MODEL_SAVE_PATH = WORK_DIR / "best_bert_br_categories.pt"

for epoch in range(NUM_EPOCHS):
    # --- Train ---
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    history["train_loss"].append(avg_train_loss)

    # --- Validate ---
    model.eval()
    total_val_loss = 0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            logits         = model(input_ids, attention_mask)
            total_val_loss += criterion(logits, labels).item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_val_loss = total_val_loss / len(val_loader)
    history["val_loss"].append(avg_val_loss)

    all_logits = torch.cat(all_logits).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-all_logits))   # sigmoid
    preds = (probs > THRESHOLD).astype(int)

    val_micro_f1 = f1_score(all_labels_np, preds, average="micro", zero_division=0)
    val_macro_f1 = f1_score(all_labels_np, preds, average="macro", zero_division=0)
    history["val_micro_f1"].append(val_micro_f1)
    history["val_macro_f1"].append(val_macro_f1)

    is_best = val_micro_f1 > best_val_micro_f1
    if is_best:
        best_val_micro_f1 = val_micro_f1
        patience_counter  = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    else:
        patience_counter += 1

    log(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | TrainLoss={avg_train_loss:.4f} "
        f"ValLoss={avg_val_loss:.4f} | MicroF1={val_micro_f1:.4f} MacroF1={val_macro_f1:.4f}"
        f"{'  -> BEST' if is_best else f'  ({patience_counter}/{PATIENCE})'}")

    if patience_counter >= PATIENCE:
        log(f"Early stopping at epoch {epoch+1}")
        break

log(f"Training done. Best Val Micro-F1: {best_val_micro_f1:.4f}")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# ── SECTION 5: Threshold Optimization ────────────────────────────────────────
log("SECTION 5: Threshold Optimization started")

model.eval()
val_logits_list, val_labels_list = [], []
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        val_logits_list.append(logits.cpu())
        val_labels_list.append(batch["labels"])

val_probs_all  = torch.sigmoid(torch.cat(val_logits_list)).numpy()
val_labels_all = torch.cat(val_labels_list).numpy()

best_threshold   = 0.5
best_thresh_micro = 0.0
for t in np.arange(0.1, 0.70, 0.025):
    preds = (val_probs_all > t).astype(int)
    micro = f1_score(val_labels_all, preds, average="micro", zero_division=0)
    if micro > best_thresh_micro:
        best_thresh_micro = micro
        best_threshold    = t

log(f"Optimal threshold: {best_threshold:.3f}  (Val Micro-F1={best_thresh_micro:.4f})")

# ── SECTION 6: Final Test Evaluation ─────────────────────────────────────────
log("SECTION 6: Final Test Evaluation started")

model.eval()
test_logits_list, test_labels_list = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        test_logits_list.append(logits.cpu())
        test_labels_list.append(batch["labels"])

test_probs_all  = torch.sigmoid(torch.cat(test_logits_list)).numpy()
test_labels_all = torch.cat(test_labels_list).numpy()
test_preds_opt  = (test_probs_all > best_threshold).astype(int)

test_micro_f1  = float(f1_score(test_labels_all, test_preds_opt, average="micro", zero_division=0))
test_macro_f1  = float(f1_score(test_labels_all, test_preds_opt, average="macro", zero_division=0))
test_hamming   = float(hamming_loss(test_labels_all, test_preds_opt))
test_subset_acc= float((test_labels_all == test_preds_opt).all(axis=1).mean())
test_lrap      = float(label_ranking_average_precision_score(test_labels_all, test_probs_all))

log(f"BERT-BR Test | Micro-F1={test_micro_f1:.4f} Macro-F1={test_macro_f1:.4f} "
    f"LRAP={test_lrap:.4f} Hamming={test_hamming:.4f}")

# Per-label F1
per_label_f1 = []
for i, label in enumerate(label_names):
    f1 = float(f1_score(test_labels_all[:, i], test_preds_opt[:, i], zero_division=0))
    support = int(test_labels_all[:, i].sum())
    per_label_f1.append({"label": label, "f1": f1, "support": support})

per_label_f1.sort(key=lambda x: x["f1"], reverse=True)
labels_gt0   = sum(1 for x in per_label_f1 if x["f1"] > 0)
labels_gt05  = sum(1 for x in per_label_f1 if x["f1"] > 0.5)
labels_eq0   = sum(1 for x in per_label_f1 if x["f1"] == 0)
log(f"Per-label: F1>0.5={labels_gt05}/{num_labels}  F1=0={labels_eq0}/{num_labels}")

# ── Final Report ──────────────────────────────────────────────────────────────
log("Writing final report...")

report_path = WORK_DIR / "phase3_categories_report.txt"
with open(report_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("PHASE 3: CATEGORY CLASSIFICATION - FINAL REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"TASK: Multi-Label Classification ({num_labels} categories)\n")
    f.write(f"DATASET: {len(df):,} games, avg {y_multilabel.sum(axis=1).mean():.2f} labels/game\n\n")
    f.write("RESULTS (Test Set):\n")
    f.write(f"  {'Model':<20} {'Micro-F1':>10} {'Macro-F1':>10} {'LRAP':>10} {'Hamming':>10}\n")
    f.write(f"  {'-'*60}\n")
    f.write(f"  {'TF-IDF BR':<20} {tfidf_results['micro_f1']:>10.4f} {tfidf_results['macro_f1']:>10.4f} {tfidf_results['lrap']:>10.4f} {tfidf_results['hamming']:>10.4f}\n")
    f.write(f"  {'BERT-BR':<20} {test_micro_f1:>10.4f} {test_macro_f1:>10.4f} {test_lrap:>10.4f} {test_hamming:>10.4f}\n\n")
    f.write(f"THRESHOLD: {best_threshold:.3f} (optimized on validation set)\n\n")
    f.write(f"PER-LABEL ANALYSIS:\n")
    f.write(f"  Labels with F1 > 0.5: {labels_gt05}/{num_labels}\n")
    f.write(f"  Labels with F1 = 0.0: {labels_eq0}/{num_labels}\n\n")
    f.write("TOP 10 LABELS:\n")
    for r in per_label_f1[:10]:
        f.write(f"  {r['label']:<30} F1={r['f1']:.4f}  support={r['support']}\n")
    f.write("\nBOTTOM 10 LABELS:\n")
    for r in per_label_f1[-10:]:
        f.write(f"  {r['label']:<30} F1={r['f1']:.4f}  support={r['support']}\n")
    f.write("\n" + "=" * 80 + "\n")

# Save JSON results
results_json = {
    "tfidf":     tfidf_results,
    "bert_br": {
        "micro_f1":  test_micro_f1,
        "macro_f1":  test_macro_f1,
        "hamming":   test_hamming,
        "subset_acc":test_subset_acc,
        "lrap":      test_lrap,
        "threshold": float(best_threshold),
    },
    "per_label_f1": per_label_f1,
    "num_labels":   num_labels,
    "dataset_size": len(df),
}
with open(WORK_DIR / "phase3_categories_results.json", "w") as f:
    json.dump(results_json, f, indent=2)

log("Phase 3 COMPLETE")
log(f"Report saved to: {report_path}")
