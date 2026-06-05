"""
Phase 5C: MLGN (Adaptive Lambda) for Mechanism Classification
==============================================================
Direct port of Phase 4B (best MLGN variant) to the mechanism task.

Key differences vs Phase 4B:
  - Dataset:       bgg_mechanism_subset.parquet  (was bgg_category_subset.parquet)
  - Label column:  mechanisms_list               (was categories_list)
  - COOC_MIN_COUNT raised from 5 → 10           (mechanisms have stronger imbalance;
                                                  rare pairs with < 10 co-occurrences
                                                  produce unreliable PMI edges)
  - GCN graph sanity check: uses most-frequent mechanism label (not 'Card Game')
  - All MLGN hyperparameters inherited from Phase 4B (adaptive λ, τ=0.07, etc.)

Hypothesis: Mechanisms have stronger pairwise correlations than categories
(e.g., Hand Management ↔ Deck Building, Dice Rolling ↔ Push Your Luck).
The GCN co-occurrence module may provide more genuine lift here than on categories.

Reference baselines (to be compared in Results):
  - Phase 5A BERT-BR:     to be filled after Phase 5A completes
  - Phase 5B Focal Loss:  to be filled after Phase 5B completes
  - Phase 4B MLGN (cats): Micro-F1=0.6267, Macro-F1=0.4934, LRAP=0.7182
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
PROGRESS_LOG    = WORK_DIR / "phase5c_progress.txt"

SEED            = 42
BERT_MODEL      = "bert-base-uncased"
BATCH_SIZE      = 32
MAX_LEN         = 256
NUM_EPOCHS      = 15
PATIENCE        = 5
LR              = 2e-5
WARMUP_RATIO    = 0.1
DROPOUT         = 0.3
POS_WEIGHT_CAP  = 10.0
LABEL_SMOOTHING = 0.1

# MLGN-specific (inherited from Phase 4B except COOC_MIN_COUNT)
GCN_HIDDEN             = 768
GCN_LAYERS             = 2
COOC_TOP_K             = 50
COOC_MIN_COUNT         = 10    # raised from 5 — mechanisms have more sparse rare pairs
CONTRASTIVE_T          = 0.07
CONTRASTIVE_LAM_TARGET = 0.1   # adaptive target ratio
LAMBDA_MODE            = "adaptive"

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

label_counts    = y_multilabel.sum(axis=0)
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

# ── SECTION 3: MLGN Components ────────────────────────────────────────────────
log("SECTION 3: Building MLGN components")

# --- 3a. Label co-occurrence graph ---
log("  3a. Building label co-occurrence graph")

def build_label_graph(y_train, label_names, top_k=COOC_TOP_K, min_count=COOC_MIN_COUNT):
    """
    Build a normalised adjacency matrix from label co-occurrence in training data.
    PMI-weighted, top-k edges per label, symmetric, self-loops included.
    COOC_MIN_COUNT=10 (raised from 5 in Phase 4B) to filter spurious rare-mechanism pairs.
    Returns: adj_norm [num_labels, num_labels] torch tensor
    """
    L    = y_train.shape[1]
    cooc = y_train.T.astype(np.float32) @ y_train.astype(np.float32)  # [L, L]
    freq = y_train.sum(axis=0).astype(np.float32)                      # [L]
    N    = float(len(y_train))

    pmi = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            if i == j or cooc[i, j] < min_count:
                continue
            p_ij = cooc[i, j] / N
            p_i  = freq[i]  / N
            p_j  = freq[j]  / N
            if p_i > 0 and p_j > 0:
                pmi[i, j] = max(0.0, float(np.log(p_ij / (p_i * p_j))))

    adj = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        row = pmi[i].copy()
        row[i] = 0
        k = min(top_k, int((row > 0).sum()))
        if k > 0:
            top_idx = np.argpartition(row, -k)[-k:]
            adj[i, top_idx] = row[top_idx]

    adj  = np.maximum(adj, adj.T)       # symmetrise
    adj += np.eye(L, dtype=np.float32)  # self-loops

    degree     = adj.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    adj_norm   = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]

    num_edges = int((adj > 0).sum() - L)
    log(f"  Co-occurrence graph: {num_edges} edges  "
        f"avg degree {(adj > 0).sum(axis=1).mean() - 1:.1f} (excl. self-loops)  "
        f"min_count={min_count}")

    # Sanity check: top-5 neighbours of most frequent mechanism
    top_label_idx = int(freq.argmax())
    top_neighbors = np.argsort(adj[top_label_idx])[::-1][1:6]
    log(f"  Top-5 neighbours of '{label_names[top_label_idx]}': "
        f"{[label_names[j] for j in top_neighbors]}")

    # Also check a well-known correlated pair
    for a, b in [("Hand Management", "Deck Construction"), ("Dice Rolling", "Push Your Luck")]:
        if a in label_names and b in label_names:
            ia, ib = list(label_names).index(a), list(label_names).index(b)
            log(f"  PMI({a!r}, {b!r}) = {pmi[ia, ib]:.4f}  "
                f"edge weight = {adj[ia, ib]:.4f}")

    return torch.tensor(adj_norm, dtype=torch.float32)

adj_norm = build_label_graph(y_train, label_names).to(device)

# --- 3b. Label name embeddings via BERT ---
log("  3b. Encoding label names with BERT to initialise label embeddings")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

def encode_label_names(label_names, tokenizer, bert_model, device, batch_size=16):
    bert_model.eval()
    all_embeddings = []
    names = list(label_names)
    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=32, return_tensors="pt")
        enc   = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = bert_model(**enc)
            embeddings = out.last_hidden_state[:, 0]
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

_bert_for_labels  = AutoModel.from_pretrained(BERT_MODEL).to(device)
label_init_embeds = encode_label_names(label_names, tokenizer, _bert_for_labels, device)
log(f"  Label embeddings initialised: {label_init_embeds.shape}")
del _bert_for_labels
torch.cuda.empty_cache()

# ── SECTION 4: Dataset & DataLoaders ─────────────────────────────────────────
log("SECTION 4: DataLoaders setup")

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

# ── SECTION 5: MLGN Model ─────────────────────────────────────────────────────
log("SECTION 5: MLGN Model definition")

class LabelGCN(nn.Module):
    def __init__(self, in_features, hidden_features, num_layers=2, dropout=0.1):
        super().__init__()
        dims         = [in_features] + [hidden_features] * num_layers
        self.layers  = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.dropout   = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_features)

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = torch.mm(adj, layer(x))
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return self.layernorm(x)


class MLGNSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for multi-label setting (Liu et al. 2023 [12]).
    Games sharing at least one mechanism label are treated as positives.
    """
    def __init__(self, temperature=CONTRASTIVE_T):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        B   = embeddings.size(0)
        z   = F.normalize(embeddings, dim=1)
        sim = torch.mm(z, z.T) / self.temperature

        label_overlap = torch.mm(labels, labels.T) > 0
        self_mask     = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        pos_mask      = label_overlap & self_mask

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()

        exp_sim   = torch.exp(sim) * self_mask
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        log_prob  = sim - log_denom

        num_pos  = pos_mask.float().sum(dim=1)
        loss_per = -(pos_mask.float() * log_prob).sum(dim=1) / (num_pos + 1e-8)

        valid = num_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return loss_per[valid].mean()


class MLGNModel(nn.Module):
    def __init__(self, num_labels, label_init_embeds, dropout_rate=DROPOUT):
        super().__init__()
        self.bert           = AutoModel.from_pretrained(BERT_MODEL)
        hidden              = self.bert.config.hidden_size  # 768
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.dropout        = nn.Dropout(dropout_rate)
        self.relu           = nn.ReLU()
        self.label_embeds   = nn.Parameter(label_init_embeds)
        self.gcn            = LabelGCN(hidden, GCN_HIDDEN, GCN_LAYERS, dropout=0.1)
        self.game_proj      = nn.Linear(hidden, GCN_HIDDEN)
        self.label_bias     = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids, attention_mask, adj):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled   = bert_out.last_hidden_state[:, 0]
        pooled   = self.relu(self.pre_classifier(pooled))
        pooled   = self.dropout(pooled)
        h_prime  = self.gcn(self.label_embeds, adj)
        g        = self.game_proj(pooled)
        logits   = torch.mm(g, h_prime.T) + self.label_bias
        return logits, g


model = MLGNModel(num_labels=num_labels, label_init_embeds=label_init_embeds).to(device)
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"MLGN | Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

# ── SECTION 6: Training ────────────────────────────────────────────────────────
log("SECTION 6: Training MLGN (adaptive lambda)")

pos_counts  = y_train.sum(axis=0).astype(float)
neg_counts  = float(len(y_train)) - pos_counts
pos_weight  = torch.tensor(
    np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, POS_WEIGHT_CAP),
    dtype=torch.float32
).to(device)
n_capped = int((neg_counts / np.maximum(pos_counts, 1) > POS_WEIGHT_CAP).sum())
log(f"pos_weight range (cap={POS_WEIGHT_CAP}): "
    f"[{pos_weight.min().item():.2f}, {pos_weight.max().item():.2f}]  "
    f"labels hitting cap: {n_capped}/{num_labels}")

bce_criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
supcon_loss_fn = MLGNSupConLoss(temperature=CONTRASTIVE_T)

optimizer    = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
log(f"Scheduler: warmup={warmup_steps} steps  |  total={total_steps} steps")
log(f"Contrastive: mode={LAMBDA_MODE}  target_ratio={CONTRASTIVE_LAM_TARGET}  τ={CONTRASTIVE_T}")

best_val_loss    = float("inf")
patience_counter = 0
history = {
    "train_loss": [], "train_bce_loss": [], "train_con_loss": [], "train_lam_eff": [],
    "val_loss": [], "val_micro_f1": [], "val_macro_f1": []
}
MODEL_SAVE = WORK_DIR.parent / "Models" / "best_mlgn_mechanisms_5c.pt"
MODEL_SAVE.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_bce, total_con, total_loss_sum, total_lam = 0.0, 0.0, 0.0, 0.0

    for batch in train_loader:
        iids  = batch["input_ids"].to(device)
        amask = batch["attention_mask"].to(device)
        labs  = batch["labels"].to(device)

        labs_smooth         = labs * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        logits, game_embeds = model(iids, amask, adj_norm)

        bce_loss = bce_criterion(logits, labs_smooth)
        con_loss = supcon_loss_fn(game_embeds, labs)

        # Adaptive lambda: locks Con/BCE ratio to CONTRASTIVE_LAM_TARGET every step
        lambda_eff = CONTRASTIVE_LAM_TARGET * (
            bce_loss.detach() / (con_loss.detach() + 1e-8)
        )
        loss = bce_loss + lambda_eff * con_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_bce      += bce_loss.item()
        total_con      += con_loss.item()
        total_loss_sum += loss.item()
        total_lam      += lambda_eff.item()

    avg_bce  = total_bce  / len(train_loader)
    avg_con  = total_con  / len(train_loader)
    avg_loss = total_loss_sum / len(train_loader)
    avg_lam  = total_lam  / len(train_loader)

    history["train_loss"].append(avg_loss)
    history["train_bce_loss"].append(avg_bce)
    history["train_con_loss"].append(avg_con)
    history["train_lam_eff"].append(avg_lam)

    # --- Validate ---
    model.eval()
    total_val_loss = 0.0
    all_logits_v, all_labs_v = [], []

    with torch.no_grad():
        for batch in val_loader:
            iids  = batch["input_ids"].to(device)
            amask = batch["attention_mask"].to(device)
            labs  = batch["labels"].to(device)
            logits, _ = model(iids, amask, adj_norm)
            total_val_loss += bce_criterion(logits, labs).item()
            all_logits_v.append(logits.cpu())
            all_labs_v.append(labs.cpu())

    avg_val_loss  = total_val_loss / len(val_loader)
    all_probs_v   = torch.sigmoid(torch.cat(all_logits_v)).numpy()
    all_labels_np = torch.cat(all_labs_v).numpy()
    preds_05      = (all_probs_v > 0.5).astype(int)

    val_micro = f1_score(all_labels_np, preds_05, average="micro", zero_division=0)
    val_macro = f1_score(all_labels_np, preds_05, average="macro", zero_division=0)

    history["val_loss"].append(avg_val_loss)
    history["val_micro_f1"].append(val_micro)
    history["val_macro_f1"].append(val_macro)

    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss    = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE)
    else:
        patience_counter += 1

    con_ratio      = avg_con / (avg_bce + 1e-8)
    effective_ratio = avg_lam * avg_con / (avg_bce + 1e-8)
    log(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
        f"BCE={avg_bce:.4f}  Con={avg_con:.4f} (raw_ratio={con_ratio:.2f})  "
        f"lam_eff={avg_lam:.5f} (eff_ratio={effective_ratio:.2f})  "
        f"Total={avg_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
        f"MicroF1={val_micro:.4f}  MacroF1={val_macro:.4f}"
        f"{'  -> BEST' if is_best else f'  ({patience_counter}/{PATIENCE})'}")

    if patience_counter >= PATIENCE:
        log(f"Early stopping at epoch {epoch+1}")
        break

log(f"Training done. Best Val Loss: {best_val_loss:.4f}")
model.load_state_dict(torch.load(MODEL_SAVE))

# ── SECTION 7: Per-Label Threshold Optimisation ────────────────────────────────
log("SECTION 7: Per-Label Threshold Optimisation")

model.eval()
val_logits_list, val_labs_list = [], []
with torch.no_grad():
    for batch in val_loader:
        logits, _ = model(batch["input_ids"].to(device),
                          batch["attention_mask"].to(device), adj_norm)
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

# ── SECTION 8: Final Test Evaluation ─────────────────────────────────────────
log("SECTION 8: Final Test Evaluation")

model.eval()
test_logits_list, test_labs_list = [], []
with torch.no_grad():
    for batch in test_loader:
        logits, _ = model(batch["input_ids"].to(device),
                          batch["attention_mask"].to(device), adj_norm)
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

log(f"MLGN 5C (global thresh={best_global_thresh:.3f}) | "
    f"Micro={res_global['micro_f1']:.4f}  Macro={res_global['macro_f1']:.4f}  "
    f"LRAP={res_global['lrap']:.4f}")
log(f"MLGN 5C (per-label thresh) | "
    f"Micro={res_perlabel['micro_f1']:.4f}  Macro={res_perlabel['macro_f1']:.4f}  "
    f"LRAP={res_perlabel['lrap']:.4f}")

# Cross-task references
PHASE4B_MICRO = 0.6267   # MLGN adaptive λ, categories
PHASE4B_MACRO = 0.4934
PHASE4B_LRAP  = 0.7182
delta_vs_4b_micro = res_perlabel["micro_f1"] - PHASE4B_MICRO
delta_vs_4b_macro = res_perlabel["macro_f1"] - PHASE4B_MACRO
delta_vs_4b_lrap  = res_perlabel["lrap"]     - PHASE4B_LRAP
log(f"vs Phase 4B MLGN (categories cross-task ref): "
    f"ΔMicro={delta_vs_4b_micro:+.4f}  ΔMacro={delta_vs_4b_macro:+.4f}  "
    f"ΔLRAP={delta_vs_4b_lrap:+.4f}")

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

# Save predictions for McNemar tests vs 5A and 5B
np.save(WORK_DIR / "phase5c_test_preds_perlabel.npy", test_preds_perlabel)
np.save(WORK_DIR / "phase5c_test_probs.npy",          test_probs_all)
np.save(WORK_DIR / "phase5c_test_labels.npy",         test_labels_all)
log("Saved: phase5c_test_preds_perlabel.npy, phase5c_test_probs.npy, phase5c_test_labels.npy")

# ── SECTION 9: Report & JSON ───────────────────────────────────────────────────
log("SECTION 9: Writing report and results JSON")

report_path = WORK_DIR / "phase5c_mlgn_report.txt"
with open(report_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("PHASE 5C: MECHANISM CLASSIFICATION — MLGN ADAPTIVE LAMBDA\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"MODEL: MLGN (BERT-base + GCN + Contrastive Loss, adaptive λ)\n")
    f.write(f"Reference: Liu et al. 2023 [12]\n\n")
    f.write(f"MLGN CONFIG:\n")
    f.write(f"  GCN layers: {GCN_LAYERS}  |  GCN hidden: {GCN_HIDDEN}\n")
    f.write(f"  Cooc top-k: {COOC_TOP_K}  |  Cooc min-count: {COOC_MIN_COUNT} "
            f"(raised from 5 in Phase 4B)\n")
    f.write(f"  Contrastive λ mode: {LAMBDA_MODE}  target_ratio: {CONTRASTIVE_LAM_TARGET}  "
            f"temp: {CONTRASTIVE_T}\n\n")
    f.write(f"BERT CONFIG:\n")
    f.write(f"  Model: {BERT_MODEL}  |  LR: {LR}  |  Batch: {BATCH_SIZE}  |  MaxLen: {MAX_LEN}\n")
    f.write(f"  Dropout: {DROPOUT}  |  pos_weight_cap: {POS_WEIGHT_CAP}  |  "
            f"label_smooth: {LABEL_SMOOTHING}\n\n")
    f.write(f"DATASET: {len(df):,} games  |  {num_labels} labels  |  "
            f"Avg {y_multilabel.sum(axis=1).mean():.2f} labels/game\n")
    f.write(f"Imbalance: {imbalance_ratio:.0f}x  |  "
            f"Labels < 100 examples: {(label_counts < 100).sum()}/{num_labels}\n")
    f.write(f"Splits: Train={len(X_train_df):,}  Val={len(X_val_df):,}  "
            f"Test={len(X_test_df):,}\n\n")
    f.write("RESULTS (Test Set):\n")
    f.write(f"  {'Model':<40} {'Micro-F1':>10} {'Macro-F1':>10} {'LRAP':>10} {'Hamming':>10}\n")
    f.write(f"  {'-'*80}\n")
    f.write(f"  {'TF-IDF BR':<40} {tfidf_results['micro_f1']:>10.4f} "
            f"{tfidf_results['macro_f1']:>10.4f} {tfidf_results['lrap']:>10.4f} "
            f"{tfidf_results['hamming']:>10.4f}\n")
    f.write(f"  {'Phase 5A BERT-BR (fill after 5A)':<40} {'TBD':>10} {'TBD':>10} "
            f"{'TBD':>10} {'TBD':>10}\n")
    f.write(f"  {'Phase 5B Focal (fill after 5B)':<40} {'TBD':>10} {'TBD':>10} "
            f"{'TBD':>10} {'TBD':>10}\n")
    f.write(f"  {'MLGN 5C (global thresh)':<40} {res_global['micro_f1']:>10.4f} "
            f"{res_global['macro_f1']:>10.4f} {res_global['lrap']:>10.4f} "
            f"{res_global['hamming']:>10.4f}\n")
    f.write(f"  {'MLGN 5C (per-label thresh)':<40} {res_perlabel['micro_f1']:>10.4f} "
            f"{res_perlabel['macro_f1']:>10.4f} {res_perlabel['lrap']:>10.4f} "
            f"{res_perlabel['hamming']:>10.4f}\n")
    f.write(f"\n[Cross-task ref] Phase 4B MLGN adaptive λ (categories, 85 labels):\n")
    f.write(f"  Micro-F1=0.6267  Macro-F1=0.4934  LRAP=0.7182\n")
    f.write(f"  ΔMicro={delta_vs_4b_micro:+.4f}  ΔMacro={delta_vs_4b_macro:+.4f}  "
            f"ΔLRAP={delta_vs_4b_lrap:+.4f}\n\n")
    f.write(f"THRESHOLDS:\n")
    f.write(f"  Global optimal: {best_global_thresh:.3f}\n")
    f.write(f"  Per-label range: [{per_label_thresholds.min():.3f}, "
            f"{per_label_thresholds.max():.3f}]  mean={per_label_thresholds.mean():.3f}\n\n")
    f.write(f"PER-LABEL ANALYSIS:\n")
    f.write(f"  Labels F1 > 0.5: {labels_gt05}/{num_labels}\n")
    f.write(f"  Labels F1 = 0.0: {labels_eq0}/{num_labels}\n")
    f.write(f"  Spearman r(F1, support) = {corr:.3f}  (p={pval:.4f})\n\n")
    f.write("TOP 10 LABELS:\n")
    for r in per_label_rows[:10]:
        f.write(f"  {r['label']:<40} F1={r['f1']:.4f}  support={r['support']}\n")
    f.write("\nBOTTOM 10 LABELS:\n")
    for r in per_label_rows[-10:]:
        f.write(f"  {r['label']:<40} F1={r['f1']:.4f}  support={r['support']}\n")
    f.write("\nTRAINING HISTORY:\n")
    for i, (tl, bce, con, lam, vl, mf, maf) in enumerate(zip(
        history["train_loss"], history["train_bce_loss"], history["train_con_loss"],
        history["train_lam_eff"],
        history["val_loss"], history["val_micro_f1"], history["val_macro_f1"]
    )):
        f.write(f"  Epoch {i+1:2d}: TrainLoss={tl:.4f} "
                f"(BCE={bce:.4f} Con={con:.4f} lam_eff={lam:.5f})  "
                f"ValLoss={vl:.4f}  MicroF1={mf:.4f}  MacroF1={maf:.4f}\n")
    f.write("\n" + "=" * 80 + "\n")

results_json = {
    "model":  "MLGN (BERT-base + GCN + Contrastive, adaptive λ)",
    "phase":  "5C",
    "task":   "mechanisms",
    "tfidf":  tfidf_results,
    "mlgn_global_thresh":   {**res_global,   "threshold": float(best_global_thresh)},
    "mlgn_perlabel_thresh": {**res_perlabel, "thresholds": per_label_thresholds.tolist()},
    "cross_task_ref_phase4b": {
        "micro_f1": PHASE4B_MICRO, "macro_f1": PHASE4B_MACRO, "lrap": PHASE4B_LRAP,
        "note": "Phase 4B MLGN adaptive λ — categories, 85 labels"
    },
    "delta_vs_phase4b": {
        "micro_f1": float(delta_vs_4b_micro),
        "macro_f1": float(delta_vs_4b_macro),
        "lrap":     float(delta_vs_4b_lrap),
    },
    "per_label_f1":    per_label_rows,
    "rq3_spearman":    {"r": float(corr), "p": float(pval)},
    "history":         history,
    "num_labels":      num_labels,
    "dataset_size":    len(df),
    "imbalance_ratio": float(imbalance_ratio),
    "labels_lt100":    int((label_counts < 100).sum()),
    "config": {
        "bert_model":            BERT_MODEL,
        "gcn_layers":            GCN_LAYERS,
        "gcn_hidden":            GCN_HIDDEN,
        "cooc_top_k":            COOC_TOP_K,
        "cooc_min_count":        COOC_MIN_COUNT,
        "contrastive_temp":      CONTRASTIVE_T,
        "contrastive_lam_mode":  LAMBDA_MODE,
        "contrastive_lam_target": CONTRASTIVE_LAM_TARGET,
        "batch_size":            BATCH_SIZE,
        "max_len":               MAX_LEN,
        "num_epochs":            NUM_EPOCHS,
        "patience":              PATIENCE,
        "lr":                    LR,
        "warmup_ratio":          WARMUP_RATIO,
        "dropout":               DROPOUT,
        "pos_weight_cap":        POS_WEIGHT_CAP,
        "label_smoothing":       LABEL_SMOOTHING,
        "early_stop_on":         "val_loss",
        "seed":                  SEED,
    }
}
with open(WORK_DIR / "phase5c_mlgn_results.json", "w") as f:
    json.dump(results_json, f, indent=2)

log("Phase 5C MLGN COMPLETE")
log(f"Report: {report_path}")
