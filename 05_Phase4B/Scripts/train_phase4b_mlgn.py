"""
Phase 4: MLGN — Multi-Label Guided Network for BGG Category Classification
Architecture based on Liu et al. 2023 [12].

Two additions over the Phase 3 BERT-BR Enhanced baseline:
  [A] Label Semantic Guidance (contrastive loss): pulls games sharing labels
      closer together in BERT embedding space.
  [B] Label Correlation Module (GCN): propagates information between
      semantically related labels using a co-occurrence graph.

Comparison target: Phase 3 Enhanced BERT-BR
  Micro-F1 = 0.6736  |  Macro-F1 = 0.6046  |  LRAP = 0.7769

Usage (HPC):
  python train_phase4b_mlgn.py
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
WORK_DIR        = Path("/cfs/home/u037341/tese/Tese-Marco/05_Phase4B/Results")
DATA_DIR        = Path("/cfs/home/u037341/tese/Tese-Marco/Dataset/XML Dataset")
PROGRESS_LOG    = WORK_DIR / "phase4b_progress.txt"

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

# MLGN-specific
GCN_HIDDEN      = 768          # same as BERT hidden size
GCN_LAYERS      = 2            # number of GCN layers
COOC_TOP_K      = 50           # max edges per label in co-occurrence graph
COOC_MIN_COUNT  = 5            # minimum co-occurrence count to include an edge
CONTRASTIVE_T          = 0.07   # temperature for contrastive loss
CONTRASTIVE_LAM_TARGET = 0.1   # target Con/BCE ratio (adaptive scaling)
# LAMBDA_MODE: "adaptive" = scale lambda so Con/BCE == target each step
#              "fixed"    = original Phase 4 behaviour (lam=0.1 fixed)
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
df = pd.read_parquet(DATA_DIR / "bgg_category_subset.parquet")
empty_mask = df["description_clean"].str.strip() == ""
df = df[~empty_mask].reset_index(drop=True)
log(f"Clean dataset: {df.shape}")

# ── SECTION 1: Encoding & Stratified Split ─────────────────────────────────────
log("SECTION 1: Multi-Label Encoding & Stratified Split")
from skmultilearn.model_selection import iterative_train_test_split

mlb          = MultiLabelBinarizer()
y_multilabel = mlb.fit_transform(df["categories_list"])
label_names  = mlb.classes_
num_labels   = len(label_names)
log(f"Labels: {num_labels}  |  Avg/game: {y_multilabel.sum(axis=1).mean():.2f}")

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
    Strategy: PMI-weighted, top-k edges per label, symmetric, self-loops included.
    Returns: adj_norm [num_labels, num_labels] torch tensor (normalised for GCN)
    """
    L = y_train.shape[1]
    # Raw co-occurrence counts
    cooc = y_train.T.astype(np.float32) @ y_train.astype(np.float32)  # [L, L]
    freq = y_train.sum(axis=0).astype(np.float32)                      # [L]
    N    = float(len(y_train))

    # PMI: log(P(i,j) / P(i)*P(j)), clipped to [0, inf)
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

    # Top-k edges per label (keep strongest co-occurrences)
    adj = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        row = pmi[i].copy()
        row[i] = 0  # no self-loop from PMI
        k = min(top_k, int((row > 0).sum()))
        if k > 0:
            top_idx = np.argpartition(row, -k)[-k:]
            adj[i, top_idx] = row[top_idx]

    # Symmetrise
    adj = np.maximum(adj, adj.T)

    # Add self-loops
    adj += np.eye(L, dtype=np.float32)

    # D^{-1/2} A D^{-1/2} normalisation
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]

    num_edges = int((adj > 0).sum() - L)  # subtract self-loops
    log(f"  Co-occurrence graph: {num_edges} edges, "
        f"avg degree {(adj > 0).sum(axis=1).mean() - 1:.1f} (excl. self-loops)")

    # Validate: print top-5 neighbors of 'Card Game' (most frequent)
    card_idx = list(label_names).index("Card Game") if "Card Game" in label_names else 0
    top_neighbors = np.argsort(adj[card_idx])[::-1][1:6]
    log(f"  Top-5 neighbors of '{label_names[card_idx]}': "
        f"{[label_names[j] for j in top_neighbors]}")

    return torch.tensor(adj_norm, dtype=torch.float32)

adj_norm = build_label_graph(y_train, label_names).to(device)

# --- 3b. Label name embeddings via BERT ---
log("  3b. Encoding label names with BERT to initialise label embeddings")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

def encode_label_names(label_names, tokenizer, bert_model, device, batch_size=16):
    """Encode each label name into a BERT [CLS] embedding."""
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
            embeddings = out.last_hidden_state[:, 0]  # [CLS]
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)  # [num_labels, hidden_size]

_bert_for_labels = AutoModel.from_pretrained(BERT_MODEL).to(device)
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
    """
    Graph Convolutional Network over the label co-occurrence graph.
    Input:  label_embeds [L, d], adj_norm [L, L]
    Output: refined label embeddings [L, d]
    """
    def __init__(self, in_features: int, hidden_features: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        dims   = [in_features] + [hidden_features] * num_layers
        self.layers  = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = torch.mm(adj, layer(x))          # graph convolution
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return self.layernorm(x)                  # normalise final output


class MLGNSupConLoss(nn.Module):
    """
    Supervised contrastive loss for multi-label setting (label semantic guidance).
    Games that share at least one label are treated as positives.
    Reference: Liu et al. 2023 [12], adapted from SupConLoss (Khosla et al. 2020).
    """
    def __init__(self, temperature: float = CONTRASTIVE_T):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = embeddings.size(0)
        z = F.normalize(embeddings, dim=1)                   # [B, d] unit vectors

        # Similarity matrix
        sim = torch.mm(z, z.T) / self.temperature            # [B, B]

        # Positive mask: pairs sharing at least one label (excluding self)
        label_overlap = torch.mm(labels, labels.T) > 0      # [B, B]
        self_mask     = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        pos_mask      = label_overlap & self_mask            # [B, B]

        # Numerical stability
        sim_max, _  = sim.max(dim=1, keepdim=True)
        sim         = sim - sim_max.detach()

        # Log-sum-exp over all other samples (denominator)
        exp_sim     = torch.exp(sim) * self_mask             # [B, B], zero diagonal
        log_denom   = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # [B, 1]

        # Log-prob of each positive pair
        log_prob    = sim - log_denom                        # [B, B]

        # Mean log-prob over positives per anchor
        num_pos     = pos_mask.float().sum(dim=1)            # [B]
        loss_per    = -(pos_mask.float() * log_prob).sum(dim=1) / (num_pos + 1e-8)

        # Only include anchors that have at least one positive
        valid       = num_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return loss_per[valid].mean()


class MLGNModel(nn.Module):
    """
    Multi-Label Guided Network (Liu et al. 2023 [12]).

    Components:
      1. BERT encoder   → game embedding g [B, d]
      2. Label GCN      → refined label embeddings h' [L, d]
      3. Classification → logits = g @ h'.T [B, L]

    The contrastive loss is computed externally using the game embeddings g.
    """
    def __init__(self, num_labels: int, label_init_embeds: torch.Tensor,
                 dropout_rate: float = DROPOUT):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(BERT_MODEL)
        hidden          = self.bert.config.hidden_size        # 768

        # Game encoder head
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.dropout        = nn.Dropout(dropout_rate)
        self.relu           = nn.ReLU()

        # Label correlation module (GCN)
        self.label_embeds = nn.Parameter(label_init_embeds)  # [L, 768], learnable
        self.gcn          = LabelGCN(hidden, GCN_HIDDEN, GCN_LAYERS, dropout=0.1)

        # Final projection (optional — keeps game/label in same space)
        self.game_proj  = nn.Linear(hidden, GCN_HIDDEN)
        self.label_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits      [B, L] — raw scores for BCE loss
            game_embeds [B, d] — normalised embeddings for contrastive loss
        """
        # 1. BERT → game embedding
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled   = bert_out.last_hidden_state[:, 0]           # [B, 768]
        pooled   = self.relu(self.pre_classifier(pooled))
        pooled   = self.dropout(pooled)                       # [B, 768]

        # 2. GCN → refined label embeddings
        h_prime  = self.gcn(self.label_embeds, adj)           # [L, 768]

        # 3. Classification via dot product
        g        = self.game_proj(pooled)                     # [B, 768]
        logits   = torch.mm(g, h_prime.T) + self.label_bias  # [B, L]

        return logits, g


model = MLGNModel(num_labels=num_labels, label_init_embeds=label_init_embeds).to(device)
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"MLGN | Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

# ── SECTION 6: Training ────────────────────────────────────────────────────────
log("SECTION 6: Training MLGN")

pos_counts  = y_train.sum(axis=0).astype(float)
neg_counts  = float(len(y_train)) - pos_counts
pos_weight  = torch.tensor(
    np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, POS_WEIGHT_CAP),
    dtype=torch.float32
).to(device)
log(f"pos_weight range (capped at {POS_WEIGHT_CAP}): "
    f"[{pos_weight.min().item():.2f}, {pos_weight.max().item():.2f}]")

bce_criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
supcon_loss_fn = MLGNSupConLoss(temperature=CONTRASTIVE_T)

optimizer    = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
log(f"Scheduler: warmup={warmup_steps} steps  |  total={total_steps} steps")
log(f"Contrastive mode={LAMBDA_MODE}  target_ratio={CONTRASTIVE_LAM_TARGET}  temperature={CONTRASTIVE_T}")

best_val_loss    = float("inf")
patience_counter = 0
history = {
    "train_loss": [], "train_bce_loss": [], "train_con_loss": [], "train_lam_eff": [],
    "val_loss": [], "val_micro_f1": [], "val_macro_f1": []
}
MODEL_SAVE = WORK_DIR.parent / "Models" / "best_mlgn4b_categories.pt"
MODEL_SAVE.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    # --- Train ---
    model.train()
    total_bce, total_con, total_loss_sum, total_lam = 0.0, 0.0, 0.0, 0.0

    for batch in train_loader:
        iids  = batch["input_ids"].to(device)
        amask = batch["attention_mask"].to(device)
        labs  = batch["labels"].to(device)

        labs_smooth = labs * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

        logits, game_embeds = model(iids, amask, adj_norm)

        bce_loss = bce_criterion(logits, labs_smooth)
        con_loss = supcon_loss_fn(game_embeds, labs)

        # Adaptive lambda: ensures Con/BCE ratio == CONTRASTIVE_LAM_TARGET
        # regardless of the raw scale difference between the two losses.
        if LAMBDA_MODE == "adaptive":
            lambda_eff = CONTRASTIVE_LAM_TARGET * (
                bce_loss.detach() / (con_loss.detach() + 1e-8)
            )
        else:  # original fixed lambda (Phase 4 behaviour)
            lambda_eff = CONTRASTIVE_LAM_TARGET

        loss = bce_loss + lambda_eff * con_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_bce      += bce_loss.item()
        total_con      += con_loss.item()
        total_loss_sum += loss.item()
        total_lam      += lambda_eff.item() if hasattr(lambda_eff, 'item') else lambda_eff

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

    con_ratio = avg_con / (avg_bce + 1e-8)
    effective_ratio = avg_lam * avg_con / (avg_bce + 1e-8)
    log(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
        f"BCE={avg_bce:.4f}  Con={avg_con:.4f} (raw_ratio={con_ratio:.2f})  "
        f"lam_eff={avg_lam:.4f} (eff_ratio={effective_ratio:.2f})  "
        f"Total={avg_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
        f"MicroF1={val_micro:.4f}  MacroF1={val_macro:.4f}"
        f"{'  -> BEST' if is_best else f'  ({patience_counter}/{PATIENCE})'}")

    if patience_counter >= PATIENCE:
        log(f"Early stopping at epoch {epoch+1}")
        break

log(f"Training done. Best Val Loss: {best_val_loss:.4f}")
model.load_state_dict(torch.load(MODEL_SAVE))

# ── SECTION 7: Threshold Optimisation (on val) ────────────────────────────────
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

log(f"Per-label thresh range: [{per_label_thresholds.min():.3f}, "
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

log(f"MLGN (global thresh={best_global_thresh:.3f}) | "
    f"Micro={res_global['micro_f1']:.4f}  Macro={res_global['macro_f1']:.4f}  "
    f"LRAP={res_global['lrap']:.4f}")
log(f"MLGN (per-label thresh) | "
    f"Micro={res_perlabel['micro_f1']:.4f}  Macro={res_perlabel['macro_f1']:.4f}  "
    f"LRAP={res_perlabel['lrap']:.4f}")

# Comparison with Phase 3 Enhanced baseline (primary) and Phase 4 MLGN (ablation reference)
BASELINE_MICRO     = 0.6736  # Phase 3 Enhanced
BASELINE_MACRO     = 0.6046
BASELINE_LRAP      = 0.7769
PHASE4_MICRO       = 0.6103  # Phase 4 MLGN fixed-lam
PHASE4_MACRO       = 0.4628
PHASE4_LRAP        = 0.7063
delta_micro = res_perlabel['micro_f1'] - BASELINE_MICRO
delta_macro = res_perlabel['macro_f1'] - BASELINE_MACRO
delta_lrap  = res_perlabel['lrap']     - BASELINE_LRAP
delta4_micro = res_perlabel['micro_f1'] - PHASE4_MICRO
delta4_macro = res_perlabel['macro_f1'] - PHASE4_MACRO
delta4_lrap  = res_perlabel['lrap']     - PHASE4_LRAP
log(f"vs Phase 3 Enhanced: "
    f"ΔMicro={delta_micro:+.4f}  ΔMacro={delta_macro:+.4f}  ΔLRAP={delta_lrap:+.4f}")
log(f"vs Phase 4 MLGN (fixed lam): "
    f"ΔMicro={delta4_micro:+.4f}  ΔMacro={delta4_macro:+.4f}  ΔLRAP={delta4_lrap:+.4f}")

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

# ── SECTION 9: McNemar Test vs Phase 3 Enhanced ─────────────────────────────
log("SECTION 9: Statistical Significance (McNemar Test)")
# NOTE: McNemar requires per-example predictions from both models.
# The Phase 3 Enhanced test predictions are not saved to disk in this run,
# so we log a reminder to run this comparison separately using saved predictions.
# Formula: load phase3_enhanced_test_preds.npy + mlgn4b_test_preds.npy → mcnemar_exact()
log("  Reminder: save test_preds_perlabel for McNemar comparison with Phase 3 Enhanced and Phase 4")
np.save(WORK_DIR / "mlgn4b_test_preds_perlabel.npy", test_preds_perlabel)
np.save(WORK_DIR / "mlgn4b_test_probs.npy",          test_probs_all)
np.save(WORK_DIR / "mlgn4b_test_labels.npy",         test_labels_all)
log("  Saved: mlgn4b_test_preds_perlabel.npy, mlgn4b_test_probs.npy, mlgn4b_test_labels.npy")

# ── SECTION 10: Report & JSON ─────────────────────────────────────────────────
log("SECTION 10: Writing report and results JSON")

report_path = WORK_DIR / "phase4b_mlgn_report.txt"
with open(report_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("PHASE 4B: MLGN ADAPTIVE LAMBDA — CATEGORY CLASSIFICATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"MODEL: MLGN (BERT-base + GCN + Contrastive Loss)\n")
    f.write(f"Reference: Liu et al. 2023 [12]\n\n")
    f.write(f"MLGN CONFIG:\n")
    f.write(f"  GCN layers:         {GCN_LAYERS}\n")
    f.write(f"  GCN hidden dim:     {GCN_HIDDEN}\n")
    f.write(f"  Cooc top-k:         {COOC_TOP_K}\n")
    f.write(f"  Cooc min-count:     {COOC_MIN_COUNT}\n")
    f.write(f"  Contrastive lambda mode:   {LAMBDA_MODE}\n")
    f.write(f"  Contrastive target ratio:  {CONTRASTIVE_LAM_TARGET}\n")
    f.write(f"  Contrastive temp:   {CONTRASTIVE_T}\n\n")
    f.write(f"BERT CONFIG (same as Phase 3 Enhanced):\n")
    f.write(f"  Model:       {BERT_MODEL}\n")
    f.write(f"  LR:          {LR}\n")
    f.write(f"  Batch size:  {BATCH_SIZE}\n")
    f.write(f"  Max len:     {MAX_LEN}\n")
    f.write(f"  Dropout:     {DROPOUT}\n")
    f.write(f"  pos_weight_cap: {POS_WEIGHT_CAP}\n")
    f.write(f"  Label smooth:   {LABEL_SMOOTHING}\n\n")
    f.write("RESULTS (Test Set):\n")
    f.write(f"  {'Model':<40} {'Micro-F1':>10} {'Macro-F1':>10} {'LRAP':>10} {'Hamming':>10}\n")
    f.write(f"  {'-'*80}\n")
    f.write(f"  {'TF-IDF BR':<40} {tfidf_results['micro_f1']:>10.4f} {tfidf_results['macro_f1']:>10.4f} {tfidf_results['lrap']:>10.4f} {tfidf_results['hamming']:>10.4f}\n")
    f.write(f"  {'Phase 3 Enhanced (BERT-BR, per-label)':<40} {'0.6736':>10} {'0.6046':>10} {'0.7769':>10} {'0.0244':>10}\n")
    f.write(f"  {'Phase 4 MLGN (fixed lam, per-label)':<40} {'0.6103':>10} {'0.4628':>10} {'0.7063':>10} {'0.0272':>10}\n")
    f.write(f"  {'MLGN 4B (global thresh)':<40} {res_global['micro_f1']:>10.4f} {res_global['macro_f1']:>10.4f} {res_global['lrap']:>10.4f} {res_global['hamming']:>10.4f}\n")
    f.write(f"  {'MLGN 4B (per-label thresh)':<40} {res_perlabel['micro_f1']:>10.4f} {res_perlabel['macro_f1']:>10.4f} {res_perlabel['lrap']:>10.4f} {res_perlabel['hamming']:>10.4f}\n\n")
    f.write(f"DELTA vs Phase 3 Enhanced (per-label):\n")
    f.write(f"  ΔMicro-F1 vs Phase3 = {delta_micro:+.4f}\n")
    f.write(f"  ΔMacro-F1 vs Phase3 = {delta_macro:+.4f}\n")
    f.write(f"  ΔLRAP     vs Phase3 = {delta_lrap:+.4f}\n")
    f.write(f"  ΔMicro-F1 vs Phase4 = {delta4_micro:+.4f}\n")
    f.write(f"  ΔMacro-F1 vs Phase4 = {delta4_macro:+.4f}\n")
    f.write(f"  ΔLRAP     vs Phase4 = {delta4_lrap:+.4f}\n\n")
    f.write(f"PER-LABEL ANALYSIS:\n")
    f.write(f"  Labels with F1 > 0.5: {labels_gt05}/{num_labels}\n")
    f.write(f"  Labels with F1 = 0.0: {labels_eq0}/{num_labels}\n")
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
        f.write(f"  Epoch {i+1:2d}: "
                f"TrainLoss={tl:.4f} (BCE={bce:.4f} Con={con:.4f} lam_eff={lam:.5f})  "
                f"ValLoss={vl:.4f}  MicroF1={mf:.4f}  MacroF1={maf:.4f}\n")
    f.write("\n" + "=" * 80 + "\n")

# JSON
results_json = {
    "model":  "MLGN (BERT-base + GCN + Contrastive)",
    "tfidf":  tfidf_results,
    "mlgn_global_thresh":   {**res_global,   "threshold": float(best_global_thresh)},
    "mlgn_perlabel_thresh": {**res_perlabel, "thresholds": per_label_thresholds.tolist()},
    "phase3_enhanced_baseline": {
        "micro_f1": BASELINE_MICRO, "macro_f1": BASELINE_MACRO, "lrap": BASELINE_LRAP
    },
    "phase4_mlgn_baseline": {
        "micro_f1": PHASE4_MICRO, "macro_f1": PHASE4_MACRO, "lrap": PHASE4_LRAP
    },
    "delta_vs_baseline": {
        "micro_f1": float(delta_micro), "macro_f1": float(delta_macro), "lrap": float(delta_lrap)
    },
    "per_label_f1": per_label_rows,
    "rq3_spearman": {"r": float(corr), "p": float(pval)},
    "history": history,
    "num_labels":   num_labels,
    "dataset_size": len(df),
    "config": {
        "bert_model":        BERT_MODEL,
        "gcn_layers":        GCN_LAYERS,
        "gcn_hidden":        GCN_HIDDEN,
        "cooc_top_k":        COOC_TOP_K,
        "cooc_min_count":    COOC_MIN_COUNT,
        "contrastive_temp":  CONTRASTIVE_T,
        "contrastive_lam_mode":   LAMBDA_MODE,
        "contrastive_lam_target": CONTRASTIVE_LAM_TARGET,
        "batch_size":        BATCH_SIZE,
        "max_len":           MAX_LEN,
        "num_epochs":        NUM_EPOCHS,
        "patience":          PATIENCE,
        "lr":                LR,
        "warmup_ratio":      WARMUP_RATIO,
        "dropout":           DROPOUT,
        "pos_weight_cap":    POS_WEIGHT_CAP,
        "label_smoothing":   LABEL_SMOOTHING,
        "early_stop_on":     "val_loss",
        "seed":              SEED,
    }
}
with open(WORK_DIR / "phase4b_mlgn_results.json", "w") as f:
    json.dump(results_json, f, indent=2)

log("Phase 4B MLGN COMPLETE")
