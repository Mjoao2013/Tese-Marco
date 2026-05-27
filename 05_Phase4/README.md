# Phase 4: MLGN — Multi-Label Guided Network

## Table of Contents
- [Overview](#overview)
- [Research Questions](#research-questions)
- [Architecture](#architecture)
- [MLGN vs BERT-BR: Key Differences](#mlgn-vs-bert-br-key-differences)
- [Label Co-occurrence Graph](#label-co-occurrence-graph)
- [Contrastive Loss (Label Semantic Guidance)](#contrastive-loss-label-semantic-guidance)
- [Configuration](#configuration)
- [Baseline to Beat](#baseline-to-beat)
- [Experimental Diagnostics](#experimental-diagnostics)
- [Files in This Folder](#files-in-this-folder)
- [Running on HPC](#running-on-hpc)
- [Glossary](#glossary)

---

## Overview

Phase 4 implements **MLGN (Multi-Label Guided Network)** from Liu et al. 2023 [12] for BGG category classification (85 labels, 165,708 games). This is the thesis's primary contribution: extending the Phase 3 BERT-BR Enhanced baseline with explicit label dependency modelling.

**Task:** Multi-label category classification  
**Labels:** 85 categories (e.g., Card Game, Wargame, Fantasy)  
**Baseline to beat:** Phase 3 Enhanced BERT-BR  — Micro-F1 = 0.6736 | Macro-F1 = 0.6046

---

## Research Questions

| RQ | Question | How Phase 4 Answers It |
|---|---|---|
| **RQ2** | Does explicit label correlation modelling (MLGN) outperform Binary Relevance (BERT-BR)? | Compare MLGN vs Phase 3 Enhanced; McNemar test p < 0.05 |
| **RQ3** | How do text characteristics affect performance? | Spearman r(support, F1); per-label delta analysis |

---

## Architecture

`
Game Description Text
        │
        ▼
  BERT-base-uncased
  (BERT encoder, shared weights)
        │
        ▼
 [CLS] pooling → Linear → ReLU → Dropout
        │
        ├──────────────────────────► Contrastive Loss [A]
        │                            (SupCon on game embeddings)
        │
  Game Embedding g [B × 768]
        │
        │   Label Names (85 names)
        │          │
        │       BERT encode
        │          │
        │   Label Embeddings h [85 × 768]  (learnable)
        │          │
        │       Label GCN [B]
        │     (adj from co-occurrence)
        │          │
        │   Refined h' [85 × 768]
        │
        ▼
  Logits = g · h'ᵀ + bias  [B × 85]
        │
        ▼
   BCEWithLogitsLoss [B]
`

Total loss = BCE + λ × SupConLoss  (λ = 0.1)

---

## MLGN vs BERT-BR: Key Differences

| Component | Phase 3 BERT-BR | Phase 4 MLGN |
|---|---|---|
| Text encoder | BERT → [CLS] → Linear(768→85) | BERT → [CLS] → Linear → g |
| Label representation | Fixed weight matrix | GCN-refined label embeddings h' |
| Classification | logits = Linear(g) | logits = g · h'ᵀ |
| Label dependencies | None (independent BCE per label) | GCN propagates co-occurrence info |
| Training signal | BCE only | BCE + λ × contrastive |
| Label graph | — | PMI-weighted top-50 co-occurrence edges |

---

## Label Co-occurrence Graph

Built from training data using **Pointwise Mutual Information (PMI)**:

`
PMI(i,j) = log[ P(i,j) / (P(i) × P(j)) ]
`

- Only edges with ≥ 5 co-occurrences are included
- Top-50 edges per label (adaptive: fewer if label has <50 co-occurring partners)
- Symmetric adjacency matrix + self-loops
- Normalised: D^{-1/2} A D^{-1/2} (standard GCN normalisation)

**Diagnostic check printed at startup:**  
Top-5 neighbours of "Card Game" — expected: "Collectible Components", "Dice", "Party Game", etc. (Martoglia & Pontiroli 2021 [1])

---

## Contrastive Loss (Label Semantic Guidance)

Supervised contrastive loss at the game level:

- **Positives:** pairs of games that share at least one category label
- **Negatives:** pairs with no shared labels (within the same batch)
- **Temperature:** τ = 0.07
- **Effect:** pulls together game embeddings with similar label profiles; pushes apart games from different categories

`
L_con = mean over anchors[ -log( Σ_pos exp(sim/τ) / Σ_all exp(sim/τ) ) ]
`

**Monitoring during training:**  
`
Con/BCE ratio logged each epoch — target: 10–30% (i.e., ratio 0.10–0.30)
If ratio > 0.50 → reduce λ (0.1 → 0.05)
If ratio < 0.05 → increase λ (0.1 → 0.2)
`

---

## Configuration

| Parameter | Value | Justification |
|---|---|---|
| Base model | bert-base-uncased | Consistent with Phase 3 Enhanced |
| Batch size | 32 | GPU memory constraint (A100 80GB) |
| Max length | 256 | Covers 95%+ of descriptions |
| Learning rate | 2e-5 | Best LR from Phase 3 grid search |
| Epochs | 15 | Same as Phase 3; early stop on val loss |
| Patience | 5 | Same as Phase 3 |
| Dropout | 0.3 | FIX 4 from Phase 3 Enhanced |
| pos_weight cap | 10.0 | FIX 1 from Phase 3 Enhanced |
| Label smoothing | 0.1 | FIX 3 from Phase 3 Enhanced |
| GCN layers | 2 | Standard for label GCN (Liu et al. 2023) |
| GCN hidden dim | 768 | Same as BERT hidden size |
| Cooc top-k | 50 | Liu et al. 2023 default |
| Cooc min count | 5 | Excludes spurious co-occurrences |
| Contrastive τ | 0.07 | Standard SupCon temperature |
| Contrastive λ | 0.1 | 10% contrastive weight |
| Seed | 42 | Reproducibility |

---

## Baseline to Beat

| Model | Micro-F1 | Macro-F1 | LRAP | Hamming |
|---|---|---|---|---|
| TF-IDF BR | 0.5693 | 0.3904 | 0.7557 | 0.0219 |
| DistilBERT Baseline (Phase 3 Iter 1) | 0.6781 | 0.5760 | 0.7923 | 0.0203 |
| BERT-base Optimized (Phase 3 Iter 2) | 0.6484 | 0.5895 | 0.7709 | 0.0278 |
| **BERT-BR Enhanced (Phase 3 Iter 3)** | **0.6736** | **0.6046** | **0.7769** | **0.0244** |
| MLGN (Phase 4) | *TBD* | *TBD* | *TBD* | *TBD* |

**Success criteria (RQ2):**
- Micro-F1 > 0.6736 OR Macro-F1 > 0.6046 (any improvement)
- McNemar test p < 0.05 (statistically significant)
- Per-label analysis: highly correlated label pairs improve by ≥ 2%

---

## Experimental Diagnostics

### If MLGN beats BERT-BR
- Run McNemar test on per-example predictions
- Compute per-label F1 delta: which labels improve with GCN?
- Check: do labels with many co-occurrence edges improve most?
- Hypothesis supported: GCN adds value beyond BERT's self-attention

### If MLGN does NOT beat BERT-BR
This is still a valid, publishable result. Investigate:
1. **GCN not learning?** Compare label_embeds norm before vs after training (should differ by > 0.05)
2. **Contrastive dominates?** Check Con/BCE ratio each epoch (should be 10–30%)
3. **BERT already captures dependencies?** BERT's self-attention may encode label co-occurrences implicitly (Azarbonyad et al. 2018 [11])
4. **Data constraint?** 85-label space with 792× imbalance may not have enough signal for explicit graph modelling

Conclusion: *"On BGG categories with 85 labels and 792× imbalance, MLGN's label correlation module does not significantly outperform a well-regularised BERT-BR baseline, suggesting BERT's self-attention already captures implicit label co-occurrences at this scale."*

---

## Files in This Folder

| File | Description |
|---|---|
| Scripts/train_phase4_mlgn.py | Full MLGN implementation — BERT + GCN + contrastive loss |
| deploy_phase4.sh | Shell script to run on HPC server (g08.hlt.inesc-id.pt) |
| Results/phase4_progress.txt | Timestamped training log (generated at runtime) |
| Results/phase4_run.log | Full stdout from server (generated at runtime) |
| Results/phase4_mlgn_report.txt | Human-readable summary of all results |
| Results/phase4_mlgn_results.json | Machine-readable metrics + config + per-label F1 |
| Results/mlgn_test_preds_perlabel.npy | Test predictions (for McNemar test) |
| Results/mlgn_test_probs.npy | Test probabilities (for threshold analysis) |
| Results/mlgn_test_labels.npy | Ground-truth test labels |
| Results/Images/ | PNG visualizations (generated post-training) |
| Models/best_mlgn_categories.pt | Best MLGN checkpoint — gitignored, local only |

---

## Running on HPC

`ash
# 1. SSH to server
ssh u037341@g08.hlt.inesc-id.pt

# 2. Pull latest code
cd /cfs/home/u037341/tese/Tese-Marco
git pull origin main

# 3. Run deployment script
bash 05_Phase4/deploy_phase4.sh

# 4. Monitor (in a new SSH session)
tail -f 05_Phase4/Results/phase4_run.log
cat 05_Phase4/Results/phase4_progress.txt
`

---

## Glossary

**Adjacency matrix** — A square matrix [L × L] representing the label co-occurrence graph. Entry (i,j) = PMI weight of the co-occurrence between labels i and j. After normalisation, this becomes the matrix passed to the GCN at each layer.

**BCEWithLogitsLoss** — Binary Cross-Entropy loss with sigmoid applied internally. Used as the primary classification loss. Applied independently to each of the 85 output labels.

**Binary Relevance (BR)** — Multi-label strategy from Phase 3: one independent sigmoid per label. No label interaction. MLGN improves on this by adding explicit GCN-based label correlation.

**Contrastive loss** — Loss that pulls similar examples together and pushes dissimilar ones apart in embedding space. Here: games sharing categories are positives; games with no shared categories are negatives.

**D^{-1/2} A D^{-1/2} normalisation** — Standard GCN normalisation (Kipf & Welling 2017). D = degree matrix. Prevents label embeddings from exploding when labels have many neighbours.

**GCN (Graph Convolutional Network)** — Neural network layer: new_embedding = relu(A_norm × W × old_embedding). Applied to the label graph. Propagates information from each label to its co-occurring neighbours.

**Label embedding** — A dense vector representation of a label (e.g., "Card Game" → [0.12, -0.34, ...]). Initialised from BERT encoding of the label name. Updated by the GCN and by gradient descent during training.

**LRAP (Label Ranking Average Precision)** — Measures whether correct labels are ranked higher than incorrect ones by the model's probability scores. Threshold-independent. Ranges 0–1.

**Macro-F1** — Per-label F1 averaged equally across all 85 labels. Treats a label with 6 examples the same as one with 6,657. Crucial for evaluating rare-label performance.

**McNemar test** — Statistical test comparing two classifiers on the same test set. Based on the disagreement pattern: examples model A gets right that B gets wrong (and vice versa). p < 0.05 = significant difference.

**Micro-F1** — F1 computed by pooling all label predictions globally. Dominated by frequent labels. Good for overall comparison.

**MLGN (Multi-Label Guided Network)** — Liu et al. 2023 [12]. Extends BERT multi-label classification with: (1) label semantic guidance via contrastive loss, (2) label correlation module via GCN on co-occurrence graph.

**PMI (Pointwise Mutual Information)** — log[P(i,j) / (P(i)×P(j))]. Positive = labels co-occur more than expected by chance. Negative = they tend to avoid each other. Used to weight graph edges.

**pos_weight** — Per-label weight in BCEWithLogitsLoss that upweights false negatives for rare labels. Capped at 10.0 to prevent gradient explosion (lesson from Phase 3 Optimized run).

**Self-loop** — Adding the identity matrix to the adjacency (A = A + I). Ensures each label also attends to itself during GCN propagation. Standard practice.

**Sigmoid** — Maps logits to (0, 1) probabilities. Applied per-label, independently. Threshold t: predict label present if sigmoid(logit) > t.

**SupCon (Supervised Contrastive Loss)** — Contrastive loss where positivity is defined by shared class label rather than augmentation. Adapted here for multi-label: positives = games sharing at least one category.

**Temperature (τ)** — Controls sharpness of contrastive distributions. Lower τ = harder push between positives and negatives. τ = 0.07 is standard (Khosla et al. 2020).