# 02 — Phase 1: DistilBERT Baseline for Geek Type Classification

## Table of Contents
- [Overview](#overview)
- [Research Question Addressed](#research-question-addressed)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Comparison to Classical Baseline](#comparison-to-classical-baseline)
- [What Phase 1 Establishes](#what-phase-1-establishes)
- [Files in This Folder](#files-in-this-folder)

---

## Overview

Phase 1 establishes the first neural transfer learning baseline for the thesis. The task is single-label classification of board games into 8 geek type categories using only their text description. DistilBERT is chosen as the first model because it is lighter (66.4M vs BERT's 110M parameters), faster to iterate with, and well-documented in the literature as achieving ~97% of BERT's performance at 60% of the size (Sanh et al. 2019 [7]).

**Task:** Single-label classification (8 classes)  
**Model:** `distilbert-base-uncased`  
**Dataset:** 14,713 games with geek_type label, split 70/15/15  
**Notebook:** `Modelling_Phase01.ipynb`

---

## Research Question Addressed

> **RQ1:** Can text descriptions alone reliably identify a game's geek type?

A Micro-F1 above 0.60–0.70 on this task would confirm that descriptions carry enough discriminative signal for type classification, validating the text-only approach before scaling to multi-label tasks.

---

## Model Architecture

```
DistilBERT (distilbert-base-uncased)
  └─ 6 transformer layers, 768 hidden dim, 66.4M params
  └─ [CLS] token representation
  └─ Dropout (0.2)
  └─ Linear (768 → 8)
  └─ CrossEntropyLoss (standard, no weighting)
```

The [CLS] pooled representation is used directly as the sentence embedding. No pooling over all tokens — consistent with the original BERT fine-tuning recipe (Devlin et al. 2019 [6]).

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Model | distilbert-base-uncased |
| Max sequence length | 256 tokens |
| Batch size | 32 |
| Learning rate | 2e-5 |
| LR scheduler | CosineAnnealingLR |
| Epochs | 10 (max) |
| Early stopping patience | 3 (on val Micro-F1) |
| Loss | CrossEntropyLoss (unweighted) |
| Seed | 42 |
| Split | 70% train / 15% val / 15% test |
| Stratification | `sklearn.train_test_split(stratify=y)` |

---

## Results

| Metric | Value |
|---|---|
| **Test Micro-F1** | **0.7068** |
| Test Accuracy | — |

This result confirms RQ1: descriptions alone achieve F1 = 0.71 on an 8-class problem with 11.4× imbalance, well above the majority-class baseline (~0.29).

### Per-class interpretation

The 8 geek types vary in lexical distinctiveness:
- **High F1 expected:** War (military vocabulary is unmistakable), Abstract (geometric/mathematical language), Thematic (narrative/adventure language)
- **Lower F1 expected:** Family vs Children's (overlapping vocabulary), Strategy vs War (partial overlap in tactical language), Party (generic social vocabulary)

Phase 2 per-label confusion matrix confirms this pattern.

---

## Comparison to Classical Baseline

| Model | Test F1 | Note |
|---|---|---|
| Majority class | ~0.29 | Predict "War" always |
| TF-IDF + LogReg (Phase 3 context) | 0.57 | Established in Phase 3 for reference |
| **DistilBERT (Phase 1)** | **0.7068** | +14pp over TF-IDF |

The +14 percentage point gain over TF-IDF confirms that contextual embeddings (Devlin et al. 2019 [6]) add substantial value over bag-of-words representations for this domain.

---

## What Phase 1 Establishes

1. **The pipeline works end-to-end:** Parquet → tokenization → BERT → training loop → evaluation is verified and reproducible.
2. **0.7068 is the benchmark** that all Phase 2 optimizations must beat to claim improvement.
3. **DistilBERT is a viable architecture** for this domain — game descriptions are short enough (avg 118 tokens) that the model never truncates, and the vocabulary is domain-specific enough to benefit from sub-word tokenisation.
4. **Text alone is sufficient for type classification** — F1 = 0.71 on a moderately imbalanced 8-class problem is a respectable result that validates the thesis premise (RQ1).

---

## Files in This Folder

| File | Description |
|---|---|
| `Modelling_Phase01.ipynb` | Full notebook — data loading, training loop, evaluation, per-class analysis |
| `Models/best_baseline_model.pt` | Best checkpoint by val Micro-F1 — gitignored, local only |
