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
- [Glossary](#glossary)

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

---

## Glossary

**[CLS] token** — A special token added at the start of every input by BERT. After passing through all transformer layers, the [CLS] vector is used as a fixed-size summary of the whole sentence — the "sentence embedding". This vector is then fed into the classification head.

**Accuracy** — Fraction of correctly classified examples out of total. For single-label problems: (correct predictions) / (total predictions). Simpler than F1 but misleading with imbalanced data.

**Baseline** — A simple reference model used to judge whether a more complex model is actually better. Without a baseline, you can’t know if 0.70 F1 is good or bad.

**BERT (Bidirectional Encoder Representations from Transformers)** — A large language model pre-trained on massive text corpora (Wikipedia, BookCorpus) using masked word prediction. It reads text in both directions simultaneously, producing rich contextual representations. Fine-tuning adds a task-specific head on top.

**CosineAnnealingLR** — A learning rate scheduler that gradually decreases the learning rate following a cosine curve — fast at first, then slowing down, reaching near-zero at the end of training. Helps avoid overshooting the optimal weights.

**CrossEntropyLoss** — The standard loss function for single-label multi-class classification. It penalises the model when it assigns low probability to the correct class. Mathematically: −log(p_correct_class).

**DistilBERT** — A compressed (distilled) version of BERT with 6 layers instead of 12, trained to mimic BERT’s behaviour. 40% smaller, 60% faster, retains ~97% of BERT’s performance on most tasks (Sanh et al. 2019).

**Dropout** — A regularisation technique that randomly sets a fraction of neuron outputs to zero during training. Prevents the model from memorising training data (overfitting). Dropout=0.2 means 20% of values are zeroed at each step.

**Early stopping** — Stops training when performance on the validation set stops improving for N consecutive epochs (patience=N). Prevents overfitting and saves compute.

**Embedding** — A dense numeric vector that represents a word, sentence, or label in a continuous space. Similar meanings are close together in this space. BERT produces contextual embeddings — the same word gets different vectors depending on its sentence context.

**Epoch** — One complete pass through the entire training dataset. If you train for 10 epochs, the model sees every training example 10 times.

**Fine-tuning** — Taking a pre-trained model (e.g., BERT trained on Wikipedia) and continuing to train it on your specific task with your labelled data. The pre-trained weights provide a good starting point; fine-tuning adapts them to the domain.

**Hyperparameter** — A setting chosen before training begins that controls the learning process, not learned from data. Examples: learning rate, batch size, dropout rate, number of epochs.

**Learning rate (LR)** — How large a step the model takes when updating its weights after each batch. Too high: training is unstable, overshoots. Too low: training is slow. Typical values for BERT fine-tuning: 1e-5 to 5e-5.

**Linear layer** — A simple mathematical transformation: multiply input by a weight matrix, add a bias. Also called a fully-connected layer. Used as the classification head on top of BERT to produce class scores.

**Logits** — Raw, unnormalised scores output by the model before any activation function (sigmoid or softmax). Can be any real number — positive or negative.

**Majority-class baseline** — The simplest possible model: always predict the most frequent class. For geek_type, always predicting “War” gives ~29% accuracy. Any serious model must beat this.

**Micro-F1** — A single F1 score computed across all classes by counting total true positives, false positives, and false negatives globally. Dominated by frequent classes. High Micro-F1 means the model does well overall, but may be terrible on rare classes.

**Overfitting** — When a model learns the training data so well that it fails to generalise to new examples. Signs: training loss keeps dropping while validation loss starts rising.

**Patience** — In early stopping: the number of epochs without improvement to tolerate before stopping training. Patience=3 means “stop if validation metric hasn’t improved for 3 consecutive epochs”.

**Recall** — Of all the actual positives, how many did the model find? Recall = TP / (TP + FN). A model with high recall finds most positives but may also flag many negatives incorrectly.

**Seed (random seed)** — A starting number that controls all randomness in the experiment (weight initialisation, data shuffling). Setting seed=42 makes results reproducible: anyone running the same code with the same seed gets the same result.

**Softmax** — Converts a vector of raw logits into probabilities that sum to 1. Used for single-label classification to pick the most likely class.

**Stratification** — Ensuring that each class is represented proportionally in train/val/test splits. Without it, a rare class might appear in training but not in test (or vice versa), making evaluation unreliable.

**Transfer learning** — Using knowledge learned from one task (BERT pre-trained on Wikipedia) to improve performance on a different task (game type classification). The pre-trained model already “understands” language; you just teach it your specific classification task.

**Transformer** — A neural network architecture based on self-attention mechanisms, enabling the model to weigh the importance of each word relative to every other word in a sequence. BERT, RoBERTa, and DistilBERT are all transformer models.
