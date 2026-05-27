# 01 — Exploratory Data Analysis

## Table of Contents
- [Overview](#overview)
- [The Three Classification Tasks](#the-three-classification-tasks)
- [Label Distribution Analysis](#label-distribution-analysis)
  - [Geek Type](#geek-type-8-classes)
  - [Categories](#categories-85-labels)
  - [Mechanisms](#mechanisms-195-labels)
- [Label Frequency Plots](#label-frequency-plots)
- [Text Characteristics](#text-characteristics)
- [Description Length by Label](#description-length-by-label)
- [Label Co-occurrence Analysis](#label-co-occurrence-analysis)
- [Key EDA Findings for Modelling](#key-eda-findings-for-modelling)

---

## Overview

This folder contains the EDA notebook that characterises the BGG dataset before any modelling begins. The goal is to understand the structure of the three classification tasks, quantify label imbalance, identify text characteristics that may affect model performance, and ground the experimental design in data-driven decisions.

**Input:** `bgg_clean_lemmatized.parquet` (165,708 games)  
**Notebook:** `Análise_Exploratória.ipynb`

---

## The Three Classification Tasks

| Task | Label type | Labels | Games with label | Avg labels/game |
|---|---|---|---|---|
| `geek_type` | Single-label | 8 | 14,713 | 1.00 |
| `categories` | Multi-label | 85 | 165,709 | 2.74 |
| `mechanisms` | Multi-label | 195 | 143,016 | 2.94 |

Not all games have all three labels — geek_type in particular is missing for ~90% of the corpus, which defines the training set size for Phase 1 and 2.

---

## Label Distribution Analysis

### Geek Type (8 classes)

| Label | Count | % of subset |
|---|---|---|
| War | 4,219 | 28.7% |
| Family | 2,867 | 19.5% |
| Strategy | 2,180 | 14.8% |
| Abstract | 1,847 | 12.6% |
| Thematic | 1,403 | 9.5% |
| Children's | 942 | 6.4% |
| Party | 756 | 5.1% |
| Customizable | 499 | 3.4% |

**Imbalance ratio:** 11.4× (War vs Customizable)  
**Shannon entropy:** 2.73 (out of max 3.0 for 8 classes)  
→ Relatively balanced. Standard CrossEntropy works without weighting.

### Categories (85 labels)

**Imbalance ratio:** 792× (Card Game: 44,384 vs Third-party Expansion: 56)  
**Shannon entropy:** 5.53  
**Top 5:** Card Game (44,384), Wargame (24,894), Fantasy (19,526), Abstract Strategy (17,001), Fighting (14,201)  
**Rare labels (< 500 examples):** 52 out of 85

### Mechanisms (195 labels)

**Imbalance ratio:** 4,245× (Dice Rolling: 42,453 vs rarest)  
**Shannon entropy:** 6.04  
**Top 5:** Dice Rolling (42,453), Hand Management (30,814), Variable Powers (26,789), Hex-and-Counter (23,045), Simulation (21,879)  
**Rare labels (< 100 examples):** ~40 out of 195

---

## Label Frequency Plots

### Geek Type — Top 10 labels by frequency

![Geek Type label frequency](Images/label_freq_geek_type.png)

### Categories — Top 20 labels by frequency

![Category label frequency](Images/label_freq_categories.png)

### Mechanisms — Top 20 labels by frequency

![Mechanism label frequency](Images/label_freq_mechanisms.png)

---

## Text Characteristics

| Metric | Value |
|---|---|
| Mean description length | ~118 tokens |
| Median description length | ~95 tokens |
| Type-token ratio | 0.70–0.72 |
| Stopword density | <1% |
| Max token length (99th pct) | ~380 tokens |

**Key finding:** The vast majority of descriptions fit within BERT's 512-token limit. The 99th percentile is ~380 tokens, meaning truncation will affect fewer than 1% of games.

The high type-token ratio (0.70+) indicates rich, diverse vocabulary — game descriptions use domain-specific jargon ("hexagonal grid", "worker placement", "deckbuilding") that lemmatization preserves well.

---

## Description Length by Label

Games within specific categories or using specific mechanisms tend to have systematically longer or shorter descriptions. This matters because longer descriptions give the model more signal.

### Description length vs. number of mechanisms

![Description length vs number of mechanisms](Images/desc_length_vs_mechanisms.png)

Positive correlation (r = 0.18) — games with more mechanisms have longer descriptions, likely because complex games require more explanation.

### Description length by Geek Type

![Description length by Geek Type label](Images/desc_length_by_geek_type.png)

### Description length by Category

![Description length by Category label](Images/desc_length_by_category.png)

### Description length by Mechanism

![Description length by Mechanism label](Images/desc_length_by_mechanism.png)

---

## Label Co-occurrence Analysis

Pearson correlation between label presence vectors:

| Pair | Correlation |
|---|---|
| mechanisms ↔ categories | **0.36** |
| description length ↔ mechanisms | 0.18 |
| description length ↔ categories | ~0.12 |

**Finding:** mechanisms and categories are moderately correlated — games with more mechanisms tend to have more categories. This is the empirical basis for expecting that explicit label correlation modelling (MLGN, Phase 4) may help on the mechanisms task.

Within categories, strong co-occurrences include:
- Wargame + World War II
- Card Game + Hand Management
- Fantasy + Fighting

Within mechanisms, consistent with Martoglia & Pontiroli (2021):
- Dice Rolling ↔ Push Your Luck
- Hand Management ↔ Deck Building
- Hex-and-Counter ↔ Simulation

### Category co-occurrence heatmap

![Category co-occurrence heatmap](Images/cooccurrence_categories.png)

### Mechanism co-occurrence heatmap

![Mechanism co-occurrence heatmap](Images/cooccurrence_mechanisms.png)

---

## Key EDA Findings for Modelling

1. **Geek type is easy but small.** 14,713 games, mild imbalance — ideal for pipeline validation (Phase 1 & 2).

2. **Categories is the sweet spot.** 165,708 games, 85 labels, moderate imbalance (792×) — large enough for BERT to generalise, small enough label space to be manageable. This is Phase 3.

3. **Mechanisms is the hard problem.** 195 labels, extreme imbalance (4,245×), ~40 labels with fewer than 100 examples. Rare label failure is almost guaranteed without aggressive handling. This is the intended Phase 4 / MLGN task.

4. **Text alone has limits.** Game descriptions often use overlapping vocabulary across semantically adjacent categories (War games and Strategy games, Family games and Children's games). The moderate type-token ratio suggests the text carries genuine signal, but per-class ambiguity will create a performance ceiling.

5. **Spearman r(F1, support) = 0.25–0.27** — confirmed in Phase 3 results — meaning label frequency is a statistically significant but not dominant predictor of per-label performance. Topic specificity (war games have very distinct vocabulary) matters more than raw count.
