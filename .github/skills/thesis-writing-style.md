---
name: thesis-writing-style
description: >
  Academic writing style guide extracted from the author's own partial thesis (João 2026) and two
  peer Iscte Master's dissertations (Arsénio 2025; Borralho 2025). Primary source is the author's
  own text — all templates and patterns match the exact register, vocabulary, and sentence structure
  already established in the dissertation. Use when drafting or reviewing any section of the BGG
  multi-label classification thesis.
---

# Thesis Writing Style Guide

**Primary source (the author's own text):**
- **João (2026, partial)** — *Text Mining and Semantic Classification of Board Games Using Natural Language Processing* — Chapters 1–3 complete; Chapters 4–6 (Methodology, Results, Conclusion) to be developed.

**Supplementary reference sources:**
- **Arsénio (2025)** — *Anticipating Financial Risk: ML for Debt Management in Telecommunications* (147 pages)
- **Borralho (2025)** — *A Data-Driven Approach to Nighttime Urban Mobility* (71 pages)

> **Usage rule:** When generating or reviewing text, match the style of the primary source (João 2026) exactly. The supplementary sources inform structural patterns only.

---

## Table of Contents

- [Core Register Rules](#core-register-rules)
- [Sentence-Level Patterns](#sentence-level-patterns)
- [Section Templates](#section-templates)
  - [Abstract](#abstract)
  - [Introduction](#introduction)
  - [Related Work / Literature Review](#related-work--literature-review)
  - [Methodology](#methodology)
  - [Results and Discussion](#results-and-discussion)
  - [Conclusion](#conclusion)
- [Transition Vocabulary](#transition-vocabulary)
- [Citing Results with Precision](#citing-results-with-precision)
- [Common Anti-Patterns to Avoid](#common-anti-patterns-to-avoid)
- [BGG Thesis Application Examples](#bgg-thesis-application-examples)

---

## Author's Writing Fingerprint (João 2026)

These are the precise stylistic traits observed in the existing chapters. All generated text must match them.

### Spelling convention
**British English** throughout — use: *characterised*, *organised*, *recognised*, *modelling*, *categorisation*, *behaviour*, *favour*, *emphasise*, *analyse*, *colour*, *neighbours*.  
Never use American variants (*characterized*, *organized*, *modeling*).

### Vocabulary fingerprint

| Preferred term | Avoid |
|----------------|-------|
| `board-game descriptions` | "game descriptions", "game texts" |
| `ludic metadata` / `ludic attributes` | "game metadata" alone |
| `geek type` | "geek_type", "game type" |
| `mechanics` (plural, never "mechanism" alone) | "game mechanics" (redundant) |
| `multi-label classification` | "multilabel", "multi label" |
| `unstructured textual data` | "raw text", "free text" |
| `pre-trained language models` | "pretrained LLMs" without context |
| `the present dissertation` / `the present work` | "my thesis", "our work" |
| `the proposed approach` | "our approach", "my method" |
| `community-generated` | "user-generated" in isolation |
| `CRISP-DM paradigm` / `CRISP-DM methodology` | "CRISP-DM process" alone |

### Paragraph structure rules (from analysis)

1. **Each subsection opens by linking back** to the previous one:
   - *"As discussed in the previous sections, …"*
   - *"Having established that …, the next step is to examine …"*
   - *"Building on this foundation, …"*
   - *"The previous sections showed that …"*
   - *"A key methodological consequence of this problem formulation is that …"*

2. **Each literature paragraph follows this arc:**
   - Name the author(s) and reference: *"[Author] et al. [N] [verb — examine / propose / highlight / demonstrate] …"*
   - Describe their finding or contribution.
   - Connect explicitly to the BGG thesis problem: *"This is especially relevant for board-game descriptions, where …"* or *"This perspective aligns closely with the present work, since …"*

3. **Each subsection ends with a synthesis sentence** that names the gap or transition:
   - *"Taken together, these studies demonstrate that …"*
   - *"Both studies converge on a central point: …"*
   - *"Overall, the work of [X] demonstrates that …"*

4. **No personal pronouns** in methodology or analysis — use passive or impersonal constructions:
   - ✅ *"the problem is set up as …"* / *"a complementary search was conducted"*
   - ❌ *"I set up the problem as …"* / *"we conducted a search"*

5. **Hedging is precise** — not generic:
   - ✅ *"models trained from scratch tend to struggle to learn stable linguistic patterns"*
   - ❌ *"it might be difficult for models to learn"*

### Citation style
- Inline numeric: `[N]` — e.g., *"Devlin et al. [3]"*, *"Liu et al. [12]"*
- First mention: full surname(s) + `[N]` → *"Martoglia and Pontiroli [1]"*
- Subsequent mentions: *"Liu et al."* without repeat number
- Never use *"(Author, Year)"* APA format — this thesis uses IEEE-style numeric references

### Sentence-length profile
Mix long compound sentences (40–60 words, subordinate clauses) with shorter declarative punches (10–18 words). Avoid three consecutive short sentences.

---

## Core Register Rules

| Rule | ✅ Do | ❌ Don't |
|------|-------|---------|
| **Voice** | Use third-person passive for methodology steps ("A comparative evaluation was conducted…") | First-person action verbs for methods ("I ran…", "We tested…") |
| **Tense** | Present for claims and contributions; past for experiments and results | Mixing tenses arbitrarily within a paragraph |
| **Hedging** | Hedge claims that are not proven ("suggests", "indicates", "was observed to") | Over-confident assertions without citing evidence ("proves", "definitively shows") |
| **Precision** | Cite exact numbers ("achieved a 99% accuracy", "72 representative grid cells") | Vague quantifiers ("very high accuracy", "many cells") |
| **Paragraph structure** | One claim per paragraph; end paragraph with implication or limitation | Multi-topic paragraphs or paragraphs with no conclusion sentence |
| **Nomenclature** | Define acronyms on first use ("Principal Component Analysis (PCA)") | Using acronyms before definition |
| **Linking** | Every section opens with a sentence linking back to the previous section | Abrupt section starts with no context |

---

## Sentence-Level Patterns

### Opening a chapter or major section

> *"This chapter presents / addresses / describes …"*

**Example (Arsénio):**
> "The telecommunications industry is characterized by intense competition and rapid technological evolution, making financial stability a critical factor for sustained growth."

**Pattern:** `[Domain] is characterized by [challenges], making [concept] a [adjective] factor for [goal].`

---

### Stating the research objective

> *"This [thesis / dissertation / work] focuses on / aims to / addresses …"*

**Example (Arsénio):**
> "This dissertation focuses on leveraging machine learning techniques to analyze and predict customer payment behavior …, aiming to reduce financial losses associated with unpaid debts."

**Pattern:** `This dissertation focuses on [method] to [action] [object], aiming to [desired outcome].`

---

### Describing methodology at high level

> *"Following the [framework] methodology, [approach] was adopted …"*

**Example (Borralho):**
> "Following the CRISP-DM methodology, a two-phase approach was adopted."

**Pattern:** `Following [methodology name], [noun phrase for approach] was adopted.`

---

### Introducing a phase or step

> *"In the [first / second] phase, [method] was applied to [goal] …"*

**Example (Borralho):**
> "In the first phase, an unsupervised analysis was applied to identify and characterize urban zones associated with nighttime mobility, combining in a complementary manner a static POI-based classification with behavioral clustering supported by PCA and K-Means."

**Pattern:** `In the [N]th phase, [method] was applied to [verb phrase], combining [component A] with [component B].`

---

### Reporting a result

> *"[Model / Method] achieved / demonstrated / showed [metric], [condition qualifier]."*

**Example (Arsénio):**
> "Among the algorithms tested, Random Forest achieved the highest accuracy of 99%, enabling early identification of potential defaulters."

**Example (Borralho):**
> "The MLP model achieved the highest predictive accuracy, maintaining strong performance even when applied exclusively on the nighttime period."

**Pattern:** `[Model] achieved [superlative] [metric], [present participle clause with implication].`

---

### Connecting findings to literature / implications

> *"[Finding] is consistent with / aligns with / confirms … [implication]."*

**Example (Borralho):**
> "Feature importance analysis confirmed that short-term temporal lags are the main determinants of model accuracy."

**Pattern:** `[Analysis type] confirmed that [specific finding] are the main [determinant/driver/factor] of [target variable].`

---

### Acknowledging limitations and future work

> *"Future work may / could explore [topic], [alternative approach], and [extension] to further improve …"*

**Example (Arsénio):**
> "Future work may explore focused clustering of non-compliant clients, alternative data preprocessing, and time series forecasting to further improve predictive accuracy and operational utility."

**Pattern:** `Future work may explore [topic 1], [topic 2], and [topic 3] to further improve [metric] and [utility].`

---

## Section Templates

### Abstract

**Length:** 200–300 words, 4–5 paragraphs.

```
[Paragraph 1 — Problem & Context]
[Domain] presents [challenge], making [objective] essential for [stakeholders].

[Paragraph 2 — Methodology]
Following the CRISP-DM methodology, [high-level approach description].
[Describe Phase 1 briefly.]
[Describe Phase 2 briefly.]

[Paragraph 3 — Key Results]
[Best model/approach] achieved [metric]. [Second key finding.]
[Optional: quantitative characterization of output.]

[Paragraph 4 — Implications]
The findings demonstrate the effectiveness of [approach combination] for [goal].
[One sentence on practical or theoretical contribution.]

[Paragraph 5 — Future Work (optional)]
Future work may explore [directions] to further improve [aspect].

Keywords: [Keyword 1], [Keyword 2], [Keyword 3], [Keyword 4], [Keyword 5]
```

---

### Introduction

**Recommended subsections:**
1. **Context and Motivation** — situate the domain problem; cite economic/social relevance
2. **Research Gap** — what is missing in the literature
3. **Research Question and Objectives** — one primary RQ + 3–5 sub-objectives
4. **Work Outline** — one paragraph per chapter, forward-looking

**Opening sentence pattern:**
> "The growing availability of [data type] has opened new possibilities for [analytical task], particularly in [domain context]."

**Research gap sentence pattern:**
> "Despite the advances in [related area], there remains a lack of / limited work on [specific gap]."

**Objective formulation:**
> "The main objective of this thesis is to [verb] [method/model] to [task] in the context of [domain], with a focus on [specific challenge]."

---

### Related Work / Literature Review

**Structure:** Use a systematic review framing (as Borralho does):
1. **Review Methodology** — search strategy, inclusion/exclusion criteria, PRISMA or similar
2. **Thematic subsections** — each subsection = one conceptual cluster of papers
3. **Synthesis paragraph** — end each subsection with a 2–3 sentence synthesis that identifies gaps

**Synthesis paragraph pattern:**
> "The studies reviewed in this section demonstrate [common finding]. However, [limitation common to most works] remains an open challenge. This thesis addresses this gap by [your contribution]."

**Table pattern for paper comparison:**

| Reference | Method | Dataset | Metric | Limitation |
|-----------|--------|---------|--------|------------|
| Author (Year) | ... | ... | ... | ... |

---

### Methodology

**Opening sentence:**
> "This chapter describes the methodological framework adopted in this work, structured according to the CRISP-DM process model."

**Subsection order (CRISP-DM):**
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation *(may be merged with Results)*

**Data description sentence pattern:**
> "The dataset comprises [N] records / observations, covering [time period / scope], collected from [source]. Each record contains [features description]."

**Model selection justification pattern:**
> "[Model] was selected because [theoretical reason], which is particularly suited to [characteristic of the problem]."

**Evaluation metrics justification pattern:**
> "Given the [class imbalance / multi-label nature / regression nature] of the problem, [metric] was chosen as the primary evaluation criterion, as it [why it is appropriate]."

---

### Results and Discussion

**Opening sentence:**
> "This chapter presents and discusses the results obtained across the [N] experimental conditions / model configurations described in Chapter [N]."

**Result reporting pattern:**
> "Table [N] summarises the performance of all models evaluated. [Model] outperformed the remaining approaches, achieving [metric value] on [dataset/split], compared to [second-best] ([metric value])."

**Discussion paragraph pattern:**
> "The superiority of [model] can be attributed to [mechanistic explanation]. This result is consistent with [reference], who report [similar finding] in [related domain]."

**Negative result framing:**
> "Contrary to initial expectations, [model] underperformed relative to [baseline]. A possible explanation is [reason], which [mechanism]."

---

### Conclusion

**Structure:**
1. **Main Findings** — 1 paragraph per key contribution
2. **Limitations** — honest, specific (not generic), 3–5 bullet points or short paragraphs
3. **Future Work** — 3–5 concrete directions, ordered by feasibility

**Opening sentence:**
> "This thesis investigated [research question], contributing [specific deliverable] to the field of [domain]."

**Limitation sentence pattern:**
> "A key limitation of this work is [specific constraint], which [consequence]. Future research should address this by [mitigation]."

---

## Transition Vocabulary

| Function | Phrases |
|----------|---------|
| **Adding information** | Furthermore, Moreover, In addition, Building on this, |
| **Contrasting** | However, Nevertheless, In contrast, Despite this, Conversely, |
| **Causal** | As a result, Consequently, Therefore, This led to, This enabled, |
| **Sequencing** | First, Subsequently, In the first phase, Following this, |
| **Exemplifying** | For instance, For example, Specifically, As illustrated in Figure N, |
| **Summarising** | Overall, In summary, Taken together, These findings suggest, |
| **Conceding** | Although, While, Even though, It should be noted that, |
| **Emphasising** | Notably, Importantly, Of particular interest, It is worth highlighting, |

---

## Citing Results with Precision

Always pair a claim with:
1. A **precise number** (accuracy, F1, RMSE, etc.)
2. The **condition** under which it was obtained (dataset, split, configuration)
3. A **comparison baseline** when relevant

**Template:**
> "[Model] achieved [metric] = [value] on [test/validation set], outperforming [baseline] by [delta] ([baseline metric] = [baseline value])."

**Example:**
> "The Random Forest classifier achieved an accuracy of 99% on the held-out test set, outperforming the Logistic Regression baseline by 4 percentage points (95%)."

---

## Common Anti-Patterns to Avoid

| Anti-Pattern | Instead |
|--------------|---------|
| "We can clearly see from the graph..." | "Figure N shows that..." |
| "The results are very good" | "The model achieved an F1-macro of 0.87, representing a 12% improvement over the baseline." |
| "In this section we will..." | "This section presents..." |
| "It is obvious that..." | Remove — state the claim directly and cite evidence |
| "A lot of studies have shown..." | "Several studies [1, 4, 9] demonstrate that..." with specific citations |
| Long lists of methods with no synthesis | End every sub-section with a synthesis/gap sentence |
| Passive + vague agent: "It was decided to..." | "Random Forest was selected because..." |
| Abbreviating before defining | "Multi-Label Guided Network (MLGN)..." on first use |

---

## BGG Thesis — Continuation Templates

These examples are written to sound like a seamless continuation of the existing Chapters 1–3. Use them as direct templates for the remaining chapters.

---

### Methodology chapter opening (Chapter 4)

> "This chapter describes the data collection, preparation, and modelling pipeline adopted in this dissertation. The workflow is structured according to the CRISP-DM paradigm introduced in Section 1.5, progressing from business and data understanding through to the design and evaluation of the multi-label classification models. Section 4.1 characterises the dataset; Section 4.2 details the preprocessing and feature engineering steps; Section 4.3 defines the architectures evaluated; Section 4.4 presents the evaluation protocol."

---

### Data Understanding paragraph

> "The dataset used in this dissertation was collected from the BoardGameGeek platform and comprises [N] board-game entries, each accompanied by a free-text description and a set of community-assigned labels spanning three hierarchically structured taxonomies: mechanics ([N] distinct labels), categories ([N] labels), and geek type ([N] labels). As noted in Chapter 1, these labels are manually assigned by community members and frequently exhibit inconsistencies, overlapping semantics, and severe class imbalance — characteristics that compound the difficulty of the multi-label prediction task."

---

### Model introduction paragraph (MLGN)

> "Building on the transfer learning foundations discussed in Section 2.2.3 and the label-dependency modelling reviewed in Section 2.2.6, the primary architecture evaluated in this dissertation is the Multi-Label Guided Network (MLGN) proposed by Liu et al. [12]. MLGN extends a BERT-based encoder with two complementary components: a contrastive learning objective that aligns document representations with their associated label sets, and a Graph Convolutional Network (GCN) module that explicitly captures co-occurrence dependencies between labels. This architecture is particularly well suited to the present problem, since mechanics, categories and geek type exhibit the structured co-occurrence patterns that MLGN is designed to exploit."

---

### Baseline description paragraph

> "In order to assess the contribution of label-aware modelling, each of the three classification tasks is also evaluated using a Binary Relevance (BR) baseline in which an independent BERT classifier is fine-tuned for each label. Although this approach ignores inter-label dependencies — a known limitation identified in Section 2.2.4 — it provides a transparent reference point against which the benefits of MLGN's correlation module can be measured."

---

### Results introduction paragraph

> "This chapter presents the experimental results obtained across the three classification tasks — mechanics, categories, and geek type — and discusses the performance of the evaluated architectures relative to the research questions formulated in Section 1.4. Section 5.1 provides an overview of the experimental setup; Section 5.2 reports global performance metrics; Section 5.3 examines per-label performance and the effect of label imbalance; Section 5.4 discusses the findings in relation to the literature reviewed in Chapter 2."

---

### Discussion of a negative or unexpected result

> "Contrary to initial expectations, [model] did not outperform the Binary Relevance baseline on the *geek type* task. A plausible explanation is that geek type labels represent abstract, community-driven perceptions of complexity and audience — attributes that are less directly recoverable from descriptive text than structural mechanics. This finding is consistent with the observation in Section 1.1 that geek type 'represents a more abstract and community-driven classification', and suggests that its prediction may benefit from auxiliary signals beyond the free-text description alone."

---

### Limitation paragraph template

> "A key limitation of this work concerns [specific issue], which [consequence for validity or scope]. This limitation arises primarily from [cause], a constraint that is difficult to eliminate given [reason]. Future research should address this by [concrete mitigation], which would allow [improvement]."

---

### Conclusion opening (Chapter 6)

> "This dissertation investigated whether written board-game descriptions contain sufficient semantic information to support the automatic prediction of multiple ludic attributes — specifically mechanics, categories, and geek type — through multi-label text classification. The research was motivated by the practical limitations of manually assigned BGG metadata, which, as established in Chapter 2, is informal, inconsistent, and unable to scale with the continuous growth of the platform's game catalogue."