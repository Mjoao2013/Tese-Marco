# 00 — Data Processing: Building the BGG Dataset from Raw XML

## Table of Contents
- [Overview](#overview)
- [Files](#files)
- [XML.ipynb — Raw Parsing](#xmlipynb--raw-parsing)
- [XMLDataset.ipynb — Cleaning & Feature Engineering](#xmldatasetipynb--cleaning--feature-engineering)
- [Data Quality Notes](#data-quality-notes)

---

## Overview

This folder contains the two notebooks responsible for the entire data pipeline: from raw BoardGameGeek XML files downloaded from the BGG API to the clean, structured parquet files that all downstream modelling phases consume.

**Source data:** ~166,000 individual XML files (one per game) stored in `Dataset/XML Dataset/xml_raw/`  
**Output:** `bgg_clean_lemmatized.parquet` — the single dataset used by all phases

---

## Files

| File | Purpose |
|---|---|
| `XML.ipynb` | Parses raw XML files, extracts structured fields, builds initial DataFrame |
| `XMLDataset.ipynb` | Cleans text, applies lemmatization, engineers label columns, writes parquet |

---

## XML.ipynb — Raw Parsing

### What it does

Each BGG game is distributed as an individual XML file returned by the BGG XML API v2. The notebook iterates through every file in `xml_raw/` and extracts the following fields:

| Field | XML path | Notes |
|---|---|---|
| `id` | `@id` attribute | Unique BGG game ID |
| `name` | `//name[@type='primary']` | Primary English title |
| `description` | `//description` | Full HTML-encoded text |
| `year_published` | `//yearpublished` | Publication year |
| `min_players` / `max_players` | `//minplayers`, `//maxplayers` | Player count range |
| `playing_time` | `//playingtime` | Stated play time (minutes) |
| `min_age` | `//minage` | Recommended minimum age |
| `categories` | `//link[@type='boardgamecategory']` | List of category labels |
| `mechanisms` | `//link[@type='boardgamemechanic']` | List of mechanism labels |
| `geek_type` | `//link[@type='boardgamefamily']` | BGG family classification |
| `average_rating` | `//statistics/ratings/average` | Community rating (1–10) |
| `num_ratings` | `//statistics/ratings/usersrated` | Number of ratings |

### Key challenges handled
- **HTML entities** in description fields (`&amp;`, `&lt;`, `&#10;`, etc.) — stripped via `html.unescape()`
- **Missing fields** — many games lack player counts, ages, or ratings; handled with `None` defaults
- **Malformed XML** — a small fraction of files have encoding errors; skipped with error logging
- **List fields** — categories, mechanisms, and geek_type are parsed as Python lists, not strings

---

## XMLDataset.ipynb — Cleaning & Feature Engineering

### Text pipeline

The raw `description` field is noisy HTML-decoded prose of varying quality. The pipeline:

1. **HTML stripping** — remove any residual tags
2. **Lowercasing**
3. **Punctuation removal** — keep only alphanumeric and spaces
4. **Tokenisation** — NLTK word tokenizer
5. **Stopword removal** — NLTK English stopwords, resulting in <1% stopword density in final corpus
6. **Lemmatization** — WordNetLemmatizer with POS tagging (nouns, verbs, adjectives)
7. **Short text filter** — games with fewer than 5 tokens after cleaning are dropped

Result stored in column `description_clean`.

### Label engineering

Three classification targets are constructed from the raw list fields:

| Column | Source field | Type | Label count |
|---|---|---|---|
| `geek_type_list` | `geek_type` | Single-label (first element) | 8 |
| `categories_list` | `categories` | Multi-label list | 85 |
| `mechanisms_list` | `mechanisms` | Multi-label list | 195 |

Geek type is treated as single-label because >99% of games have exactly one family assignment. Categories and mechanisms are genuinely multi-label.

### Final dataset statistics

| Statistic | Value |
|---|---|
| Total games (raw) | 166,903 |
| Games dropped (empty/short descriptions) | ~1,195 |
| **Final games** | **165,708** |
| Avg description length | ~118 tokens |
| Type-token ratio | 0.70–0.72 |
| Stopword density | <1% |

### Output files written

| File | Contents |
|---|---|
| `bgg_clean_lemmatized.parquet` | Full dataset, all columns |
| `bgg_geektype_subset.parquet` | Filtered to games with geek_type label |
| `bgg_category_subset.parquet` | Filtered to games with ≥1 category |
| `bgg_mechanisms_subset.parquet` | Filtered to games with ≥1 mechanism |
| `bgg_metadata_final.json` | Label vocabularies and frequency tables |

---

## Data Quality Notes

- **Description quality varies widely:** Flagship games have rich multi-paragraph descriptions; niche/older games may have a single sentence. This directly impacts model performance on rare labels.
- **Label noise:** BGG labels are community-assigned and occasionally inconsistent (e.g., a game tagged both `War` and `Family`). No manual cleaning was performed — this is acknowledged as a limitation.
- **Language assumption:** The pipeline assumes English. Some games have non-English descriptions that survive filtering but degrade tokenisation quality.
- **Temporal bias:** Games published before ~2000 tend to have shorter, lower-quality descriptions written retrospectively by the community.
