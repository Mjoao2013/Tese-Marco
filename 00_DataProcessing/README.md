# 00 — Data Processing: Building the BGG Dataset from Raw XML

## Table of Contents
- [Overview](#overview)
- [Files](#files)
- [XML.ipynb — Raw Parsing](#xmlipynb--raw-parsing)
- [XMLDataset.ipynb — Cleaning & Feature Engineering](#xmldatasetipynb--cleaning--feature-engineering)
- [Data Quality Notes](#data-quality-notes)
- [Glossary](#glossary)

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

---

## Glossary

**API (Application Programming Interface)** — A standardised way for programs to communicate with each other. Here, the BGG XML API v2 is the web service that returns game data as XML files when queried with a game ID.

**DataFrame** — A tabular data structure (like a spreadsheet) used in Python via the `pandas` library. Rows are games, columns are fields like `name`, `description`, `categories`.

**Feature engineering** — The process of creating or transforming input variables (features) to make them more useful for a machine learning model. Here: building `description_clean`, `categories_list`, etc. from raw XML fields.

**HTML entities** — Special character codes used in HTML to represent symbols that would otherwise be interpreted as markup (e.g., `&amp;` = `&`, `&lt;` = `<`, `&#10;` = newline). Game descriptions from BGG are HTML-encoded and must be decoded before processing.

**Label** — A category or class assigned to a data point. In this project, labels are things like `Wargame`, `Dice Rolling`, or `Strategy` — the attributes we want the model to predict.

**Label noise** — Errors or inconsistencies in the ground-truth labels. In BGG, labels are community-assigned, so the same game may be tagged differently by different users, or tags may be applied inconsistently across games.

**Lemmatization** — Reducing a word to its dictionary base form (lemma), taking grammar into account. Example: `rolling` → `roll`, `games` → `game`, `strategies` → `strategy`. Different from stemming, which just chops endings without linguistic knowledge.

**Multi-label** — A classification setting where each data point can belong to multiple classes simultaneously. Example: a game can be both `Card Game` AND `Fantasy` AND `Fighting`.

**Multi-hot encoding** — A binary vector representation for multi-label problems. A game with 85 possible labels is represented as a vector of 85 zeros and ones, where `1` means the label is present. Example: `[0, 0, 1, 0, 1, 0, ...]`.

**Parquet** — A compressed, columnar binary file format for tabular data. Much faster to read than CSV and preserves data types (including Python lists). Used here to store the processed dataset efficiently.

**POS tagging (Part-of-Speech tagging)** — Labelling each word in a sentence with its grammatical role (noun, verb, adjective, etc.). Used here to improve lemmatization accuracy — the word `playing` is lemmatized differently as a verb (`play`) vs. a noun (`playing`).

**Single-label** — A classification setting where each data point belongs to exactly one class. Example: a game has exactly one geek type: either `War` or `Family` or `Strategy`, never both.

**Stopwords** — Very common words that carry little meaning and are typically removed before text analysis: `the`, `a`, `is`, `of`, `and`, etc. After removal, the remaining words are more informative for classification.

**Tokenisation** — Splitting text into individual units (tokens), usually words or sub-words. `"Dice rolling game"` → `["dice", "rolling", "game"]`.

**Token** — A single unit produced by tokenisation. Can be a word, a word fragment (in sub-word tokenisers like BERT's WordPiece), or punctuation.

**Type-token ratio (TTR)** — A measure of vocabulary richness = (number of unique words) / (total words). A TTR of 0.70 means 70% of words in a text are unique, indicating high lexical diversity. Higher TTR = richer vocabulary.

**XML (Extensible Markup Language)** — A structured text format for storing and transporting data using nested tags. Example: `<name type="primary">Catan</name>`. BGG distributes game data as XML files.
