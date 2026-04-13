---
marp: true
theme: poli3148
paginate: true
header: 'POLI3148. 2026 Spring. Text Analysis I.'
---

<!-- _class: title -->

# Text Analysis I
## Introduction an Feature Engineering

POLI3148 Data Science in Politics and Public Administration

Dr. Chen Haohan 
The University of Hong Kong


---

## Course Plan: From Traditional Text Analysis to LLMs

### Traditional Text Analysis (Sessions 1--3)

| Session | Topic |
|---------|-------|
| 1 | Text as Data: Introduction |
| 2 | Word Frequencies, Word Clouds, and Topic Modeling |
| 3 | Sentiment Analysis and Text Classification |

### LLM-Powered Text Analysis (Sessions 4--6)

| Session | Topic |
|---------|-------|
| 4 | LLM-Powered Text Analysis: Introduction |
| 5 | LLM Deep Dive -- Batch Processing and Evaluation |
| 6 | Embeddings, RAG, and LLM as Knowledge Base |

---

## One Dataset, Increasingly Powerful Methods

- We will use the **same dataset** throughout all 6 sessions
- First analyze it with traditional methods (Sessions 1--3)
- Then re-do the same tasks with LLMs (Sessions 4--6)
- Compare results: where do LLMs improve? Where do traditional methods hold up?

---

<!-- _class: divider -->

# Case: China's MoFA Press Conferences

---

## What is the MoFA Corpus?

- Chinese Ministry of Foreign Affairs Press Conference Corpus
- 35,346 question-and-answer pairs (2002--2025)
- Published on Harvard Dataverse
- Journalists ask questions, spokespeople answer
- A window into Chinese diplomacy over two decades

---

## Sample Q&A Pair

**Date:** October 15, 2002 | **Spokesperson:** Zhang Qiyue

> **Q:** "How many Hong Kong and Taiwan people were injured in the blasts in Bali Island, Indonesia? Are there any Mainland citizens injured? What has the Chinese Embassy in Indonesia done in this regard?"

> **A:** "China is always opposed to terrorism of all forms. We strongly condemn this violent activity and would like to express deep sympathy for victims..."

This is the very first entry in the dataset -- from the earliest days of regular MoFA press conferences.

---

## What Questions Can We Ask With These Data?

- **Corpus level:** What topics dominate Chinese diplomacy overall? What are the most frequently mentioned countries?
- **Document level:** What topic does this question discuss? Is this specific question aggressive? What countries and people are mentioned?
- **Across documents:** How do different spokespeople differ in their responses? How has diplomatic tone changed over time?

---

## Key Concept: Units of Text Analysis

| Unit | Definition | Example |
|------|-----------|---------|
| **Corpus** | The whole collection | All 35,346 Q&A pairs |
| **Document** | A single text unit | One question or one answer |
| **Paragraph / Sentence etc.** | Sub-document units | A sentence within an answer |
| **Token** | The smallest unit | A word, punctuation mark, or number |

Corpus → Documents → (Paragraphs, Sentences, etc.) → Tokens

Most analysis operates at the **corpus**, **document** or **token** level.

---

<!-- _class: divider -->

# Feature Engineering for Text Data

---

## What the MoFA Dataset Already Contains

The dataset comes with pre-extracted features for each Q&A pair:

| Column | What it contains |
|--------|-----------------|
| `question`, `answer` | Raw text |
| `question_lem`, `answer_lem` | Lemmatized text |
| `q_loc`, `q_per`, `q_org` | Named entities (locations, persons, organizations) |
| `q_sentiment`, `a_sentiment` | Sentiment scores |

How were these features extracted? That's what we'll learn today -- **feature engineering for text**.

---

## Turning Raw Text into Data

- Raw text is unstructured -- computers need structured features
- **Feature engineering:** extract structured information from text
- Two categories:
  - **Linguistic features:** tokenization, lemmatization, POS tagging, stopword removal
  - **Substantive features:** named entities, sentiment scores

The linguistic features are pre-processing steps; the substantive features are what we ultimately care about for analysis.

---

## Tokenization

- Breaking text into individual tokens (words, punctuation)
- `"China condemns the attack."` → `["China", "condemns", "the", "attack", "."]`
- The foundation of all text analysis

> **Aside:** Even state-of-the-art Large Language Models rely on tokenization -- it just runs in the background. Notice that LLM APIs charge you per **token**, not per word or character. We'll revisit this in Session 4.

---

## Stopword Removal

- **Stopwords:** common words that carry little meaning
- Examples: "the", "is", "a", "of", "in", "to", "and"
- They are frequent but **not informative** for most analyses
- Removing them helps focus on substantive content

**Before:** "China is always opposed to terrorism of all forms"
**After:** "China opposed terrorism forms"

---

## Lemmatization

- Reducing words to their base form
  - "running" → "run"
  - "countries" → "country"
  - "was" → "be"
- Why? Different forms of the same word should be treated as the same concept
- The MoFA dataset provides lemmatized text in `question_lem` and `answer_lem`

---

## POS Tagging (Part-of-Speech Tagging)

- Labeling each word's grammatical role
- Useful for filtering: keep only nouns and verbs, remove determiners and prepositions

| POS Tag | Meaning | Examples |
|---------|---------|---------|
| PROPN | Proper noun | China, Biden, NATO |
| NOUN | Common noun | country, attack, policy |
| VERB | Verb | condemn, visit, discuss |
| ADJ | Adjective | nuclear, bilateral, serious |
| DET | Determiner | the, a, this |
| ADP | Preposition | in, on, of, to |
| PUNCT | Punctuation | . , ? ! |

---

## Tokenization in Action

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("China condemns the attack on diplomats.")
```

| Text | Lemma | POS | Stopword |
|------|-------|-----|----------|
| China | China | PROPN | No |
| condemns | condemn | VERB | No |
| the | the | DET | Yes |
| attack | attack | NOUN | No |
| on | on | ADP | Yes |
| diplomats | diplomat | NOUN | No |
| . | . | PUNCT | No |

---

<!-- _class: divider -->

# Named Entity Recognition (NER)

---

## What are Named Entities?

A **named entity** is a real-world object that can be denoted with a proper name: a person, organization, location, date, etc.

Common named entity types:

| Type | Description | Examples |
|------|------------|---------|
| **PERSON** | People's names | Xi Jinping, Biden, Pelosi |
| **ORGANIZATION** | Companies, agencies, institutions | UN, NATO, CPC, APEC |
| **LOCATION** (GPE) | Countries, cities, regions | Taiwan, South China Sea, Beijing |
| **MISC** | Other entities | Chinese, Japanese (nationalities) |

**NER** is the task of automatically finding and classifying these entities in text.

---

## NER Example: A Real MoFA Question

**ID 91** | November 12, 2002 | Spokesperson: Kong Quan

> "Nancy Pelosi has become the minority leader of the U.S. House of Representatives. In light of her stance on the question of human rights, is China afraid that her election will affect Sino-U.S. relations?"

| Entity | Type |
|--------|------|
| Nancy Pelosi | PERSON |
| U.S. | LOCATION |
| House of Representatives | ORGANIZATION |
| China | LOCATION |

Dataset labels: `q_per`: Nancy Pelosi | `q_loc`: U.S.; China | `q_org`: House of Representatives

---

## NER Tools Can Disagree

- spaCy, Flair, and LLMs may extract different entities from the same text
- Why? Different training data, different models, different entity definitions
- The MoFA dataset uses Flair; we use spaCy in our notebook
- Always validate your NER output

---

<!-- _class: divider -->

# Hands-On: Notebook Demo

---

## What We'll Do in the Notebook

- Load the MoFA dataset and explore its structure
- Tokenize sample questions with spaCy
- See tokens, lemmas, POS tags, stopwords
- Extract named entities (NER)
- Compare spaCy NER with the dataset's pre-labeled entities

---

<!-- _class: exercise -->

## Exercise

1. Pick 5 rows from the `answer` column
2. Tokenize them and extract NER with spaCy
3. Compare your entities with the dataset's `a_loc`, `a_per`, `a_org` columns
4. Where do they agree? Where do they disagree? Why?

