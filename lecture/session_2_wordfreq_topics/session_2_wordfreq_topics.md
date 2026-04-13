---
marp: true
theme: poli3148
paginate: true
header: 'POLI3148. 2026 Spring. Text Analysis II.'
---

<!-- _class: title -->

# Text Analysis II
## Word Frequencies, Word Clouds, and Topic Modeling

POLI3148 Data Science in Politics and Public Administration

Dr. Chen Haohan 
The University of Hong Kong


---

## Review: What We've Done

- **Session 1:** Loaded MoFA data, extracted textual features
  - Tokenization, lemmatization, POS tagging, NER
- **Session 2:** Use these features to construct a dataset --> generate **insights** about the corpus

---

<!-- _class: divider -->

# Part 1: Bag-of-Words Model

---

## Turning Text into Numbers

- Computers need numbers, not words
- **Bag-of-Words (BoW):** Turn a corpus of words into a table. In this table, each document takes a row, each word takes a column, and the numbers in the cells indicate how *often* each word appears in each document
- Ignore word order. Just count occurrences

---

## BoW Example

|  | she | loves | pizza | is | delicious | good | person |
|---|---|---|---|---|---|---|---|
| "She loves pizza, pizza is delicious" | 1 | 1 | 2 | 1 | 1 | 0 | 0 |
| "She is a good person" | 1 | 0 | 0 | 1 | 0 | 1 | 1 |

- Each row = a document
- Each column = a token
- Each cell = an indicator of prevalence (e.g., count, normalized count)

<!-- Missing "a" as a column -->

<!-- I do not put "count" here, to make run for bag-of-word -->


---

## BoW Strengths and Limitations

- **Strengths:** simple, fast, surprisingly effective for many tasks
- **Limitation:** word order is lost
  - "Russia attacks Ukraine" and "Ukraine attacks Russia" → identical BoW
- Addressing the limitation
  - Using multiple words (N-gram)
  - Large Language Models


---

<!-- _class: divider -->

# Part 2: Word Frequencies

---

## Pre-processing Pipeline

Before counting words, we clean:

1. Remove stopwords ("the", "is", "a"...)
2. Remove punctuation
3. Lemmatize ("running" → "run")
4. Remove very rare words (appear < 10 times)

Goal: keep only **substantively meaningful** words

---

## Word Frequencies Tell Us about the Corpus

- Most frequent words = main content
- In MoFA data: "comment", "President", "foreign", "country", "Minister"...
- But some frequent words are **domain-specific noise** ("China" appears in almost every Q&A)
- Solution: manual filtering based on domain knowledge

---

## Visualizing with Word Clouds

- Size = frequency (bigger word = more common)
- Quick visual overview of corpus content
- Can compare subgroups:
  - By spokesperson
  - By time period
  - By topic or country

---

<!-- _class: divider -->

# Part 3: N-grams

---

## Beyond Single Words

- **Unigram:** "South", "China", "Sea" (three separate words)
- **Bigram:** "South China", "China Sea"
- **Trigram:** "South China Sea"
- Multi-word expressions carry meaning that single words miss

---

## N-gram Examples in MoFA Data

- "Foreign Minister" -- more informative than "Foreign" alone
- "human rights" -- a concept, not two words
- "South China Sea" -- a specific location
- "press conference" -- domain phrase

Implementation: `CountVectorizer(ngram_range=(1,2))` -- one parameter change

---

<!-- _class: divider -->

# Part 4: TF-IDF

---

## The Problem with Raw Counts

- "China" appears 2,796 times -- most frequent word
- But it appears in almost every document -- **not informative**
- "Uyghur" appears rarely -- but very informative when it does
- We need to weight words by **informativeness**

---

## TF-IDF Formula

**TF** (Term Frequency): how often does this word appear in THIS document?

**IDF** (Inverse Document Frequency): how rare is this word across ALL documents?

**TF-IDF = TF × IDF**

High TF-IDF = frequent in this document BUT rare overall = **informative**

---

## TF-IDF in Practice

- Swap `CountVectorizer` for `TfidfVectorizer` -- one line change
- Top by raw frequency: China, comment, say, report...
- Top by TF-IDF: more specific, substantive terms surface
- TF-IDF features often outperform raw counts for classification (Session 3)

---

<!-- _class: divider -->

# Part 5: Topic Modeling

---

## Limitations of Individual Words

- Words are sometimes "too small" to capture themes
- We want to summarize a corpus into **coherent topics**
- Topic Modeling discovers these themes **automatically** from the data

---

## What is Topic Modeling? (LDA)

- **Assumption:** each document is a mixture of topics; each topic is a mixture of words
- **Input:** document-term matrix + K (number of topics)
- **Output:**
  - Topic-word distributions (which words belong to which topics)
  - Document-topic distributions (which topics belong to which documents)
- The model discovers **both** from the data

---

## Interpreting Topics

- Each topic is represented by its top words
- **You** label the topic based on those words
- Example from MoFA:
  - Topic: "Japan, water, japanese, government" → Japan-related issues
  - Topic: "Taiwan, peace, reunification" → Cross-strait relations
  - Topic: "Russia, Ukraine, conflict" → Russia-Ukraine war

---

## Choosing K (Number of Topics)

- K is a **researcher's choice** -- not automatically determined
- Too few: topics too broad, mix unrelated themes
- Too many: topics too narrow, hard to interpret
- Common approach: try several K values, evaluate interpretability
- In political science: **Structural Topic Modeling (STM)** extends LDA with covariates

---

## Practical Tips for Topic Modeling

- Results depend on preprocessing (stopwords, min frequency)
- Set `random_state` for reproducibility
- Some topics may be "junk" -- that's normal
- Use interactive visualization (pyLDAvis) to explore
- Topic modeling is **exploratory** -- there's no single "correct" result

---

<!-- _class: divider -->

# Hands-On: Notebook Demo

---

## What We'll Do in the Notebook

- Build word frequency tables from MoFA questions
- Generate word clouds (whole corpus, by spokesperson)
- Explore N-grams and TF-IDF
- Fit LDA topic model (K=15) and interpret
- Visualize with pyLDAvis

---

<!-- _class: exercise -->

## Exercise

1. Make comparative word clouds: questions about Japan vs. US
2. Try K=10 and K=25 for topic modeling
3. Discuss: which K produces the most interpretable topics?

