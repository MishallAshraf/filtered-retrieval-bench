# Data and evaluation

## Source data

| Item | Value |
|---|---|
| Dataset | FiQA-2018, from the BEIR benchmark |
| Content | Finance questions and forum-style answers |
| Documents | 57,638, of which 57,600 are used after dropping 38 empty ones |
| Document length | Mean 133 words, median 90, 95th percentile 368 |
| Dev questions | 500, with human relevance labels, for tuning |
| Test questions | 648, with 1,706 labels (2.63 relevant documents per question), for final numbers |
| Train questions | Not used, as no model is trained |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, normalised |

The FiQA challenge site says its data is for non-commercial use only, while the BEIR Hugging Face card says cc-by-sa-4.0. This repo publishes code and results, never the dataset.

Documents are stored whole, not chunked. The model reads about 256 tokens, so the ends of the longest documents are ignored. This affects relevance scores but not the comparison with an exact scan.

## Two kinds of correct answer

| Source | Question it answers | Metrics | Script |
|---|---|---|---|
| Exact scan under the same RLS policy | Did the index return the rows it should have? | Rows returned, zero-result rate, short-result rate, recall@10 against exact | `evaluate_fixes.py` |
| FiQA human labels | Are the results useful? | nDCG@10, recall@10, MRR@10 | `evaluate_relevance.py` |

Latency is measured inside the database, so it excludes the network trip.

## Ownership is simulated

FiQA has no users, so ownership is assigned.

**Random ownership, for `evaluate_fixes.py`.** Each user owns a fixed share of randomly chosen documents: 50%, 10%, 5%, 1%, 0.1%. Everything else belongs to `other`. This needs no labels.

**Labelled ownership, for `evaluate_relevance.py`.** With random ownership, a 1% user would own a relevant document for only about 3% of questions, leaving too few to score. So:

1. Split the questions between users so no question is shared.
2. Give each user every relevant document for their questions.
3. Drop any question whose relevant documents are already claimed by another user, or that would overfill the user's share.
4. Fill the rest of each user's share at random.

| User owns | Dev questions | Test questions |
|---|---|---|
| 50% | 250 | 200 |
| 10% | 130 | 178 |
| 5% | 60 | 150 |
| 1% | 60 | 120 |

No questions were dropped in any of the three seeds. The 0.1% user owns about 57 documents, too few for this, so that user is measured only against an exact scan. Placing relevant documents with the user could make relevance scores look better than they would be with natural ownership, so these scores compare search methods and settings and don't predict production quality.

## The guide's schema

`evaluate_guide_schema.py` rebuilds the data in the shape of Supabase's [RAG with Permissions guide](https://supabase.com/docs/guides/ai/rag-with-permissions). Embeddings are copied from the single table, so nothing is re-embedded.

| Table | Rows | Notes |
|---|---|---|
| `guide_documents` | 7,200 | One owner each, assigned in the same shares as the single table |
| `guide_document_owners` | 7,200 | One row per document, for the guide's multiple-owner policy |
| `guide_sections` | 57,600 | One FiQA text each, grouped at random into documents of 8 |

Sections also carry a copied `owner` column, which only the "owner copied onto sections" fix reads. The guide's policies are used as written, with `(select auth.uid())`. Each result records whether it came back in distance order, because `relaxed_order` can return results out of order.

## Search methods compared

- **Vector, no fix:** default settings under RLS.
- **Vector, fixed:** owner index and query filter. Users owning up to 6,000 rows use the owner index, and larger users use the HNSW index with iterative scan.
- **Vector, exact scan:** the upper limit for this embedding model.
- **Keyword:** Postgres full-text search with `ts_rank_cd`.
- **Hybrid:** Reciprocal Rank Fusion (k=60) over the top 50 of each ranking.

Keyword and hybrid were tuned on dev questions with seed 42, across AND and OR and minimum keyword scores of 0, 0.01 and 0.1.

## Rare-term questions

Vector search is often expected to miss exact terms. Questions containing a word found in at most 5 documents are reported separately. This uses the existing labels, so no new labelling was needed. The set is small: 93 scored questions across three seeds.

## Rules that keep results trustworthy

- Settings are chosen on dev questions, then reported on test questions.
- The labelled evaluation uses three ownership seeds and 95% bootstrap confidence intervals. The intervals treat each scored question as independent, though questions repeat across seeds.
- A run fails if any user receives a row they don't own.
- The plan is read with `EXPLAIN` for every question, on the exact query and parameters that run, because the planner can choose differently from one question to the next.
- Results record dataset, model, seed and settings, and the gate refuses to compare results from a different setup.
- Search functions are tested separately, because a function can get a different plan from the same query run directly.

## Not tested yet

- Ownership clustered by topic. Real users' documents tend to share topics, and this could change the results.
- Natural ownership, such as the 150 real inboxes in EnronQA. The Hugging Face card states no dataset licence, and it is personal email.
- Chunking long documents. In the guide's schema, sections are whole FiQA texts grouped at random, not chunks of one document.
- Real sharing between users in the multiple-owner version.
- Relevance scoring on hosted Supabase, where the free database limit is about 500 MB.
- Whether AI agents can diagnose and fix this problem.
