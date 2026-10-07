# Evaluation results

> Current numbers are from the fictional sample report (`scripts/make_sample_report.py`) with the offline TF-IDF backend, as a pipeline check. Replace with results on real annual reports and `EMBED_BACKEND=ollama`.

Index: `tfidf-lsa`, 6 chunks (size 900, overlap 150).

## Retrieval

| Mode | k | Questions | Hit@k | Recall@k | MRR |
|---|---|---|---|---|---|
| bm25 | 5 | 8 | 1.00 | 1.00 | 1.00 |
| vector | 5 | 8 | 1.00 | 1.00 | 0.88 |
| hybrid | 5 | 8 | 1.00 | 1.00 | 0.88 |

Hit@k: share of questions where at least one correct page is in the top k. Recall@k: share of correct pages retrieved. MRR: mean of 1/rank of the first correct page.
