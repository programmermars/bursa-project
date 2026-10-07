# Evaluation results

*Last run: 2026-10-07 by `rag run`. Questions: auto-labelled (keyword rules). Search: `tfidf-lsa`. Answers: `none`.*

| Report | Pages (with text) |
|---|---|
| DemoPlantation 2025 | 6 (6) |

| Search mode | Questions | Hit@5 | Recall@5 | MRR |
|---|---|---|---|---|
| bm25 | 7 | 1.00 | 1.00 | 0.93 |
| vector | 7 | 1.00 | 1.00 | 0.83 |
| hybrid | 7 | 1.00 | 1.00 | 0.83 |

| Variant (hybrid search) | Hit@5 | MRR |
|---|---|---|
| Current: chunk 900, section prefix on | 1.00 | 0.83 |
| Chunk size 600 | 1.00 | 0.90 |
| Section prefix off | 1.00 | 0.83 |

Best search mode: **bm25**. Hybrid missed 0 question(s). Details in [eval/results.md](results.md).

> These numbers come from the fictional sample report and only show that the pipeline works. Upload real annual reports and run `rag run` to replace them.

> Correct pages were found by keyword rules (`src/rag/autolabel.py`), not checked by hand, which favours keyword (BM25) search. Treat them as indicative.

Index: `tfidf-lsa`, 6 chunks (size 900, overlap 150, section prefix on).

## Retrieval

| Mode | k | Questions | Hit@k | Recall@k | MRR |
|---|---|---|---|---|---|
| bm25 | 5 | 7 | 1.00 | 1.00 | 0.93 |
| vector | 5 | 7 | 1.00 | 1.00 | 0.83 |
| hybrid | 5 | 7 | 1.00 | 1.00 | 0.83 |

Hit@k: share of questions where at least one correct page is in the top k. Recall@k: share of correct pages retrieved. MRR: mean of 1/rank of the first correct page.
