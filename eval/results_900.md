# Evaluation results

Index: `ollama:nomic-embed-text`, 4237 chunks (size 900, overlap 150).

## Retrieval

| Mode | k | Questions | Hit@k | Recall@k | MRR |
|---|---|---|---|---|---|
| bm25 | 5 | 21 | 0.81 | 0.67 | 0.63 |
| vector | 5 | 21 | 0.81 | 0.57 | 0.58 |
| hybrid | 5 | 21 | 0.81 | 0.60 | 0.67 |

Hit@k: share of questions where at least one correct page is in the top k. Recall@k: share of correct pages retrieved. MRR: mean of 1/rank of the first correct page.
