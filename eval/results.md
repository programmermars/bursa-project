# Evaluation results

Index: `ollama:nomic-embed-text`, 7107 chunks (size 600, overlap 150).

## Retrieval

| Mode | k | Questions | Hit@k | Recall@k | MRR |
|---|---|---|---|---|---|
| bm25 | 5 | 21 | 0.86 | 0.69 | 0.68 |
| vector | 5 | 21 | 0.76 | 0.50 | 0.59 |
| hybrid | 5 | 21 | 0.95 | 0.68 | 0.76 |

## Generation (ollama)

| Metric | Value |
|---|---|
| Citation validity (cited source exists) | 1.00 |
| Citation precision (cited page is a correct page) | 0.62 |
| Abstention accuracy (says 'not found' when it should) | 1.00 |

Hit@k: share of questions where at least one correct page is in the top k. Recall@k: share of correct pages retrieved. MRR: mean of 1/rank of the first correct page.
