# Task for Claude Code on the RTX 2060 desktop (Windows)

Paste everything below the line into Claude Code, started in an empty folder such as `C:\Projects`.

---

You are setting up and evaluating my RAG portfolio project on this Windows PC (NVIDIA RTX 2060, 6 GB). Work step by step, fix problems yourself, and only stop to ask me if something needs my login or a payment. Save tokens: never print or read whole PDFs or long logs; use short scripts and `Select-String`/`grep` with small outputs.

## 1. Setup
1. Check `python --version` (need 3.10+), `git --version`, `ollama --version`. If Ollama is missing, install it with `winget install Ollama.Ollama`. If Python is missing, `winget install Python.Python.3.11`.
2. `git clone https://github.com/programmermars/bursa-project.git` and `cd bursa-project`.
3. `python -m venv .venv`, activate it, `pip install -e ".[dev]"`, `copy .env.example .env`.
4. `ollama pull nomic-embed-text` and `ollama pull qwen2.5:7b`. Make sure `ollama serve` is running.
5. Run `pytest -q` (expect all passed) and `rag doctor`. Fix anything red.

## 2. Get three annual reports
Download the latest full annual report PDF (not the summary) for KLK (Kuala Lumpur Kepong Berhad), Top Glove Corporation and Press Metal Aluminium Holdings from each company's investor relations page or Bursa Malaysia announcements. Use the same year for all three if possible (2025, else 2024). Save into `data\raw\` as `KLK_<year>_annual_report.pdf`, `TopGlove_<year>_annual_report.pdf`, `PressMetal_<year>_annual_report.pdf`. Check each file opens with pypdf and has more than 100 pages. If a site blocks downloading, tell me the URL and I will download it manually.

## 3. Build the index
`rag ingest`. Report the number of chunks.

## 4. Write the evaluation set (about 24 questions)
Write `eval\questions.csv` with columns `id,company,question,expected_pages`.
- For each company, 7 answerable questions covering: principal risks, how one named risk is mitigated, number of Audit Committee meetings, internal audit function (in-house or outsourced, reports to whom, cost), key audit matters from the auditor's report, sustainability or climate risk, one specific figure (e.g. revenue or profit for the year).
- Plus 3 unanswerable questions in total (blank `expected_pages`), e.g. "What is the company's policy on cryptocurrency investments?".
- Find the correct page numbers with `python scripts/find_pages.py <keywords...> --company <code>` (add `--any` to match any keyword), e.g. `"Audit Committee" "met"`, `"key audit matter"`, `"internal audit function"`. It prints only page numbers and a 150-character snippet. Do NOT read whole reports. Use PDF page index starting at 1 (what the viewer shows), not the printed page number. Each question should list every page that contains the answer, separated by `;`.

## 5. Evaluate
1. `rag eval --with-llm` (uses local qwen2.5:7b; free).
2. Also try a chunk-size comparison: set `RAG_CHUNK_SIZE=600` in `.env`, `rag ingest`, `rag eval --out eval\results_600.md`; then set it back to 900 and `rag ingest` again.

## 6. Screenshot and README
1. Start `streamlit run app.py`, ask "What are the principal risks disclosed and how are they mitigated?" for KLK, take a screenshot of the browser window and save it as `docs\screenshot.png` (use any screenshot method available, e.g. PowerShell with System.Windows.Forms, or ask me to take it).
2. In `README.md`, under `## Evaluation`, add the retrieval and generation tables from `eval\results.md` and a line `Chunk size 600 vs 900:` with the two hybrid Hit@5/MRR values. Add `![Demo](docs/screenshot.png)` under the title. Edit `eval\results.md` to remove the "fictional sample report" note.
3. `git add -A`, `git commit -m "Add real-report evaluation and screenshot"`, `git push` (sign in to GitHub if asked; tell me if push fails).

## 7. Report back to me (keep it short, this goes to another assistant)
Print exactly this block and nothing else at the end:

```
REPORT
reports: <company year pages> x3
chunks: <n>
retrieval (k=5): bm25 Hit/Recall/MRR = ... | vector = ... | hybrid = ...
generation (qwen2.5:7b): citation_validity=..., citation_precision=..., abstention=...
chunk600 hybrid: Hit=..., MRR=...
questions: <n answerable> + <n unanswerable>
worst 3 questions (id, question, retrieved pages vs expected): ...
errors or fixes made: ...
pushed: yes/no, commit <hash>
```
