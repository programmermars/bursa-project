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

## 3. Run everything
`rag run`. It checks the setup, builds the index, generates the evaluation questions (`eval\questions_auto.csv`), evaluates retrieval and the local LLM, compares chunk size 600 vs 900, and writes the results into `eval\results.md`, `README.md` and `docs\PROJECT_WRITEUP.md`. If it prints a `!` warning (e.g. Ollama not reachable), fix it and run again.

## 4. Check the auto labels (optional, better numbers)
Open `eval\questions_auto.csv`. For a few rows per company, check the `expected_pages` with `python scripts/find_pages.py <keywords...> --company <code>`. If you correct any, save the corrected rows as `eval\questions.csv` (same columns) and run `rag run` again; it then uses your file instead.

## 5. Screenshot and README
1. Start `streamlit run app.py`, ask "What are the principal risks disclosed and how are they mitigated?" for KLK, take a screenshot of the browser window and save it as `docs\screenshot.png` (use any screenshot method available, e.g. PowerShell with System.Windows.Forms, or ask me to take it).
2. Add `![Demo](docs/screenshot.png)` under the title in `README.md` (the results are already filled in by `rag run`).
3. `git add -A`, `git commit -m "Add real-report evaluation and screenshot"`, `git push` (sign in to GitHub if asked; tell me if push fails).

## 6. Report back to me (keep it short, this goes to another assistant)
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
