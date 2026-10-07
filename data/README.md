# Getting the annual reports

The PDFs are not stored in this repository. Download them yourself:

1. Go to the company's investor relations page, or Bursa Malaysia's company announcements (category "Annual Report").
2. Download the full annual report PDF (not the summary).
3. Save it in `data/raw/` named `COMPANY_YEAR_anything.pdf`, for example:
   - `KLK_2025_annual_report.pdf`
   - `TopGlove_2025_annual_report.pdf`
   - `PressMetal_2025_annual_report.pdf`

The part before the first underscore becomes the company name used in filters and citations.

Then run `rag ingest`.

To try the project without downloading anything, run `python scripts/make_sample_report.py`. It creates a short fictional report.
