"""Generate a FICTIONAL sample annual report (DemoPlantation_2025.pdf) for tests and a no-download demo.

The company and every figure in it are invented. Use real reports in data/raw/ for actual analysis.
"""
from pathlib import Path
import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("Corporate Information",
     "Demo Plantation Berhad is a fictional company used only to demonstrate this project. "
     "It operates oil palm estates in Perak and Sabah and a refinery in Klang."),
    ("Chairman's Statement",
     "Revenue rose 8 percent to RM2.1 billion on higher crude palm oil prices, while fresh fruit bunch "
     "production fell 3 percent because of ageing trees and labour shortages in Sabah."),
    ("Statement on Risk Management and Internal Control",
     "The Board has identified the following principal risks. Commodity price risk: CPO price volatility is "
     "managed through forward sales of up to 40 percent of expected output. Labour shortage risk: reliance on "
     "foreign workers is mitigated by mechanisation of harvesting and in-field collection. Climate and weather "
     "risk: El Nino related drought may reduce yields; the Group invests in water management and irrigation. "
     "Regulatory and sustainability risk: non-compliance with MSPO and EU deforestation rules could restrict "
     "market access; traceability to plantation level is maintained."),
    ("Audit Committee Report",
     "The Audit Committee met five times during the financial year. It reviewed the quarterly results, the "
     "annual internal audit plan, related party transactions and the external auditor's audit plan. The "
     "Committee noted delays in closing audit findings on fertiliser store stock reconciliation and directed "
     "management to complete corrective actions within 90 days."),
    ("Internal Audit Function",
     "The internal audit function is performed in-house by the Group Internal Audit Department, which reports "
     "functionally to the Audit Committee and administratively to the Group Managing Director. Audits completed "
     "during the year covered estate manuring, chemical spraying, procurement and fixed assets. The total cost "
     "of the internal audit function was RM1.8 million."),
    ("Independent Auditors' Report - Key Audit Matters",
     "Key audit matter 1: impairment assessment of bearer plants, given judgement in yield and price "
     "assumptions. Key audit matter 2: valuation of biological assets (fresh fruit bunches on trees), which "
     "depends on estimated oil extraction rates and forecast CPO prices."),
]


def main(out_dir: str = "data/raw") -> Path:
    out = Path(out_dir) / "DemoPlantation_2025_annual_report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for i, (title, body) in enumerate(PAGES):
        story += [Paragraph(title, styles["Heading1"]), Spacer(1, 12), Paragraph(body, styles["BodyText"])]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(out), pagesize=A4, title="Demo Plantation Berhad Annual Report 2025 (fictional)").build(story)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
