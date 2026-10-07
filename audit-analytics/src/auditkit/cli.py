"""Command line.

    auditkit generate            create the synthetic data in data/
    auditkit run                 run all tests, score them, write output/exceptions_report.xlsx and output/results.md
    auditkit run --data my_dir   run on your own CSV extracts (same columns as data/)
    auditkit benchmark --seeds 10   repeat on independently generated years; mean and spread of each test's scores
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .checks import run_checks
from .generate import Dataset, generate
from .report import write_report
from .scoring import evaluate

ROOT = Path(__file__).resolve().parents[2]


def results_markdown(ds: Dataset, ev: dict, exceptions) -> str:
    bt = ev["by_test"]
    pa = ev["precision_at"]
    lines = [
        "# Results", "",
        f"Population: {len(ds.invoices):,} invoices, {len(ds.journals):,} journal entries, {len(ds.vendors):,} vendors, "
        f"{len(ds.estate_ops):,} estate activities, {len(ds.store):,} store months. "
        f"Planted irregularities: {len(ds.truth):,} records of {ds.truth.anomaly.nunique()} types.", "",
        f"Flagged: {ev['ranking'].shape[0]:,} unique records. Of the planted irregularities, "
        f"{ev['coverage'].found.sum() / ev['coverage'].planted.sum():.0%} were flagged by at least one test.", "",
        "Risk ranking (records sorted by combined test weight): "
        + ", ".join(f"precision of top {n} = {v:.0%}" for n, v in pa.items()) + ".", "",
        "| Test | Area | Flagged | Planted | Found | Precision | Recall |", "|---|---|---|---|---|---|---|",
    ]
    fmt = lambda v: "n/a" if v is None or v != v else f"{v:.2f}"  # noqa: E731
    for r in bt.itertuples():
        lines.append(f"| {r.test} | {r.area} | {r.flagged} | {r.planted} | {r.found} | {fmt(r.precision)} | {fmt(r.recall)} |")
    lines += ["", "Precision: share of flagged records that are the planted irregularity the test targets. "
              "Recall: share of those planted irregularities the test flagged. Flags that are not planted are mostly "
              "legitimate exceptions built into the data (e.g. year-end weekend closing), which an auditor would clear "
              "with management's explanation."]
    return "\n".join(lines) + "\n"


def benchmark(seeds: int, out: Path) -> pd.DataFrame:
    frames = []
    for seed in range(100, 100 + seeds):
        ds = generate(seed=seed)
        ev = evaluate(ds, run_checks(ds))
        frames.append(ev["by_test"].assign(seed=seed, top50=ev["precision_at"].get(50)))
        print(f"seed {seed}: done")
    allr = pd.concat(frames)
    g = allr.groupby("test", sort=False)
    table = pd.DataFrame({"precision_mean": g.precision.mean(), "precision_sd": g.precision.std(),
                          "recall_mean": g.recall.mean(), "recall_min": g.recall.min()}).reset_index()
    lines = ["# Benchmark over independent synthetic years", "",
             f"{seeds} data sets generated with different random seeds; each test is scored on each. "
             f"Precision of the top 50 risk-ranked records: mean {allr.groupby('seed').top50.first().mean():.2f}, "
             f"min {allr.groupby('seed').top50.first().min():.2f}.", "",
             "| Test | Precision (mean ± sd) | Recall (mean) | Recall (worst year) |", "|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.test} | {r.precision_mean:.2f} ± {r.precision_sd:.2f} | {r.recall_mean:.2f} | {r.recall_min:.2f} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return table


def main(argv=None):
    p = argparse.ArgumentParser(prog="auditkit", description="Audit analytics on plantation-company data")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--out", default=str(ROOT / "data")); g.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("run"); r.add_argument("--data", default=str(ROOT / "data")); r.add_argument("--out", default=str(ROOT / "output"))
    b = sub.add_parser("benchmark"); b.add_argument("--seeds", type=int, default=10); b.add_argument("--out", default=str(ROOT / "output"))
    args = p.parse_args(argv)

    if args.cmd == "benchmark":
        benchmark(args.seeds, Path(args.out))
        return

    if args.cmd == "generate":
        ds = generate(seed=args.seed)
        ds.save(Path(args.out))
        print(f"Wrote {len(ds.invoices):,} invoices, {len(ds.journals):,} journals, {len(ds.truth):,} planted irregularities to {args.out}")
        return
    try:
        ds = Dataset.load(Path(args.data))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    exceptions = run_checks(ds)
    ev = evaluate(ds, exceptions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    has_truth = len(ds.truth) > 0
    xlsx = write_report(ds, exceptions, ev, out / "exceptions_report.xlsx", with_scoring=has_truth)
    exceptions.to_csv(out / "exceptions.csv", index=False)
    print(f"{exceptions.test.nunique()} tests raised {len(exceptions):,} exceptions on {len(ev['ranking']):,} records -> {xlsx}")
    if has_truth:
        (out / "results.md").write_text(results_markdown(ds, ev, exceptions), encoding="utf-8")
        print(ev["by_test"][["test", "flagged", "found", "planted", "precision", "recall"]].round(2).to_string(index=False))
        print("Precision of top-ranked records:", {k: round(v, 2) for k, v in ev["precision_at"].items()})


if __name__ == "__main__":
    main()
