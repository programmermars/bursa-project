"""Audit analytics dashboard: streamlit run app.py"""
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import altair as alt  # noqa: E402  (ships with streamlit)
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from auditkit.benford import benford_test  # noqa: E402
from auditkit.checks import CHECKS, run_checks  # noqa: E402
from auditkit.generate import Dataset, generate  # noqa: E402
from auditkit.report import write_report  # noqa: E402
from auditkit.scoring import evaluate  # noqa: E402

st.set_page_config(page_title="Audit Analytics", page_icon="🔎", layout="wide")
TABLES = ["employees", "vendors", "invoices", "journals", "estate_ops", "store"]


@st.cache_data(show_spinner="Generating data and running tests...")
def analyse_synthetic(seed: int):
    ds = generate(seed=seed)
    exc = run_checks(ds)
    return ds, exc, evaluate(ds, exc)


def analyse_uploaded(files):
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            (Path(tmp) / f.name).write_bytes(f.getvalue())
        ds = Dataset.load(Path(tmp))
    exc = run_checks(ds)
    return ds, exc, evaluate(ds, exc)


st.title("Audit Analytics: plantation group")
st.caption("Computer-assisted audit tests on accounts payable, journals, vendor master, estate operations and the "
           "fertiliser store. On synthetic data every irregularity is known, so each test is scored.")

with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Synthetic (fictional company)", "Upload CSV extracts"])
    if source.startswith("Synthetic"):
        seed = st.number_input("Random seed", 0, 9999, 7, help="Change to generate a different year of data")
        ds, exc, ev = analyse_synthetic(int(seed))
    else:
        files = st.file_uploader("CSV files: " + ", ".join(f"{t}.csv" for t in TABLES) + " (truth.csv optional)",
                                 type="csv", accept_multiple_files=True)
        if not files:
            st.info("Upload the CSV extracts. Column layout: see the files in data/ (run `auditkit generate`).")
            st.stop()
        try:
            ds, exc, ev = analyse_uploaded(files)
        except (FileNotFoundError, KeyError, ValueError) as e:
            st.error(f"Could not read the extracts: {e}")
            st.stop()
    has_truth = len(ds.truth) > 0
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_report(ds, exc, ev, Path(tmp) / "exceptions_report.xlsx", with_scoring=has_truth)
        buf.write(path.read_bytes())
    st.download_button("Download Excel working paper", buf.getvalue(), "exceptions_report.xlsx", width="stretch")

ranking = ev["ranking"]
c = st.columns(4)
c[0].metric("Invoices tested", f"{len(ds.invoices):,}")
c[1].metric("Journal entries tested", f"{len(ds.journals):,}")
c[2].metric("Records flagged", f"{len(ranking):,}", f"{len(ranking) / (len(ds.invoices) + len(ds.journals)):.1%} of AP + GL", delta_color="off", delta_arrow="off")
if has_truth:
    found = ev["coverage"].found.sum() / ev["coverage"].planted.sum()
    c[3].metric("Planted irregularities found", f"{found:.0%}", f"top 50 precision {ev['precision_at'].get(50, 0):.0%}", delta_color="off", delta_arrow="off")

tabs = st.tabs(["Risk ranking", "Tests", "Benford's law", "Operations", "About the tests"])

with tabs[0]:
    st.subheader("Where to look first")
    st.caption("Each flagged record scores the sum of its tests' risk weights (tests that measure the same signal count once); records failing several independent tests come first.")
    areas = ["All"] + sorted(exc.area.unique())
    area = st.selectbox("Area", areas)
    view = ranking if area == "All" else ranking[ranking.record_id.isin(exc[exc.area == area].record_id)]
    if has_truth:
        bad = set(zip(ds.truth.table, ds.truth.record_id))
        view = view.assign(planted=[(t, r) in bad for t, r in zip(view.table, view.record_id)])
    st.dataframe(view.head(500), width="stretch", hide_index=True,
                 column_config={"reasons": st.column_config.TextColumn(width="large")})
    rid = st.selectbox("Inspect a record", view.record_id.head(500))
    if rid:
        tbl = view[view.record_id == rid].table.iloc[0]
        key = {"invoices": "invoice_id", "journals": "je_id", "vendors": "vendor_id", "estate_ops": "op_id", "store": "store_id"}[tbl]
        st.dataframe(getattr(ds, tbl)[getattr(ds, tbl)[key] == rid], width="stretch", hide_index=True)
        st.dataframe(exc[exc.record_id == rid][["test", "reason"]], width="stretch", hide_index=True)

with tabs[1]:
    st.subheader("Exceptions per test")
    if has_truth:
        bt = ev["by_test"]
        st.dataframe(bt[["test", "area", "objective", "flagged", "planted", "found", "precision", "recall"]],
                     width="stretch", hide_index=True,
                     column_config={"precision": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                    "recall": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        st.caption("Precision: flagged records that are the planted problem. Recall: planted problems that were flagged. "
                   "Low precision is not a failure by itself: most extra flags are legitimate exceptions (year-end closing, "
                   "rain delays) that the auditor clears with management.")
        st.dataframe(ev["coverage"], width="stretch", hide_index=True)
    else:
        st.dataframe(exc.groupby(["area", "test"]).record_id.nunique().rename("flagged").reset_index(),
                     width="stretch", hide_index=True)

with tabs[2]:
    st.subheader("Benford's law: manual journal entries, first two digits")
    je = ds.journals
    pop = st.radio("Population", ["Manual journals", "All journals", "Invoices"], horizontal=True)
    amounts = je[je.source == "manual"].amount if pop == "Manual journals" else je.amount if pop == "All journals" else ds.invoices.amount
    res = benford_test(amounts)
    st.markdown(f"**MAD = {res.mad:.4f} → {res.conclusion}** (n = {res.n:,}; Nigrini thresholds 0.0012 / 0.0018 / 0.0022). "
                f"Chi-square p = {res.p_value:.3g}.")
    t = res.table.reset_index(names="digits")
    t["spike"] = (t.z > 3) & (t.observed > t.expected)
    bars = alt.Chart(t).mark_bar().encode(
        x=alt.X("digits:O", title="First two digits", axis=alt.Axis(values=list(range(10, 100, 5)))),
        y=alt.Y("observed:Q", title="Proportion", axis=alt.Axis(format="%")),
        color=alt.condition("datum.spike", alt.value("#c0392b"), alt.value("#5b8db8")),
        tooltip=["digits", alt.Tooltip("observed", format=".3%"), alt.Tooltip("expected", format=".3%"), alt.Tooltip("z", format=".1f")])
    line = alt.Chart(t).mark_line(color="black").encode(x="digits:O", y="expected:Q")
    st.altair_chart(bars + line, width="stretch")
    st.caption("Bars: observed. Line: Benford's expected proportion. Red: significantly over-represented (z > 3). "
               "Spikes at 48–49 point to manual journals kept just under the RM50,000 approval limit.")

with tabs[3]:
    st.subheader("Estate operations and fertiliser store")
    o = ds.estate_ops[ds.estate_ops.planned_qty > 0].assign(variance=lambda d: (d.actual_qty - d.planned_qty) / d.planned_qty)
    o = o.assign(flagged=o.op_id.isin(exc.record_id))
    st.altair_chart(alt.Chart(o).mark_circle(size=50).encode(
        x=alt.X("planned_qty:Q", title="Planned quantity"), y=alt.Y("actual_qty:Q", title="Actual quantity"),
        color=alt.Color("flagged:N", scale=alt.Scale(range=["#5b8db8", "#c0392b"])),
        tooltip=["estate", "month", "activity", "planned_qty", "actual_qty", alt.Tooltip("variance", format="+.0%")]),
        width="stretch")
    s = ds.store.assign(shortage=lambda d: (d.book_closing_kg - d.physical_count_kg) / d.book_closing_kg)
    st.altair_chart(alt.Chart(s).mark_rect().encode(
        x=alt.X("month:O"), y=alt.Y("estate:O"),
        color=alt.Color("shortage:Q", title="Count shortage", scale=alt.Scale(scheme="reds"), legend=alt.Legend(format="%")),
        tooltip=["estate", "month", "book_closing_kg", "physical_count_kg", alt.Tooltip("shortage", format=".1%")]),
        width="stretch")

with tabs[4]:
    st.dataframe(pd.DataFrame([{"Test": c.name, "Area": c.area, "Risk addressed": c.objective, "Risk weight": c.weight}
                               for c in CHECKS]), width="stretch", hide_index=True)
