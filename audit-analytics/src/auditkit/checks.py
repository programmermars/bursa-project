"""Audit tests. Each test returns exceptions: one row per flagged record with a plain-language reason.

The tests are the standard computer-assisted audit techniques (CAATs) used in accounts payable, journal entry
and operational audits; thresholds are parameters so an auditor can tune them to the client's policies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

import numpy as np
import pandas as pd

from .benford import first_two_digits, benford_test
from .generate import APPROVAL_LIMITS, HOLIDAYS, JE_APPROVAL_LIMIT, USERS, Dataset

EXC_COLS = ["table", "record_id", "test", "reason"]


@dataclass
class Check:
    name: str
    area: str
    objective: str          # what risk the test addresses, in audit language
    targets: tuple          # planted anomaly types this test is meant to find (for scoring only)
    weight: float           # contribution to a record's risk score
    fn: Callable[[Dataset], pd.DataFrame]
    group: str = ""         # tests measuring the same signal share a group and are not double-counted

    def __post_init__(self):
        self.group = self.group or self.name


def _exc(table: str, ids, test: str, reasons) -> pd.DataFrame:
    ids = list(ids)
    reasons = list(reasons) if not isinstance(reasons, str) else [reasons] * len(ids)
    return pd.DataFrame({"table": table, "record_id": ids, "test": test, "reason": reasons}, columns=EXC_COLS)


def _norm_invoice_no(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str.replace(r"(?<=[A-Z])0+", "", regex=True)


def _norm_name(s: str) -> str:
    s = s.upper().replace("S/B", "SDN BHD")
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\b(SDN BHD|BHD|ENTERPRISE|TRADING)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_off_day(ts: pd.Series) -> pd.Series:
    return (ts.dt.weekday >= 5) | ts.dt.date.isin(HOLIDAYS)


# accounts payable -----------------------------------------------------------------------------------
def duplicate_exact(ds: Dataset) -> pd.DataFrame:
    inv = ds.invoices
    dup = inv[inv.duplicated(["vendor_id", "invoice_no", "amount"], keep="first")]
    first = inv.drop_duplicates(["vendor_id", "invoice_no", "amount"]).set_index(["vendor_id", "invoice_no", "amount"]).invoice_id
    reasons = [f"Same vendor, invoice no. and amount as {first[(r.vendor_id, r.invoice_no, r.amount)]}"
               for r in dup.itertuples()]
    return _exc("invoices", dup.invoice_id, "Duplicate invoice (exact)", reasons)


def duplicate_near(ds: Dataset, days: int = 14) -> pd.DataFrame:
    inv = ds.invoices.assign(norm=_norm_invoice_no(ds.invoices.invoice_no))
    exact = set(duplicate_exact(ds).record_id)
    out = []
    for _, g in inv.groupby(["vendor_id", "amount"]):
        if len(g) < 2:
            continue
        g = g.sort_values("invoice_date")
        rows = list(g.itertuples())
        for i in range(1, len(rows)):
            r = rows[i]
            if r.invoice_id in exact:
                continue
            for p in rows[:i]:
                same_norm = p.norm == r.norm and p.invoice_no != r.invoice_no
                close = abs((r.invoice_date - p.invoice_date).days) <= days and p.invoice_no != r.invoice_no
                if same_norm or close:
                    why = "invoice no. differs only in formatting" if same_norm else f"within {days} days"
                    out.append((r.invoice_id, f"Same vendor and amount as {p.invoice_id}, {why}"))
                    break
    return _exc("invoices", [o[0] for o in out], "Duplicate invoice (near)", [o[1] for o in out])


def split_purchases(ds: Dataset, limit: float = 5_000, window_days: int = 3, floor: float = 0.6) -> pd.DataFrame:
    inv = ds.invoices
    cand = inv[(inv.amount >= floor * limit) & (inv.amount < limit)].sort_values("invoice_date")
    flagged = {}
    for (vid, estate), g in cand.groupby(["vendor_id", "estate"]):  # same supplier, same cost centre
        rows = list(g.itertuples())
        for i, r in enumerate(rows):
            group = [x for x in rows if 0 <= (x.invoice_date - r.invoice_date).days <= window_days]
            total = sum(x.amount for x in group)
            if len(group) >= 2 and total >= limit:
                for x in group:
                    flagged.setdefault(x.invoice_id, f"{len(group)} invoices from {vid} for {estate} within {window_days} "
                                                     f"days, each below RM{limit:,.0f}, total RM{total:,.2f}")
    return _exc("invoices", flagged.keys(), "Split purchase below approval limit", flagged.values())


def self_approval(ds: Dataset) -> pd.DataFrame:
    inv = ds.invoices
    hit = inv[inv.created_by == inv.approved_by]
    return _exc("invoices", hit.invoice_id, "Self-approved invoice", [f"Created and approved by {u}" for u in hit.created_by])


def approval_authority(ds: Dataset) -> pd.DataFrame:
    role_of = {u: r for r, us in USERS.items() for u in us}
    rank = {r: i for i, (_, r) in enumerate(APPROVAL_LIMITS)}
    inv = ds.invoices
    needed = inv.amount.map(lambda a: next(r for lim, r in APPROVAL_LIMITS if a < lim))
    have = inv.approved_by.map(role_of)
    bad = inv[have.map(lambda r: rank.get(r, -1)) < needed.map(rank)]
    return _exc("invoices", bad.invoice_id, "Approved above authority limit",
                [f"RM{a:,.2f} needs {n.replace('_', ' ')}, approved by {u} ({role_of.get(u, 'unknown role')})"
                 for a, n, u in zip(bad.amount, needed[bad.index], bad.approved_by)])


def off_day_posting(ds: Dataset) -> pd.DataFrame:
    inv = ds.invoices
    hit = inv[_is_off_day(inv.posted_at)]
    return _exc("invoices", hit.invoice_id, "Posted on weekend / public holiday",
                [f"Posted {t:%a %d %b %Y %H:%M}" for t in hit.posted_at])


def round_amounts(ds: Dataset, unit: float = 1_000) -> pd.DataFrame:
    inv = ds.invoices
    hit = inv[(inv.amount >= unit) & (inv.amount % unit == 0)]
    return _exc("invoices", hit.invoice_id, "Round-sum amount", [f"RM{a:,.0f} is an exact multiple of RM{unit:,.0f}" for a in hit.amount])


def vendor_employee_match(ds: Dataset) -> pd.DataFrame:
    emp = ds.employees
    v = ds.vendors
    by_bank = dict(zip(emp.bank_account, emp.employee_id))
    by_addr = dict(zip(emp.address.str.upper(), emp.employee_id))
    hits = {}
    for r in v.itertuples():
        if r.bank_account in by_bank:
            hits[r.vendor_id] = f"bank account matches employee {by_bank[r.bank_account]}"
        elif r.address.upper() in by_addr:
            hits[r.vendor_id] = f"address matches employee {by_addr[r.address.upper()]}"
    inv = ds.invoices[ds.invoices.vendor_id.isin(hits)]
    return pd.concat([
        _exc("vendors", hits.keys(), "Vendor matches an employee", [f"Vendor {why}" for why in hits.values()]),
        _exc("invoices", inv.invoice_id, "Vendor matches an employee",
             [f"Paid to {vid}: {hits[vid]}" for vid in inv.vendor_id]),
    ], ignore_index=True)


def duplicate_vendors(ds: Dataset, threshold: float = 0.92) -> pd.DataFrame:
    v = ds.vendors.assign(norm=ds.vendors.name.map(_norm_name)).sort_values("vendor_id")
    rows = list(v.itertuples())
    hits = {}
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            same_bank = a.bank_account == b.bank_account
            sim = SequenceMatcher(None, a.norm, b.norm).ratio()
            if sim >= threshold or same_bank:
                why = f"name {sim:.0%} similar to {a.vendor_id} ({a.name})" + (", same bank account" if same_bank else "")
                hits.setdefault(b.vendor_id, why)
    return _exc("vendors", hits.keys(), "Possible duplicate vendor", list(hits.values()))


def blocked_vendor_paid(ds: Dataset) -> pd.DataFrame:
    inv = ds.invoices.merge(ds.vendors[["vendor_id", "status", "blocked_on"]], on="vendor_id")
    hit = inv[(inv.status == "blocked") & (inv.invoice_date >= inv.blocked_on)]
    return _exc("invoices", hit.invoice_id, "Paid after vendor was blocked",
                [f"Vendor {v} blocked on {b:%d %b %Y}" for v, b in zip(hit.vendor_id, hit.blocked_on)])


def invoice_outliers(ds: Dataset, contamination: float = 0.01) -> pd.DataFrame:
    """Isolation Forest over behavioural features; flags the most unusual 1% of invoices."""
    from sklearn.ensemble import IsolationForest

    inv = ds.invoices
    vendor_n = inv.groupby("vendor_id").invoice_id.transform("count")
    vendor_med = inv.groupby("vendor_id").amount.transform("median")
    x = pd.DataFrame({
        "log_amount": np.log1p(inv.amount),
        "amount_vs_vendor": np.log1p(inv.amount) - np.log1p(vendor_med),
        "vendor_rarity": -np.log(vendor_n),
        "off_day": _is_off_day(inv.posted_at).astype(int),
        "round": ((inv.amount % 1000) == 0).astype(int),
        "self_approved": (inv.created_by == inv.approved_by).astype(int),
        "post_lag_days": (inv.posted_at.dt.normalize() - inv.invoice_date).dt.days.clip(-30, 60),
    })
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=0).fit(x)
    score = -model.score_samples(x)
    hit = inv[model.predict(x) == -1]
    s = pd.Series(score, index=inv.index)[hit.index]
    return _exc("invoices", hit.invoice_id, "Statistical outlier (Isolation Forest)",
                [f"Anomaly score {v:.3f} (top {contamination:.0%} of invoices)" for v in s])


# journals -------------------------------------------------------------------------------------------
def je_off_hours(ds: Dataset) -> pd.DataFrame:
    je = ds.journals[ds.journals.source == "manual"]
    h = je.posted_at.dt.hour
    hit = je[_is_off_day(je.posted_at) | (h >= 21) | (h < 6)]
    return _exc("journals", hit.je_id, "Manual JE outside office hours", [f"Posted {t:%a %d %b %Y %H:%M}" for t in hit.posted_at])


def je_unusual_user(ds: Dataset) -> pd.DataFrame:
    je = ds.journals[ds.journals.source == "manual"]
    hit = je[~je.posted_by.isin(USERS["gl_accountant"])]
    return _exc("journals", hit.je_id, "Manual JE by non-GL user", [f"Posted by {u}, not a GL accountant" for u in hit.posted_by])


def je_outliers(ds: Dataset, z: float = 3.5) -> pd.DataFrame:
    je = ds.journals
    la = np.log(je.amount.clip(lower=1))
    med = la.groupby(je.account).transform("median")
    mad = (la - med).abs().groupby(je.account).transform("median") * 1.4826
    rz = (la - med) / mad
    hit = je[rz > z]
    return _exc("journals", hit.je_id, "JE amount unusual for account",
                [f"RM{a:,.0f} is {v:.1f} robust SDs above the {acc} norm" for a, v, acc in zip(hit.amount, rz[hit.index], hit.account)])


def je_keywords(ds: Dataset) -> pd.DataFrame:
    pattern = r"\b(?:plug|per (?:cfo|md|ceo)|see me|do not reverse|instruction|write back)\b"
    je = ds.journals
    hit = je[je.description.str.contains(pattern, case=False, regex=True)]
    return _exc("journals", hit.je_id, "JE description with red-flag wording", [f'"{d}"' for d in hit.description])


def je_benford(ds: Dataset) -> pd.DataFrame:
    """First-two-digit Benford test on manual JEs; drill down into digit pairs that are significantly over-represented."""
    je = ds.journals[(ds.journals.source == "manual") & (ds.journals.amount >= 10)]
    res = benford_test(je.amount)
    spikes = res.table[(res.table.z > 3) & (res.table.observed > res.table.expected)].index
    d = first_two_digits(je.amount)
    hit = je[d.isin(spikes)]
    return _exc("journals", hit.je_id, "Benford first-two-digit spike",
                [f"Starts with {int(x)}: digit pair over-represented (MAD {res.mad:.4f}, {res.conclusion})" for x in d[hit.index]])


def je_below_limit(ds: Dataset, limit: float = JE_APPROVAL_LIMIT, band: float = 0.05) -> pd.DataFrame:
    """Targeted follow-up to the Benford screen: manual JEs within 5% below the JE approval limit."""
    je = ds.journals[ds.journals.source == "manual"]
    hit = je[(je.amount >= limit * (1 - band)) & (je.amount < limit)]
    return _exc("journals", hit.je_id, "Manual JE just below approval limit",
                [f"RM{a:,.2f} is within {band:.0%} below the RM{limit:,.0f} approval limit" for a in hit.amount])


# operations -----------------------------------------------------------------------------------------
def ops_variance(ds: Dataset, tol: float = 0.2) -> pd.DataFrame:
    o = ds.estate_ops[ds.estate_ops.planned_qty > 0]
    var = (o.actual_qty - o.planned_qty) / o.planned_qty
    hit = o[var.abs() > tol]
    return _exc("estate_ops", hit.op_id, "Planned vs actual quantity variance",
                [f"{e} {m} {a}: actual {ac:,.0f} vs plan {p:,.0f} ({v:+.0%})" for e, m, a, ac, p, v in
                 zip(hit.estate, hit.month, hit.activity, hit.actual_qty, hit.planned_qty, var[hit.index])])


def ops_delay(ds: Dataset, days: int = 30) -> pd.DataFrame:
    o = ds.estate_ops.dropna(subset=["actual_date"])
    late = (o.actual_date - o.planned_date).dt.days
    hit = o[late > days]
    return _exc("estate_ops", hit.op_id, "Activity done late",
                [f"{e} {a} {d} days after plan" for e, a, d in zip(hit.estate, hit.activity, late[hit.index])])


def ops_unplanned(ds: Dataset) -> pd.DataFrame:
    o = ds.estate_ops
    hit = o[(o.planned_qty == 0) & (o.actual_qty > 0)]
    return _exc("estate_ops", hit.op_id, "Activity with no plan",
                [f"{e} {m} {a}: {q:,.0f} recorded, nothing planned" for e, m, a, q in zip(hit.estate, hit.month, hit.activity, hit.actual_qty)])


def store_shortage(ds: Dataset, tol: float = 0.02) -> pd.DataFrame:
    s = ds.store
    short = (s.book_closing_kg - s.physical_count_kg) / s.book_closing_kg.replace(0, np.nan)
    hit = s[short > tol]
    return _exc("store", hit.store_id, "Stock count shortage",
                [f"{e} {m}: count {c:,.0f} kg vs book {b:,.0f} kg ({v:.1%} short)" for e, m, c, b, v in
                 zip(hit.estate, hit.month, hit.physical_count_kg, hit.book_closing_kg, short[hit.index])])


def store_over_issue(ds: Dataset, tol: float = 0.05) -> pd.DataFrame:
    s = ds.store[ds.store.manuring_applied_kg > 0]
    over = (s.issued_kg - s.manuring_applied_kg) / s.manuring_applied_kg
    hit = s[over > tol]
    return _exc("store", hit.store_id, "Fertiliser issued exceeds field application",
                [f"{e} {m}: issued {i:,.0f} kg, applied {a:,.0f} kg ({v:+.0%})" for e, m, i, a, v in
                 zip(hit.estate, hit.month, hit.issued_kg, hit.manuring_applied_kg, over[hit.index])])


CHECKS = [
    Check("Duplicate invoice (exact)", "Accounts payable", "Duplicate payment", ("duplicate_exact",), 3, duplicate_exact),
    Check("Duplicate invoice (near)", "Accounts payable", "Duplicate payment disguised by re-typing", ("duplicate_near",), 3, duplicate_near),
    Check("Split purchase below approval limit", "Accounts payable", "Circumventing approval limits", ("split_purchase",), 2, split_purchases),
    Check("Self-approved invoice", "Accounts payable", "Segregation of duties", ("sod_self_approval",), 3, self_approval),
    Check("Approved above authority limit", "Accounts payable", "Limits of authority", ("approval_limit_breach",), 3, approval_authority),
    Check("Posted on weekend / public holiday", "Accounts payable", "Unusual timing", ("off_day_posting",), 1, off_day_posting),
    Check("Round-sum amount", "Accounts payable", "Fabricated or estimated invoices", ("round_amount",), 1, round_amounts),
    Check("Vendor matches an employee", "Vendor master", "Fictitious (ghost) vendor", ("ghost_vendor", "ghost_vendor_payment"), 4, vendor_employee_match),
    Check("Possible duplicate vendor", "Vendor master", "Duplicate vendor records enable duplicate payments", ("duplicate_vendor",), 2, duplicate_vendors),
    Check("Paid after vendor was blocked", "Accounts payable", "Payments to blocked vendors", ("blocked_vendor_paid",), 3, blocked_vendor_paid),
    Check("Statistical outlier (Isolation Forest)", "Accounts payable", "Unknown unusual patterns",
          ("duplicate_exact", "duplicate_near", "split_purchase", "sod_self_approval", "approval_limit_breach", "off_day_posting",
           "round_amount", "ghost_vendor_payment", "blocked_vendor_paid"), 1, invoice_outliers),
    Check("Manual JE outside office hours", "General ledger", "Management override", ("je_off_hours",), 2, je_off_hours),
    Check("Manual JE by non-GL user", "General ledger", "Access control / management override", ("je_unusual_user",), 3, je_unusual_user),
    Check("JE amount unusual for account", "General ledger", "Material misstatement", ("je_outlier_amount",), 2, je_outliers),
    Check("JE description with red-flag wording", "General ledger", "Management override", ("je_suspicious_text",), 2, je_keywords),
    Check("Benford first-two-digit spike", "General ledger", "Fabricated amounts / threshold avoidance", ("je_below_limit",), 1, je_benford, "threshold avoidance (JE)"),
    Check("Manual JE just below approval limit", "General ledger", "Threshold avoidance", ("je_below_limit",), 2, je_below_limit, "threshold avoidance (JE)"),
    Check("Planned vs actual quantity variance", "Estate operations", "Inputs not applied as planned", ("ops_quantity_variance",), 2, ops_variance),
    Check("Activity done late", "Estate operations", "Agronomic timing not followed", ("ops_delay",), 1, ops_delay),
    Check("Activity with no plan", "Estate operations", "Unauthorised activity", ("ops_unplanned",), 2, ops_unplanned),
    Check("Stock count shortage", "Fertiliser store", "Inventory loss / theft", ("store_shortage",), 3, store_shortage),
    Check("Fertiliser issued exceeds field application", "Fertiliser store", "Diversion of inputs", ("store_over_issue",), 3, store_over_issue),
]


def run_checks(ds: Dataset, checks: list[Check] = CHECKS) -> pd.DataFrame:
    parts = []
    for c in checks:
        e = c.fn(ds)
        if len(e):
            parts.append(e.assign(area=c.area, weight=c.weight, group=c.group))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EXC_COLS + ["area", "weight", "group"])
