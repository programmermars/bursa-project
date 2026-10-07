"""Synthetic data for a fictional Malaysian plantation group, with planted irregularities.

Everything here is invented. The point is that every irregularity is *known*: each one is recorded in
`truth` (table, record_id, anomaly), so each audit test can be scored on what it catches and what it misses.

Tables
  employees    staff master (bank account, address)
  vendors      vendor master (bank account, address, status)
  invoices     accounts payable invoices (amount, creator, approver, posting time)
  journals     general ledger journal entries (manual and system)
  estate_ops   planned vs actual manuring and spraying per estate and month
  store        fertiliser store: book stock, physical count, issues to estates
  truth        planted irregularities

The data also contains *legitimate* exceptions that a good test will still flag (year-end weekend closing,
late-night month-end journals, rain-delayed manuring, moisture loss in fertiliser, two different vendors with
similar names). They are not in `truth`, so they count as false positives, as they would in a real audit.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FY_START, FY_END = date(2024, 10, 1), date(2025, 9, 30)  # financial year ending 30 September

# Malaysian public holidays in FY2025 observed in Perak (approximate list, for the weekend/holiday test)
HOLIDAYS = {date(2024, 10, 31), date(2024, 12, 25), date(2025, 1, 1), date(2025, 1, 29), date(2025, 1, 30),
            date(2025, 3, 18), date(2025, 3, 31), date(2025, 4, 1), date(2025, 5, 1), date(2025, 5, 12),
            date(2025, 6, 2), date(2025, 6, 7), date(2025, 6, 27), date(2025, 8, 31), date(2025, 9, 1),
            date(2025, 9, 5), date(2025, 9, 16)}

APPROVAL_LIMITS = [(5_000, "supervisor"), (50_000, "finance_manager"), (float("inf"), "director")]
JE_APPROVAL_LIMIT = 50_000

USERS = {
    "clerk": ["U01", "U02", "U03", "U04", "U05", "U06"],
    "supervisor": ["U07", "U08", "U09"],
    "finance_manager": ["U10", "U11"],
    "director": ["U12"],
    "gl_accountant": ["U13", "U14", "U15", "U16"],
}
ESTATES = [f"E{n:02d}" for n in range(1, 21)]
GL_ACCOUNTS = {  # account: (typical log-amount mean, sigma)
    "5100 Fertiliser": (9.2, 1.0), "5200 Chemicals": (8.6, 1.0), "5300 Upkeep": (8.0, 1.1),
    "5400 Harvesting wages": (9.8, 0.8), "5500 Transport": (7.8, 1.0), "6100 Admin expenses": (7.2, 1.2),
    "6200 Repairs": (7.6, 1.1), "4100 Revenue - CPO": (12.0, 0.7), "4200 Revenue - PK": (10.8, 0.7),
    "1500 Accruals": (9.0, 1.2),
}
FIRST = ["Ahmad", "Siti", "Tan", "Lim", "Kumar", "Nur", "Wong", "Raj", "Lee", "Aisyah", "Chong", "Mohd", "Devi", "Ong"]
LAST = ["Ali", "Hassan", "Wei Ming", "Kok Leong", "Selvam", "Aina", "Mei Ling", "Kannan", "Hui Min", "Ismail"]
VENDOR_WORDS = ["Agro", "Perak", "Kinta", "Sawit", "Bumi", "Hijau", "Jaya", "Maju", "Delta", "Mega", "Tani",
                "Global", "Prima", "Sinar", "Harta", "Teknik", "Logistik", "Kimia", "Baja", "Motor"]
VENDOR_KINDS = ["Supplies", "Trading", "Enterprise", "Engineering", "Transport", "Services", "Industries", "Agencies"]


@dataclass
class Dataset:
    employees: pd.DataFrame
    vendors: pd.DataFrame
    invoices: pd.DataFrame
    journals: pd.DataFrame
    estate_ops: pd.DataFrame
    store: pd.DataFrame
    truth: pd.DataFrame

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for f in fields(self):
            getattr(self, f.name).to_csv(folder / f"{f.name}.csv", index=False)

    @classmethod
    def load(cls, folder: Path) -> "Dataset":
        parse = {"invoices": ["invoice_date", "posted_at"], "journals": ["posted_at"],
                 "estate_ops": ["planned_date", "actual_date"], "vendors": ["created", "blocked_on"]}
        out = {}
        for f in fields(cls):
            path = folder / f"{f.name}.csv"
            if f.name == "truth" and not path.exists():  # real data has no answer key
                out["truth"] = pd.DataFrame(columns=["table", "record_id", "anomaly"])
                continue
            if not path.exists():
                raise FileNotFoundError(f"{path} missing. Run `auditkit generate` first.")
            out[f.name] = pd.read_csv(path, parse_dates=parse.get(f.name, []),
                                      dtype={"bank_account": str, "invoice_no": str})
        return cls(**out)


class _Gen:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        days = pd.date_range(FY_START, FY_END).date
        self.workdays = [d for d in days if d.weekday() < 5 and d not in HOLIDAYS]
        self.offdays = [d for d in days if d.weekday() >= 5 or d in HOLIDAYS]
        self.truth: list[tuple[str, str, str]] = []

    # helpers -------------------------------------------------------------------------------------
    def pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def office_time(self, d: date) -> datetime:
        minutes = int(self.rng.integers(8 * 60 + 30, 18 * 60))
        return datetime.combine(d, time(minutes // 60, minutes % 60))

    def late_time(self, d: date) -> datetime:
        hour = int(self.pick([22, 23, 0, 1, 2, 3, 4]))
        return datetime.combine(d, time(hour, int(self.rng.integers(60))))

    def mark(self, table: str, record_id: str, anomaly: str) -> None:
        self.truth.append((table, record_id, anomaly))

    def bank(self) -> str:
        return "".join(str(x) for x in self.rng.integers(0, 10, 12))

    def address(self) -> str:
        return f"{int(self.rng.integers(1, 200))}, Jalan {self.pick(VENDOR_WORDS)} {int(self.rng.integers(1, 30))}, " \
               f"{self.pick(['Ipoh', 'Taiping', 'Teluk Intan', 'Kampar', 'Batu Gajah', 'Sitiawan'])}, Perak"

    # masters --------------------------------------------------------------------------------------
    def employees(self, n=80) -> pd.DataFrame:
        return pd.DataFrame([{"employee_id": f"EMP{i:03d}", "name": f"{self.pick(FIRST)} {self.pick(LAST)}",
                              "bank_account": self.bank(), "address": self.address()} for i in range(1, n + 1)])

    def vendors(self, employees: pd.DataFrame, n=160) -> pd.DataFrame:
        names, rows = set(), []
        while len(rows) < n:
            name = f"{self.pick(VENDOR_WORDS)} {self.pick(VENDOR_WORDS)} {self.pick(VENDOR_KINDS)} Sdn Bhd"
            if name in names:
                continue
            names.add(name)
            rows.append({"vendor_id": f"V{len(rows) + 1:04d}", "name": name, "bank_account": self.bank(),
                         "address": self.address(), "status": "active", "created": date(2020, 1, 1)
                         + timedelta(days=int(self.rng.integers(0, 1500))), "blocked_on": pd.NaT})
        v = pd.DataFrame(rows)
        # ghost vendors: bank account or address copied from an employee
        for k in range(4):
            i = len(v)
            emp = employees.iloc[int(self.rng.integers(len(employees)))]
            v.loc[i] = {"vendor_id": f"V{i + 1:04d}", "name": f"{self.pick(VENDOR_WORDS)} {self.pick(VENDOR_KINDS)} Enterprise",
                        "bank_account": emp.bank_account if k % 2 == 0 else self.bank(),
                        "address": emp.address if k % 2 == 1 else self.address(), "status": "active",
                        "created": date(2024, 9, 1) + timedelta(days=int(self.rng.integers(0, 60))), "blocked_on": pd.NaT}
            self.mark("vendors", v.loc[i, "vendor_id"], "ghost_vendor")
        # duplicate vendor records: same supplier entered twice with a slightly different name
        for _ in range(3):
            i = len(v)
            src = v.iloc[int(self.rng.integers(0, 160))]
            twist = self.pick([lambda s: s.replace("Sdn Bhd", "Sdn. Bhd."), lambda s: s.upper(),
                               lambda s: s.replace(" ", "  ", 1), lambda s: s.replace("Sdn Bhd", "S/B")])
            v.loc[i] = {**src.to_dict(), "vendor_id": f"V{i + 1:04d}", "name": twist(src["name"]),
                        "bank_account": src.bank_account, "created": date(2024, 11, 1)}
            self.mark("vendors", v.loc[i, "vendor_id"], "duplicate_vendor")
        # legitimate: two different suppliers whose names are almost the same
        for name in ["Kinta Agro Supplies Sdn Bhd", "Kinta Agri Supplies Sdn Bhd"]:
            i = len(v)
            v.loc[i] = {"vendor_id": f"V{i + 1:04d}", "name": name, "bank_account": self.bank(), "address": self.address(),
                        "status": "active", "created": date(2021, 5, 1), "blocked_on": pd.NaT}
        # blocked vendors (blocked part-way through the year)
        for i in self.rng.choice(160, 5, replace=False):
            v.loc[i, "status"] = "blocked"
            v.loc[i, "blocked_on"] = pd.Timestamp(date(2025, 3, 1))
        v["created"] = pd.to_datetime(v["created"])
        v["blocked_on"] = pd.to_datetime(v["blocked_on"])
        return v

    # invoices ---------------------------------------------------------------------------------------
    def approver_for(self, amount: float) -> str:
        role = next(r for limit, r in APPROVAL_LIMITS if amount < limit)
        return self.pick(USERS[role])

    def invoices(self, vendors: pd.DataFrame, n=9000) -> pd.DataFrame:
        normal = vendors[(vendors.status == "active") & vendors.vendor_id.isin(vendors.vendor_id[:160])]
        weights = self.rng.pareto(1.2, len(normal)) + 0.2  # a few vendors get most invoices
        weights /= weights.sum()
        prefixes = {vid: self.pick(["INV-", "SI", "B/", "IV", "INV"]) for vid in vendors.vendor_id}
        seq = {vid: int(self.rng.integers(100, 5000)) for vid in vendors.vendor_id}
        blocked = vendors[vendors.status == "blocked"].vendor_id.tolist()
        rows = []

        def add(vid, d, amount, desc="", creator=None, approver=None, posted=None, estate=None):
            seq[vid] += int(self.rng.integers(1, 4))
            iid = f"AP{len(rows) + 1:06d}"
            rows.append({"invoice_id": iid, "vendor_id": vid, "invoice_no": f"{prefixes[vid]}{seq[vid]:05d}",
                         "invoice_date": d, "amount": round(float(amount), 2),
                         "estate": estate or self.pick(ESTATES), "description": desc or self.pick(
                             ["Fertiliser supply", "Herbicide", "Spare parts", "Lorry hire", "Upkeep contract",
                              "Fuel", "Repair works", "Office supplies", "Harvesting tools", "Road maintenance"]),
                         "created_by": creator or self.pick(USERS["clerk"]),
                         "approved_by": approver or self.approver_for(amount),
                         "posted_at": posted or self.office_time(self.pick([d2 for d2 in self.workdays[:]
                                                                             if d2 >= d][:5] or [d]))})
            return rows[-1]

        # recurring fixed monthly charges (rent, contracts): legitimate repeats, round amounts
        recurring = [(normal.vendor_id.iloc[i], float(self.pick([3_500, 8_000, 12_000, 4_250.50])))
                     for i in self.rng.choice(len(normal), 6, replace=False)]
        for m in range(12):
            month_start = date(2024 + (9 + m) // 12, (9 + m) % 12 + 1, 1)
            for vid, amt in recurring:
                add(vid, month_start + timedelta(days=int(self.rng.integers(0, 3))), amt, "Monthly rental / contract")
        # ordinary invoices: log-normal amounts (spread over several orders of magnitude, close to Benford)
        for _ in range(n):
            vid = normal.vendor_id.iloc[int(self.rng.choice(len(normal), p=weights))]
            d = self.pick(self.workdays)
            for vb in blocked:  # blocked vendors are paid only before they were blocked
                if vid == vb and d >= date(2025, 3, 1):
                    d = d - timedelta(days=180)
            add(vid, d, min(np.exp(self.rng.normal(7.9, 1.35)) + 20, 900_000))

        inv = pd.DataFrame(rows)
        base = len(inv)

        def clone(src, **changes):
            new = {**src, **changes, "invoice_id": f"AP{len(rows) + 1:06d}"}
            rows.append(new)
            return new

        src_pool = inv.sample(400, random_state=int(self.rng.integers(1e9))).to_dict("records")
        it = iter(src_pool)
        # 1. exact duplicates: same vendor, invoice number and amount entered again later
        for _ in range(35):
            s = next(it)
            d = s["invoice_date"] + timedelta(days=int(self.rng.integers(3, 40)))
            r = clone(s, invoice_date=d, posted_at=self.office_time(self.pick([w for w in self.workdays if w >= d][:3] or [d])),
                      created_by=self.pick(USERS["clerk"]))
            self.mark("invoices", r["invoice_id"], "duplicate_exact")
        # 2. near duplicates: invoice number re-typed with a formatting change, or new number within days
        for k in range(30):
            s = next(it)
            no = s["invoice_no"]
            if k % 2 == 0:
                no = self.pick([no.replace("-", ""), no.replace("/", " "), no.lower(), no[:-5] + "0" + no[-5:],
                                no + " "])
                if no == s["invoice_no"]:
                    no = no + "."
            else:
                no = no[:-1] + str((int(no[-1]) + 1) % 10)
            d = s["invoice_date"] + timedelta(days=int(self.rng.integers(1, 12)))
            r = clone(s, invoice_no=no, invoice_date=d, posted_at=self.office_time(
                self.pick([w for w in self.workdays if w >= d][:3] or [d])))
            self.mark("invoices", r["invoice_id"], "duplicate_near")
        # 3. split purchases: one purchase cut into 2-4 invoices below the RM5,000 approval limit
        for _ in range(25):
            vid = normal.vendor_id.iloc[int(self.rng.integers(len(normal)))]
            d = self.pick(self.workdays)
            parts, estate = int(self.rng.integers(2, 5)), self.pick(ESTATES)
            for _ in range(parts):
                r = add(vid, d + timedelta(days=int(self.rng.integers(0, 3))), self.rng.uniform(3_600, 4_995),
                        "Fertiliser supply", estate=estate)
                self.mark("invoices", r["invoice_id"], "split_purchase")
        # 4. segregation of duties: creator approves own invoice
        for idx in self.rng.choice(base, 30, replace=False):
            u = self.pick(USERS["supervisor"] + USERS["finance_manager"])
            rows[idx]["created_by"] = rows[idx]["approved_by"] = u
            self.mark("invoices", rows[idx]["invoice_id"], "sod_self_approval")
        # 5. approval above authority: supervisor approves > RM5,000
        big = [i for i in range(base) if 8_000 < rows[i]["amount"] < 50_000 and rows[i]["created_by"] != rows[i]["approved_by"]]
        for idx in self.rng.choice(big, 15, replace=False):
            rows[idx]["approved_by"] = self.pick(USERS["supervisor"])
            self.mark("invoices", rows[idx]["invoice_id"], "approval_limit_breach")
        # 6. posted on weekend / public holiday
        for idx in self.rng.choice(base, 35, replace=False):
            rows[idx]["posted_at"] = self.office_time(self.pick(self.offdays))
            self.mark("invoices", rows[idx]["invoice_id"], "off_day_posting")
        # 7. round-sum fabricated invoices
        for _ in range(25):
            vid = normal.vendor_id.iloc[int(self.rng.integers(len(normal)))]
            r = add(vid, self.pick(self.workdays), float(self.pick([2_000, 3_000, 4_000, 10_000, 15_000, 20_000, 25_000])),
                    "Consultancy / services")
            self.mark("invoices", r["invoice_id"], "round_amount")
        # 8. payments to ghost vendors
        for vid in vendors.vendor_id[vendors.vendor_id.isin([t[1] for t in self.truth if t[2] == "ghost_vendor"])]:
            for _ in range(int(self.rng.integers(4, 9))):
                r = add(vid, self.pick(self.workdays), self.rng.uniform(1_500, 9_000), "Services rendered")
                self.mark("invoices", r["invoice_id"], "ghost_vendor_payment")
        # 9. blocked vendor paid after being blocked
        for vb in blocked:
            for _ in range(2):
                r = add(vb, self.pick([w for w in self.workdays if w >= date(2025, 3, 15)]),
                        np.exp(self.rng.normal(7.9, 1.0)), "Spare parts")
                self.mark("invoices", r["invoice_id"], "blocked_vendor_paid")

        # legitimate: invoices posted on Saturday 27 Sep 2025 during the authorised year-end closing
        for idx in self.rng.choice(base, 12, replace=False):
            if rows[idx]["invoice_date"] <= date(2025, 9, 27):
                rows[idx]["posted_at"] = self.office_time(date(2025, 9, 27))

        inv = pd.DataFrame(rows).sort_values("invoice_date", kind="stable").reset_index(drop=True)
        inv["invoice_date"] = pd.to_datetime(inv["invoice_date"])
        inv["posted_at"] = pd.to_datetime(inv["posted_at"])
        return inv

    # journals -----------------------------------------------------------------------------------------
    def journals(self, n=14000) -> pd.DataFrame:
        accounts = list(GL_ACCOUNTS)
        rows = []

        def add(account, amount, manual, user=None, posted=None, desc=None):
            d = self.pick(self.workdays)
            rows.append({"je_id": f"JE{len(rows) + 1:06d}", "account": account, "amount": round(float(amount), 2),
                         "source": "manual" if manual else "system",
                         "posted_by": user or (self.pick(USERS["gl_accountant"]) if manual else "SYS"),
                         "posted_at": posted or (self.office_time(d) if manual else datetime.combine(d, time(23, 30))),
                         "description": desc or self.pick(
                             ["Monthly accrual", "Reclass to correct cost centre", "Payroll posting",
                              "Depreciation", "Sales invoice batch", "Inventory adjustment", "Prepayment release",
                              "Intercompany recharge"])})
            return rows[-1]

        for _ in range(n):
            acc = self.pick(accounts)
            mu, sd = GL_ACCOUNTS[acc]
            add(acc, np.exp(self.rng.normal(mu, sd)) + 10, manual=self.rng.random() < 0.3)
        base = len(rows)
        # legitimate: month-end close journals posted late on the last working day of the month
        month_ends = sorted({max(d for d in self.workdays if (d.year, d.month) == ym)
                             for ym in {(d.year, d.month) for d in self.workdays}})
        for _ in range(10):
            add(self.pick(["1500 Accruals", "5100 Fertiliser"]), np.exp(self.rng.normal(9, 1)), True,
                posted=datetime.combine(self.pick(month_ends), time(int(self.pick([21, 22])), int(self.rng.integers(60)))),
                desc="Month-end accrual")
        base = len(rows)
        # manual JEs posted late at night or on off days
        for idx in self.rng.choice([i for i in range(base) if rows[i]["source"] == "manual"], 30, replace=False):
            d = self.pick(self.offdays) if self.rng.random() < 0.5 else self.pick(self.workdays)
            rows[idx]["posted_at"] = self.late_time(d) if d in self.workdays else self.office_time(d)
            self.mark("journals", rows[idx]["je_id"], "je_off_hours")
        # manual JEs by users who do not normally post journals
        for _ in range(15):
            r = add(self.pick(["4100 Revenue - CPO", "1500 Accruals", "6100 Admin expenses"]),
                    np.exp(self.rng.normal(9.5, 1.0)), True, user=self.pick(USERS["finance_manager"] + USERS["clerk"]))
            self.mark("journals", r["je_id"], "je_unusual_user")
        # amounts far outside the account's usual range
        for idx in self.rng.choice(base, 15, replace=False):
            rows[idx]["amount"] = round(rows[idx]["amount"] * float(self.rng.uniform(30, 100)), 2)
            self.mark("journals", rows[idx]["je_id"], "je_outlier_amount")
        # manual JEs kept just below the RM50,000 JE approval limit (shows up in Benford first-two digits 48/49)
        for _ in range(40):
            r = add(self.pick(["5300 Upkeep", "6200 Repairs", "1500 Accruals"]), self.rng.uniform(48_000, 49_990), True,
                    desc=self.pick(["Monthly accrual", "Reclass to correct cost centre", "Inventory adjustment"]))
            self.mark("journals", r["je_id"], "je_below_limit")
        # suspicious wording
        for _ in range(12):
            r = add(self.pick(accounts), np.exp(self.rng.normal(9, 1)), True,
                    desc=self.pick(["Adjustment per CFO instruction", "Plug to balance", "Write back - do not reverse",
                                    "Correction, see me", "Year end adjustment per MD"]))
            self.mark("journals", r["je_id"], "je_suspicious_text")
        je = pd.DataFrame(rows)
        je["posted_at"] = pd.to_datetime(je["posted_at"])
        return je

    # operations ---------------------------------------------------------------------------------------
    def estate_ops(self) -> pd.DataFrame:
        rows = []
        for e in ESTATES:
            hectares = float(self.rng.uniform(800, 3500))
            for m in range(12):
                month = date(2024 + (9 + m) // 12, (9 + m) % 12 + 1, 1)
                for activity, rate in (("manuring", 1.6), ("spraying", 0.9)):  # kg (or litres) per hectare per month
                    planned = round(hectares * rate * float(self.rng.uniform(0.8, 1.2)), 0) \
                        if (activity == "spraying" or m % 2 == 0) else 0.0  # manuring every second month
                    actual = round(planned * float(self.rng.normal(1.0, 0.05)), 0) if planned else 0.0
                    pdate = month + timedelta(days=int(self.rng.integers(0, 20)))
                    adate = pdate + timedelta(days=int(self.rng.integers(-3, 15))) if actual else pd.NaT
                    rows.append({"op_id": f"OP{len(rows) + 1:05d}", "estate": e, "month": month.strftime("%Y-%m"),
                                 "activity": activity, "planned_qty": planned, "actual_qty": actual,
                                 "planned_date": pdate, "actual_date": adate})
        ops = pd.DataFrame(rows)
        idx_plan = ops.index[ops.planned_qty > 0].to_numpy()
        chosen = self.rng.permutation(idx_plan)
        for i in chosen[:15]:  # actual far from plan
            ops.loc[i, "actual_qty"] = round(ops.loc[i, "planned_qty"] * float(self.pick([0.55, 0.65, 1.35, 1.5])), 0)
            self.mark("estate_ops", ops.loc[i, "op_id"], "ops_quantity_variance")
        for i in chosen[15:30]:  # done much later than planned
            ops.loc[i, "actual_date"] = ops.loc[i, "planned_date"] + timedelta(days=int(self.rng.integers(35, 80)))
            self.mark("estate_ops", ops.loc[i, "op_id"], "ops_delay")
        unplanned = self.rng.permutation(ops.index[ops.planned_qty == 0].to_numpy())[:6]
        for i in unplanned:  # activity recorded with no plan
            ops.loc[i, "actual_qty"] = round(float(self.rng.uniform(1500, 4000)), 0)
            ops.loc[i, "actual_date"] = pd.Timestamp(ops.loc[i, "month"] + "-15")
            self.mark("estate_ops", ops.loc[i, "op_id"], "ops_unplanned")
        for i in chosen[30:36]:  # legitimate: manuring postponed by heavy rain (still a valid exception to explain)
            ops.loc[i, "actual_date"] = ops.loc[i, "planned_date"] + timedelta(days=int(self.rng.integers(31, 40)))
        ops["planned_date"] = pd.to_datetime(ops["planned_date"])
        ops["actual_date"] = pd.to_datetime(ops["actual_date"])
        return ops

    def store(self, ops: pd.DataFrame) -> pd.DataFrame:
        manuring = ops[ops.activity == "manuring"].set_index(["estate", "month"]).actual_qty
        rows = []
        for e in ESTATES:
            stock = float(self.rng.uniform(5_000, 15_000))
            for month in sorted(ops.month.unique()):
                need = float(manuring.get((e, month), 0.0))
                issued = round(need * float(self.rng.uniform(0.99, 1.02)), 0)
                receipts = round(max(issued * float(self.rng.uniform(0.9, 1.2)), 0) + float(self.rng.uniform(0, 500)), 0)
                closing = stock + receipts - issued
                count = round(closing * float(self.rng.uniform(0.993, 1.002)), 0)  # small normal shrinkage
                rows.append({"store_id": f"ST{len(rows) + 1:04d}", "estate": e, "month": month,
                             "opening_kg": round(stock, 0), "receipts_kg": receipts, "issued_kg": issued,
                             "manuring_applied_kg": need, "book_closing_kg": round(closing, 0), "physical_count_kg": count})
                stock = count  # stock is reset to the count after each stock take
        st = pd.DataFrame(rows)
        ch = self.rng.permutation(len(st))
        for i in ch[:12]:  # stock missing at count
            st.loc[i, "physical_count_kg"] = round(st.loc[i, "book_closing_kg"] * float(self.rng.uniform(0.85, 0.95)), 0)
            self.mark("store", st.loc[i, "store_id"], "store_shortage")
        for i in ch[12:17]:  # legitimate: moisture loss / bag breakage slightly above tolerance
            st.loc[i, "physical_count_kg"] = round(st.loc[i, "book_closing_kg"] * float(self.rng.uniform(0.972, 0.979)), 0)
        ch = ch[17:]
        for i in [j for j in ch if st.loc[j, "manuring_applied_kg"] > 0][:10]:  # issued more than applied
            extra = round(st.loc[i, "manuring_applied_kg"] * float(self.rng.uniform(0.12, 0.3)), 0)
            st.loc[i, "issued_kg"] += extra
            st.loc[i, "book_closing_kg"] -= extra
            st.loc[i, "physical_count_kg"] -= extra
            self.mark("store", st.loc[i, "store_id"], "store_over_issue")
        return st


def generate(seed: int = 7, n_invoices: int = 9000, n_journals: int = 14000) -> Dataset:
    if n_invoices < 1000 or n_journals < 500:
        raise ValueError("Use at least 1,000 invoices and 500 journals so the irregularities can be planted.")
    g = _Gen(seed)
    emp = g.employees()
    ven = g.vendors(emp)
    inv = g.invoices(ven, n_invoices)
    je = g.journals(n_journals)
    ops = g.estate_ops()
    st = g.store(ops)
    truth = pd.DataFrame(g.truth, columns=["table", "record_id", "anomaly"])
    return Dataset(emp, ven, inv, je, ops, st, truth)
