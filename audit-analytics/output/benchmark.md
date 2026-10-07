# Benchmark over independent synthetic years

10 data sets generated with different random seeds; each test is scored on each. Precision of the top 50 risk-ranked records: mean 0.99, min 0.98.

| Test | Precision (mean ± sd) | Recall (mean) | Recall (worst year) |
|---|---|---|---|
| Duplicate invoice (exact) | 1.00 ± 0.01 | 1.00 | 1.00 |
| Duplicate invoice (near) | 0.99 ± 0.02 | 1.00 | 1.00 |
| Split purchase below approval limit | 0.43 ± 0.10 | 1.00 | 1.00 |
| Self-approved invoice | 1.00 ± 0.00 | 1.00 | 1.00 |
| Approved above authority limit | 0.75 ± 0.08 | 1.00 | 1.00 |
| Posted on weekend / public holiday | 0.73 ± 0.01 | 1.00 | 1.00 |
| Round-sum amount | 0.43 ± 0.14 | 1.00 | 1.00 |
| Vendor matches an employee | 1.00 ± 0.00 | 1.00 | 1.00 |
| Possible duplicate vendor | 0.64 ± 0.13 | 1.00 | 1.00 |
| Paid after vendor was blocked | 1.00 ± 0.00 | 1.00 | 1.00 |
| Statistical outlier (Isolation Forest) | 0.36 ± 0.05 | 0.12 | 0.09 |
| Manual JE outside office hours | 0.75 ± 0.00 | 1.00 | 1.00 |
| Manual JE by non-GL user | 1.00 ± 0.00 | 1.00 | 1.00 |
| JE amount unusual for account | 0.75 ± 0.07 | 0.78 | 0.53 |
| JE description with red-flag wording | 1.00 ± 0.00 | 1.00 | 1.00 |
| Benford first-two-digit spike | 0.33 ± 0.13 | 0.45 | 0.00 |
| Manual JE just below approval limit | 0.60 ± 0.06 | 1.00 | 1.00 |
| Planned vs actual quantity variance | 1.00 ± 0.00 | 1.00 | 1.00 |
| Activity done late | 0.71 ± 0.00 | 1.00 | 1.00 |
| Activity with no plan | 1.00 ± 0.00 | 1.00 | 1.00 |
| Stock count shortage | 0.71 ± 0.00 | 1.00 | 1.00 |
| Fertiliser issued exceeds field application | 1.00 ± 0.00 | 1.00 | 1.00 |
