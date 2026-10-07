# Results

Population: 9,273 invoices, 14,077 journal entries, 169 vendors, 480 estate activities, 240 store months. Planted irregularities: 458 records of 21 types.

Flagged: 817 unique records. Of the planted irregularities, 99% were flagged by at least one test.

Risk ranking (records sorted by combined test weight): precision of top 25 = 100%, precision of top 50 = 100%, precision of top 100 = 100%.

| Test | Area | Flagged | Planted | Found | Precision | Recall |
|---|---|---|---|---|---|---|
| Duplicate invoice (exact) | Accounts payable | 35 | 35 | 35 | 1.00 | 1.00 |
| Duplicate invoice (near) | Accounts payable | 30 | 30 | 30 | 1.00 | 1.00 |
| Split purchase below approval limit | Accounts payable | 283 | 77 | 77 | 0.27 | 1.00 |
| Self-approved invoice | Accounts payable | 30 | 30 | 30 | 1.00 | 1.00 |
| Approved above authority limit | Accounts payable | 19 | 15 | 15 | 0.79 | 1.00 |
| Posted on weekend / public holiday | Accounts payable | 49 | 35 | 35 | 0.71 | 1.00 |
| Round-sum amount | Accounts payable | 37 | 25 | 25 | 0.68 | 1.00 |
| Vendor matches an employee | Vendor master | 28 | 28 | 28 | 1.00 | 1.00 |
| Possible duplicate vendor | Vendor master | 4 | 3 | 3 | 0.75 | 1.00 |
| Paid after vendor was blocked | Accounts payable | 10 | 10 | 10 | 1.00 | 1.00 |
| Statistical outlier (Isolation Forest) | Accounts payable | 93 | 280 | 28 | 0.30 | 0.10 |
| Manual JE outside office hours | General ledger | 40 | 30 | 30 | 0.75 | 1.00 |
| Manual JE by non-GL user | General ledger | 15 | 15 | 15 | 1.00 | 1.00 |
| JE amount unusual for account | General ledger | 20 | 15 | 12 | 0.60 | 0.80 |
| JE description with red-flag wording | General ledger | 12 | 12 | 12 | 1.00 | 1.00 |
| Benford first-two-digit spike | General ledger | 66 | 40 | 28 | 0.42 | 0.70 |
| Manual JE just below approval limit | General ledger | 70 | 40 | 40 | 0.57 | 1.00 |
| Planned vs actual quantity variance | Estate operations | 15 | 15 | 15 | 1.00 | 1.00 |
| Activity done late | Estate operations | 21 | 15 | 15 | 0.71 | 1.00 |
| Activity with no plan | Estate operations | 6 | 6 | 6 | 1.00 | 1.00 |
| Stock count shortage | Fertiliser store | 17 | 12 | 12 | 0.71 | 1.00 |
| Fertiliser issued exceeds field application | Fertiliser store | 10 | 10 | 10 | 1.00 | 1.00 |

Precision: share of flagged records that are the planted irregularity the test targets. Recall: share of those planted irregularities the test flagged. Flags that are not planted are mostly legitimate exceptions built into the data (e.g. year-end weekend closing), which an auditor would clear with management's explanation.
