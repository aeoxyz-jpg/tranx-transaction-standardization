# Route Leaderboard

| route | split | Category Acc | Macro F1 | Merchant Acc | Merchant Norm | Dedup Ratio | Spend KPI | ms/txn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | random | 0.86 | 0.86 | 1.00 | 1.00 | 0.48 | 1.00 | 0.04 |
| embedding | random | 0.98 | 0.98 | 0.99 | 0.99 | 0.48 | 0.99 | 8.36 |
| slm_fewshot | random | 0.75 | 0.77 | 0.77 | 0.81 | 0.51 | 0.77 | 554.58 |
| rules | unseen | 0.42 | 0.44 | 0.94 | 0.94 | 0.19 | 0.94 | 0.08 |
| embedding | unseen | 0.81 | 0.81 | 0.00 | 0.00 | 0.18 | 0.00 | 8.22 |
| slm_fewshot | unseen | 0.71 | 0.68 | 0.73 | 0.77 | 0.25 | 0.74 | 559.72 |
