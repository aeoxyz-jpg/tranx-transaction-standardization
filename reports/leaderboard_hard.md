# Route Leaderboard

Model view: eval rows whose description is verbatim in train are removed, the rest deduplicated to one row per descriptor and capped (cache-miss accuracy; not comparable to earlier leaderboards). Ideal-cache view: cache hits take the train gold, the model view's descriptors take the route's prediction broadcast per descriptor (rules' per-row MCC category is approximated). Spend KPI: ideal-cache view only.

| route | split | view | Category Acc | Cat ±95% | Macro F1 | Merchant Acc | Merchant Norm | Merch ±95% | No-Match Rate | Retrieval Recall | Dedup Ratio | Gold Dedup | Spend KPI | ms/txn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | random | model | 0.74 | 0.03 | 0.75 | 0.67 | 0.68 | 0.04 | 0.31 | - | 0.66 | 0.45 | - | 0.24 |
| embedding | random | model | 0.79 | 0.02 | 0.79 | 0.84 | 0.84 | 0.03 | - | - | 0.46 | 0.45 | - | 6.45 |
| slm_fewshot | random | model | 0.70 | 0.04 | 0.70 | 0.50 | 0.76 | 0.04 | - | - | 0.63 | 0.45 | - | 460.32 |
| cleaner | random | model | - | - | - | 0.04 | 0.18 | 0.03 | - | - | 0.99 | 0.45 | - | 0.01 |
| metadata | random | model | 0.69 | 0.03 | 0.70 | - | - | - | - | - | - | - | - | 0.12 |
| rules | random | ideal_cache | 0.97 | 0.01 | 0.97 | 0.97 | 0.97 | 0.00 | - | - | 0.26 | 0.19 | 0.97 | 0.24 |
| embedding | random | ideal_cache | 0.98 | 0.01 | 0.98 | 0.98 | 0.98 | 0.00 | - | - | 0.19 | 0.19 | 0.98 | 6.45 |
| slm_fewshot | random | ideal_cache | 0.97 | 0.01 | 0.97 | 0.95 | 0.98 | 0.00 | - | - | 0.27 | 0.19 | 0.98 | 460.32 |
| cleaner | random | ideal_cache | - | - | - | 0.92 | 0.93 | 0.01 | - | - | 0.38 | 0.19 | 0.93 | 0.01 |
| metadata | random | ideal_cache | 0.97 | 0.01 | 0.97 | - | - | - | - | - | - | - | - | 0.12 |
| rules | unseen | model | 0.50 | 0.04 | 0.56 | 0.04 | 0.19 | 0.02 | 0.94 | - | 0.85 | 0.11 | - | 0.30 |
| embedding | unseen | model | 0.55 | 0.06 | 0.57 | 0.00 | 0.00 | 0.00 | - | - | 0.19 | 0.11 | - | 6.39 |
| slm_fewshot | unseen | model | 0.74 | 0.05 | 0.71 | 0.49 | 0.76 | 0.06 | - | - | 0.25 | 0.11 | - | 513.02 |
| cleaner | unseen | model | - | - | - | 0.05 | 0.21 | 0.02 | - | - | 0.89 | 0.11 | - | 0.01 |
| metadata | unseen | model | 0.69 | 0.03 | 0.72 | - | - | - | - | - | - | - | - | 0.12 |
| rules | unseen | ideal_cache | 0.65 | 0.06 | 0.68 | 0.04 | 0.18 | 0.04 | - | - | 0.86 | 0.11 | 0.18 | 0.30 |
| embedding | unseen | ideal_cache | 0.68 | 0.06 | 0.69 | 0.00 | 0.00 | 0.00 | - | - | 0.19 | 0.11 | 0.00 | 6.39 |
| slm_fewshot | unseen | ideal_cache | 0.83 | 0.05 | 0.83 | 0.49 | 0.74 | 0.08 | - | - | 0.28 | 0.11 | 0.73 | 513.02 |
| cleaner | unseen | ideal_cache | - | - | - | 0.05 | 0.20 | 0.04 | - | - | 0.90 | 0.11 | 0.20 | 0.01 |
| metadata | unseen | ideal_cache | 0.80 | 0.04 | 0.80 | - | - | - | - | - | - | - | - | 0.12 |
