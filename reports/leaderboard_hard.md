# Route Leaderboard

Model view: eval rows whose description is verbatim in train are removed, the rest deduplicated to one row per descriptor and capped (cache-miss accuracy; not comparable to earlier leaderboards). Ideal-cache view: cache hits take the train gold, the model view's descriptors take the route's prediction broadcast per descriptor (rules' per-row MCC category is approximated). Spend KPI: ideal-cache view only.

| route | split | view | Category Acc | Cat ±95% | Macro F1 | Merchant Acc | Merchant Norm | Merch ±95% | No-Match Rate | Retrieval Recall | Dedup Ratio | Gold Dedup | Spend KPI | ms/txn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | random | model | 0.75 | 0.03 | 0.76 | 0.69 | 0.70 | 0.04 | 0.29 | - | 0.64 | 0.44 | - | 0.27 |
| embedding | random | model | 0.79 | 0.02 | 0.79 | 0.85 | 0.85 | 0.02 | - | - | 0.45 | 0.44 | - | 6.47 |
| slm_fewshot | random | model | 0.71 | 0.04 | 0.71 | 0.51 | 0.77 | 0.03 | - | - | 0.61 | 0.44 | - | 500.98 |
| cleaner | random | model | - | - | - | 0.04 | 0.17 | 0.02 | - | - | 0.99 | 0.44 | - | 0.01 |
| metadata | random | model | 0.69 | 0.03 | 0.71 | - | - | - | - | - | - | - | - | 0.12 |
| rules | random | ideal_cache | 0.97 | 0.01 | 0.97 | 0.97 | 0.97 | 0.00 | - | - | 0.24 | 0.18 | 0.97 | 0.27 |
| embedding | random | ideal_cache | 0.97 | 0.01 | 0.97 | 0.98 | 0.98 | 0.00 | - | - | 0.18 | 0.18 | 0.98 | 6.47 |
| slm_fewshot | random | ideal_cache | 0.97 | 0.01 | 0.97 | 0.96 | 0.98 | 0.00 | - | - | 0.25 | 0.18 | 0.98 | 500.98 |
| cleaner | random | ideal_cache | - | - | - | 0.92 | 0.93 | 0.00 | - | - | 0.37 | 0.18 | 0.93 | 0.01 |
| metadata | random | ideal_cache | 0.97 | 0.01 | 0.97 | - | - | - | - | - | - | - | - | 0.12 |
| rules | unseen | model | 0.48 | 0.04 | 0.53 | 0.04 | 0.19 | 0.02 | 0.95 | - | 0.86 | 0.10 | - | 0.33 |
| embedding | unseen | model | 0.55 | 0.06 | 0.56 | 0.00 | 0.00 | 0.00 | - | - | 0.18 | 0.10 | - | 6.54 |
| slm_fewshot | unseen | model | 0.74 | 0.06 | 0.71 | 0.48 | 0.76 | 0.05 | - | - | 0.25 | 0.10 | - | 516.17 |
| cleaner | unseen | model | - | - | - | 0.05 | 0.21 | 0.02 | - | - | 0.89 | 0.10 | - | 0.01 |
| metadata | unseen | model | 0.70 | 0.03 | 0.72 | - | - | - | - | - | - | - | - | 0.12 |
| rules | unseen | ideal_cache | 0.57 | 0.06 | 0.62 | 0.04 | 0.18 | 0.04 | - | - | 0.87 | 0.09 | 0.18 | 0.33 |
| embedding | unseen | ideal_cache | 0.64 | 0.07 | 0.65 | 0.00 | 0.00 | 0.00 | - | - | 0.18 | 0.09 | 0.00 | 6.54 |
| slm_fewshot | unseen | ideal_cache | 0.79 | 0.05 | 0.79 | 0.48 | 0.74 | 0.07 | - | - | 0.27 | 0.09 | 0.73 | 516.17 |
| cleaner | unseen | ideal_cache | - | - | - | 0.05 | 0.22 | 0.04 | - | - | 0.90 | 0.09 | 0.21 | 0.01 |
| metadata | unseen | ideal_cache | 0.74 | 0.05 | 0.75 | - | - | - | - | - | - | - | - | 0.12 |
