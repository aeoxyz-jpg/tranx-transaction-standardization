# Route Leaderboard — HARD mode, with TypeSafe Jev routes

Feed synthesized with `synth --hard`: aggregator prefixes (`SQ */PAYPAL */PP*/TST*`),
uppercasing, embedded city/state/store-id, truncation, space collapsing. 1000 eval rows
per split, same rows for every route.

Regenerated 2026-09-27 after two data fixes, so these numbers are **not comparable**
to earlier versions of this file:
- `AMZN MKTP` is no longer a synthetic wrapper or a stripped prefix (on real statements
  it is Amazon's own descriptor).
- Silver merchant labels now strip the source templates' location/time suffixes
  (`Airport`, `Campus`, `Shopping Center`, `Business District`, `Hospital`, `Weekday`).
  Before, one brand was split into up to six "merchants" (2858 labels for 451 brands) and
  98.5% of unseen-split rows had their brand in the training list; now 0%.

`Merchant Acc` is exact string match; `Merchant Norm` ignores case/punctuation.
`rules`' unseen `Merchant Norm` (0.24) is mostly its `derive_canonical` fallback
reproducing the silver label, not generalization. SLM rows vary by about +/-1 point
between runs (Ollama on GPU is not bit-deterministic at temperature 0).

- `jev_merchant`: category from a Jev Choice question over the 10 categories; merchant from
  a Jev Choice over the fuzzy top-20 known merchants plus `none_of_these` (none falls back
  to `derive_canonical`).
- `jev_slm`: same, but `none_of_these` escalates the merchant to `slm_fewshot`
  (escalated 2.9% of rows on random, 83.8% on unseen).

Jev is the best category route on unseen merchants (0.88 vs SLM 0.67, embedding 0.55) and
the best merchant route on known merchants (0.97 vs embedding 0.92). As a gate for new
merchants it leaks on this feed: ~16% of unseen rows were accepted as a known "merchant",
mostly generic list entries (`Citibank` -> `Bank`, `Salary` -> `Direct Deposit`), so
`jev_slm` (0.75) trails the SLM alone (0.90) on unseen. On real statements (MoneyData,
brand-only list) Jev said none for 97% of held-out merchants; see
`reports/real/moneydata_summary_high-medium_aliased.json`.

The spend plot `spend_C00812.png` is from the best random-split route (`jev_slm`).

| route | split | Category Acc | Macro F1 | Merchant Acc | Merchant Norm | Dedup Ratio | Spend KPI | ms/txn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | random | 0.80 | 0.79 | 0.76 | 0.77 | 0.56 | 0.76 | 0.08 |
| embedding | random | 0.92 | 0.92 | 0.92 | 0.92 | 0.41 | 0.92 | 6.51 |
| slm_fewshot | random | 0.73 | 0.75 | 0.62 | 0.89 | 0.56 | 0.62 | 405.58 |
| jev_merchant | random | 0.88 | 0.88 | 0.96 | 0.96 | 0.43 | 0.96 | 335.44 |
| jev_slm | random | 0.88 | 0.88 | 0.97 | 0.97 | 0.43 | 0.97 | 342.41 |
| rules | unseen | 0.41 | 0.45 | 0.07 | 0.24 | 0.84 | 0.07 | 0.10 |
| embedding | unseen | 0.55 | 0.54 | 0.00 | 0.00 | 0.19 | 0.00 | 6.34 |
| slm_fewshot | unseen | 0.67 | 0.68 | 0.60 | 0.90 | 0.26 | 0.60 | 409.95 |
| jev_merchant | unseen | 0.88 | 0.88 | 0.07 | 0.24 | 0.77 | 0.07 | 334.71 |
| jev_slm | unseen | 0.87 | 0.87 | 0.49 | 0.75 | 0.27 | 0.49 | 725.06 |
