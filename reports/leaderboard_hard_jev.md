# Route Leaderboard — HARD mode, with TypeSafe Jev routes

Feed synthesized with `synth --hard` (aggregator prefixes, uppercasing, embedded
city/state/store id, truncation, space collapsing). 1000 eval rows per split, the same rows
for every route. Regenerated 2026-09-27 after a label and harness audit; numbers are **not
comparable** to earlier versions of this file.

**Labels.** Transaction types (salary, transfer, loan, donation, fees; ~14% of rows) have no
merchant: they count for category only, merchant columns cover the other rows (856 random,
853 unseen). Silver merchant names strip the source templates' location/time suffixes but
keep names that contain those words (Community Center, Online Bank); Cane's is merged into
Raising Cane's; Frontier is split into Frontier Airlines / Frontier Communications.

**Splits.** `random` shares merchants between train and eval. `unseen` holds out whole brand
families by a stable hash (no eval brand, or sibling label such as Walmart for Walmart
Pharmacy, is in the training list).

**Inputs.** rules: description + MCC. embedding, slm_fewshot, Jev: description only. For
reference, a category guesser that never reads the description (type code + MCC + amount
magnitude, majority class) scores 0.68 on the random split; the audit estimates the
text-only category ceiling at about 0.99 because a few merchants carry two categories in
the source (CVS, Walgreens, Target, Post Office).

**Reading the columns.** `±95%` is a merchant-cluster bootstrap half-width. `Merchant Norm`
ignores case and punctuation; `Merchant Acc` is exact. `Spend KPI` buckets per customer on
the normalized name; at 1000 rows almost every bucket is one transaction, so it tracks
`Merchant Norm`. Compare `Dedup Ratio` with `Gold Dedup` (closer is better). On `unseen`,
the merchant scores of `rules` (0.23) and `jev_merchant` (0.21) come from their fallback,
`derive_canonical`, reproducing the silver label; they are not generalization. slm_fewshot
varies by about +/-1 point between runs (Ollama on GPU).

- `jev_merchant`: category from a Jev Choice over the 10 categories; merchant from a Jev
  Choice over the fuzzy top-20 training merchants plus `none_of_these` (none falls back to
  `derive_canonical`).
- `jev_slm`: same, but `none_of_these` escalates the merchant to `slm_fewshot` (1.6% of
  merchant rows on random, 82.5% on unseen).

Jev picks the known merchant best (random 0.97-0.98 vs embedding 0.95) and is the best
category route on unseen merchants (0.87 vs SLM 0.77, embedding 0.62), with description
only. As a gate for new merchants it leaks on this feed: on unseen, ~18% of merchant rows
were matched to a training label instead of `none_of_these`, so `jev_slm` (0.76) trails the
SLM alone (0.90). Whether a confidence threshold fixes that is in
`reports/jev_threshold_cascade.json`; real-statement results are in `reports/real/`.

| route | split | Category Acc | Cat ±95% | Macro F1 | Merchant Acc | Merchant Norm | Merch ±95% | Dedup Ratio | Gold Dedup | Spend KPI | ms/txn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | random | 0.79 | 0.03 | 0.79 | 0.77 | 0.78 | 0.03 | 0.57 | 0.42 | 0.78 | 0.09 |
| embedding | random | 0.94 | 0.01 | 0.94 | 0.95 | 0.95 | 0.01 | 0.42 | 0.42 | 0.95 | 6.48 |
| slm_fewshot | random | 0.76 | 0.04 | 0.76 | 0.61 | 0.89 | 0.03 | 0.57 | 0.42 | 0.89 | 420.00 |
| jev_merchant | random | 0.84 | 0.04 | 0.84 | 0.97 | 0.97 | 0.01 | 0.44 | 0.42 | 0.97 | 339.78 |
| jev_slm | random | 0.84 | 0.04 | 0.84 | 0.98 | 0.98 | 0.01 | 0.43 | 0.42 | 0.98 | 401.01 |
| rules | unseen | 0.50 | 0.05 | 0.54 | 0.05 | 0.23 | 0.03 | 0.89 | 0.09 | 0.23 | 0.11 |
| embedding | unseen | 0.62 | 0.09 | 0.63 | 0.00 | 0.00 | 0.00 | 0.19 | 0.09 | 0.00 | 6.46 |
| slm_fewshot | unseen | 0.77 | 0.07 | 0.76 | 0.56 | 0.90 | 0.04 | 0.22 | 0.09 | 0.90 | 445.65 |
| jev_merchant | unseen | 0.87 | 0.06 | 0.86 | 0.05 | 0.21 | 0.04 | 0.77 | 0.09 | 0.21 | 330.70 |
| jev_slm | unseen | 0.87 | 0.06 | 0.87 | 0.51 | 0.76 | 0.08 | 0.22 | 0.09 | 0.76 | 707.91 |
