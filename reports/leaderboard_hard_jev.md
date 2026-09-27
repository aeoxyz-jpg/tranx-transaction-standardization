# Route Leaderboard — HARD mode, with TypeSafe Jev routes

Feed synthesized with `synth --hard` (aggregator prefixes, uppercasing, embedded
city/state/store id, truncation, space collapsing). 1000 eval rows per split, the same rows
for every route. Regenerated 2026-09-27 after a label and harness audit; numbers are **not
comparable** to earlier versions of this file.

**Labels.** About 25% of rows have no merchant and count for category only: transaction
types (salary, transfer, loan, donation, fees) and labels that name what was bought rather
than who was paid (MRI, Toll, Broadband, Insurance, ...). Merchant columns cover the other
rows (751 random, 763 unseen). Silver merchant names strip the source templates'
location/time suffixes but keep names that contain those words (Community Center, Online
Bank); Cane's is merged into Raising Cane's; Frontier is split into Frontier Airlines /
Frontier Communications.

**Splits.** `random` shares merchants between train and eval. `unseen` holds out whole brand
families by a stable hash (no eval brand, or sibling label such as Walmart for Walmart
Pharmacy, is in the training list).

**Inputs.** rules: description + MCC. embedding, slm_fewshot, Jev: description only. For
reference, a category guesser that never reads the description (type code + MCC + amount
magnitude, majority class) scores 0.68 (random) / 0.69 (unseen); the audit estimates the
text-only category ceiling at about 0.99 because a few merchants carry two categories in
the source (CVS, Walgreens, Target, Post Office).

**Reading the columns.** `±95%` is a merchant-cluster bootstrap half-width. `Merchant Norm`
ignores case and punctuation; `Merchant Acc` is exact. `Spend KPI` buckets per customer on
the normalized name; at 1000 rows almost every bucket is one transaction, so it tracks
`Merchant Norm`. Compare `Dedup Ratio` with `Gold Dedup` (closer is better). On `unseen`,
the merchant scores of `rules` (0.25) and `jev_merchant` (0.26) come from their fallback,
`derive_canonical`, reproducing the silver label; they are not generalization. slm_fewshot
varies by about +/-1 point between runs (Ollama on GPU).

- `jev_merchant`: category from a Jev Choice over the 10 categories; merchant from a Jev
  Choice over the fuzzy top-20 training merchants plus `none_of_these` (none falls back to
  `derive_canonical`).
- `jev_slm`: same, but `none_of_these` escalates the merchant to `slm_fewshot` (1.2% of
  merchant rows on random, 87.5% on unseen).

Jev picks the known merchant best (random 0.98 vs embedding 0.96) and is the best category
route on unseen merchants (0.88 vs SLM 0.79, embedding 0.64), with description only. As a
gate for new merchants it leaks on this feed: on unseen, 12.7% of merchant rows were
matched to a training label instead of `none_of_these`, so `jev_slm` (0.81) trails the SLM
alone (0.89). A confidence threshold narrows that gap but does not close it
(`reports/jev_threshold_cascade.json`); real-statement results are in `reports/real/`.

| route | split | Category Acc | Cat ±95% | Macro F1 | Merchant Acc | Merchant Norm | Merch ±95% | Dedup Ratio | Gold Dedup | Spend KPI | ms/txn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | random | 0.79 | 0.03 | 0.79 | 0.76 | 0.77 | 0.04 | 0.57 | 0.42 | 0.77 | 0.09 |
| embedding | random | 0.94 | 0.01 | 0.94 | 0.96 | 0.96 | 0.01 | 0.42 | 0.42 | 0.96 | 6.96 |
| slm_fewshot | random | 0.76 | 0.04 | 0.76 | 0.62 | 0.90 | 0.03 | 0.56 | 0.42 | 0.90 | 437.13 |
| jev_merchant | random | 0.85 | 0.03 | 0.85 | 0.98 | 0.98 | 0.01 | 0.44 | 0.42 | 0.98 | 347.59 |
| jev_slm | random | 0.85 | 0.03 | 0.85 | 0.98 | 0.98 | 0.01 | 0.43 | 0.42 | 0.99 | 437.28 |
| rules | unseen | 0.55 | 0.06 | 0.59 | 0.05 | 0.25 | 0.04 | 0.84 | 0.09 | 0.25 | 0.10 |
| embedding | unseen | 0.64 | 0.09 | 0.64 | 0.00 | 0.00 | 0.00 | 0.18 | 0.09 | 0.00 | 6.50 |
| slm_fewshot | unseen | 0.79 | 0.07 | 0.76 | 0.59 | 0.89 | 0.06 | 0.21 | 0.09 | 0.89 | 504.83 |
| jev_merchant | unseen | 0.88 | 0.05 | 0.88 | 0.06 | 0.26 | 0.04 | 0.80 | 0.09 | 0.26 | 337.42 |
| jev_slm | unseen | 0.87 | 0.05 | 0.88 | 0.55 | 0.81 | 0.08 | 0.21 | 0.09 | 0.81 | 775.99 |
