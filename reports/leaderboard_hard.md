# Route Leaderboard — HARD mode (dirty card-network descriptors)

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

Embedding wins when merchants are known (random: 0.96 merchant, 0.94 category). On
brand-disjoint unseen merchants only the SLM can name the merchant (0.89), because it reads
the string instead of matching a list; its category (0.79) also holds up better than
embedding's (0.64). See `leaderboard_hard_jev.md` for the same table with the Jev routes.

| route | split | Category Acc | Cat ±95% | Macro F1 | Merchant Acc | Merchant Norm | Merch ±95% | Dedup Ratio | Gold Dedup | Spend KPI | ms/txn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | random | 0.79 | 0.03 | 0.79 | 0.76 | 0.77 | 0.04 | 0.57 | 0.42 | 0.77 | 0.09 |
| embedding | random | 0.94 | 0.01 | 0.94 | 0.96 | 0.96 | 0.01 | 0.42 | 0.42 | 0.96 | 6.96 |
| slm_fewshot | random | 0.76 | 0.04 | 0.76 | 0.62 | 0.90 | 0.03 | 0.56 | 0.42 | 0.90 | 437.13 |
| rules | unseen | 0.55 | 0.06 | 0.59 | 0.05 | 0.25 | 0.04 | 0.84 | 0.09 | 0.25 | 0.10 |
| embedding | unseen | 0.64 | 0.09 | 0.64 | 0.00 | 0.00 | 0.00 | 0.18 | 0.09 | 0.00 | 6.50 |
| slm_fewshot | unseen | 0.79 | 0.07 | 0.76 | 0.59 | 0.89 | 0.06 | 0.21 | 0.09 | 0.89 | 504.83 |
