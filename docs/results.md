# Detailed results

Supporting tables for the [README](../README.md). Every number comes from a file under
`reports/`, produced by the run pinned in `reports/run/manifest.json` (MoneyData and
DoDataThings predictions are fixed files under `reports/real/`).

Terms used below:

- **Model view:** eval rows whose exact descriptor was never seen in training, one row
  per descriptor. This is what a model would have to answer behind a descriptor cache.
- **Ideal-cache view:** every eval row; a descriptor seen in training takes its stored
  label, the rest take the method's answer. An upper bound for a system with a cache.
- **random / unseen:** the random split shares merchants between training and evaluation
  (known merchants); the unseen split holds out whole brand families (new merchants).
- **Merchant accuracy** is a match after ignoring case and punctuation, with the
  parent-brand rule described in the README. The "exact" column in the reports does not
  apply either.

## 1. Synthetic feed, all methods

`reports/leaderboard_hard_jev.md`. ± is the half-width of a 95% interval from resampling
whole merchants.

### Model view

| Method | random category | random merchant | unseen category | unseen merchant | ms per txn |
|---|---|---|---|---|---|
| cleaner | - | 0.18 ±0.03 | - | 0.21 ±0.02 | 0.01 |
| metadata | 0.69 ±0.03 | - | 0.69 ±0.03 | - | 0.1 |
| rules | 0.74 ±0.03 | 0.68 ±0.04 | 0.50 ±0.04 | 0.19 ±0.02 | 0.2-0.3 |
| embedding | 0.79 ±0.02 | 0.84 ±0.03 | 0.55 ±0.06 | 0.00 | 6.4 |
| slm_fewshot | 0.70 ±0.04 | 0.76 ±0.04 | 0.74 ±0.05 | 0.76 ±0.06 | 460-510 |
| jev_merchant | 0.78 ±0.03 | 0.94 ±0.02 | 0.83 ±0.04 | 0.21 ±0.02 | 360-370 |
| jev_slm | 0.78 ±0.03 | 0.95 ±0.02 | 0.83 ±0.04 | 0.74 ±0.06 | 500-780 |

Model-view sizes: 1,483 descriptors (random), 2,665 (unseen). Merchant accuracy leaves out
abbreviated fictional-local rows (40 random, 93 unseen), whose full name is not in the
input; they are reported in section 2.

### Ideal-cache view

| Method | random category | random merchant | unseen category | unseen merchant |
|---|---|---|---|---|
| cleaner | - | 0.93 | - | 0.20 |
| metadata | 0.97 | - | 0.80 | - |
| rules | 0.97 | 0.97 | 0.65 | 0.18 |
| embedding | 0.98 | 0.98 | 0.68 | 0.00 |
| slm_fewshot | 0.97 | 0.98 | 0.83 | 0.74 |
| jev_merchant | 0.98 | 0.99 | 0.89 | 0.20 |
| jev_slm | 0.97 | 1.00 | 0.89 | 0.73 |

91.8% of random-split eval rows and 38.1% of unseen-split rows are cache hits
(`reports/run/eval_stats.json`); on the unseen split every hit is a merchant-less row.

### Other columns

- **No-match rate** (share of merchant rows with no confident match): rules 0.31 random,
  0.94 unseen; Jev's `none_of_these` 0.06 random, 0.96 unseen.
- **Retrieval recall** (right merchant among Jev's 20 candidates): 0.95 random, 0.00
  unseen (by construction).
- **Share sent from Jev to the SLM** in `jev_slm`: 0.057 random, 0.962 unseen.

## 2. Where the SLM succeeds on new merchants

`subsets` in `reports/leaderboard_hard_jev.json`, model view, abbreviated rows included.

| Method, split | national brands | fictional locals | name intact | abbreviated | truncated |
|---|---|---|---|---|---|
| slm_fewshot, unseen | 0.83 | 0.50 | 0.82 | 0.19 | 0.25 |
| jev_slm, unseen | 0.80 | 0.50 | 0.81 | 0.16 | 0.25 |
| slm_fewshot, random | 0.81 | 0.49 | 0.84 | 0.11 | 0.14 |
| jev_slm, random | 0.95 | 0.97 | 0.98 | 0.79 | 0.90 |

## 3. Paired significance tests

`reports/significance.md`. Both methods are scored on the same rows; whole merchants are
resampled 4,000 times. The first two rows were fixed as the primary comparison before the
experiments were run; the rest are exploratory.

| Comparison (A - B) | Data | diff | 95% CI | merchants | rows |
|---|---|---|---|---|---|
| jev_slm - slm_fewshot, merchant (primary) | synthetic unseen | -0.018 | -0.045 to +0.001 | 213 | 1,963 |
| jev_slm - slm, merchant (primary) | MoneyData, realistic list | +0.049 | +0.019 to +0.101 | 261 | 547 |
| jev_merchant - embedding, merchant | synthetic random | +0.095 | +0.068 to +0.122 | 393 | 866 |
| jev_merchant - embedding, merchant | synthetic unseen | +0.207 | +0.190 to +0.225 | 213 | 1,963 |
| embedding - jev_merchant, category | synthetic random | +0.006 | -0.024 to +0.037 | 586 | 1,483 |
| embedding - jev_merchant, category | synthetic unseen | -0.277 | -0.342 to -0.211 | 392 | 2,665 |
| jev - embedding, merchant | MoneyData, full list | +0.080 | +0.014 to +0.126 | 261 | 547 |
| embedding - jev, category | DoDataThings | +0.096 | +0.067 to +0.126 | 1,000 descriptors | 1,000 |

On unseen merchants `jev_merchant` returns the cleaned text, so its +0.207 over embeddings
is string cleaning against a method that can only return known names.

MoneyData sensitivity for the primary comparison: weighting by rows gives +0.044 (Amazon
is 34% of rows); without Amazon, +0.067 (CI +0.020 to +0.132); dropping any one merchant
leaves the per-descriptor diff between +0.042 and +0.071. DoDataThings rows are paired by
descriptor, not clustered by merchant.

## 4. Jev as a "new merchant" detector

`reports/jev_confidence_summary.json`, `reports/jev_threshold_cascade.json`.

| Data | rows | answered none_of_these | right picks | wrong picks | AUROC (right vs wrong) | wrong-pick confidence, median |
|---|---|---|---|---|---|---|
| synthetic random | 906 | 5.8% | 844 | 9 | 0.97 | 0.50 |
| synthetic unseen | 2,056 | 96.4% | 0 | 75 | - | 0.58 |
| MoneyData, full list | 547 | 4.2% | 522 | 2 | 0.996 | 0.58 |
| MoneyData, realistic list | 547 | 36.4% | 342 | 6 | 0.96 | 0.76 |

These rows keep abbreviated local merchants, so counts differ from section 1.

Cascade accuracy (Jev's pick accepted only above a confidence threshold, otherwise the
SLM's answer), and the share of rows sent to the SLM:

| Threshold | synthetic random | synthetic unseen | MoneyData realistic list |
|---|---|---|---|
| SLM only | 0.723 (100%) | 0.724 (100%) | 0.806 (100%) |
| none (accept every pick) | 0.949 (5.8%) | 0.704 (96.4%) | 0.857 (36.4%) |
| 0.7 | 0.919 (10.9%) | 0.714 (98.7%) | 0.856 (37.5%) |
| 0.9 | 0.900 (19.5%) | 0.721 (99.6%) | 0.843 (39.7%) |
| 0.95 | 0.877 (24.6%) | 0.722 (99.8%) | 0.841 (40.6%) |

The synthetic columns combine the `jev_merchant` picks with the `slm_fewshot` answers; a
parent-brand answer on a row sent to the SLM is judged by Jev's category. The MoneyData
column uses a separately saved set of Jev answers that differs from the main MoneyData
predictions on 2 of 547 descriptors (0.857 here, 0.856 in section 5).

## 5. MoneyData (real UK statements)

`reports/real/moneydata_summary_high-medium_aliased.json`. 547 labelled descriptors,
3,521 rows, 261 merchants; 168 low-confidence labels excluded.

| Setting | cleaner | fuzzy | embedding | SLM | Jev | Jev, none to SLM |
|---|---|---|---|---|---|---|
| Every merchant on the list, per descriptor | 0.18 | 0.87 | 0.88 | 0.81 | 0.96 | |
| per row | | 0.84 | 0.82 | 0.83 | 0.94 | |
| per merchant (macro) | | 0.90 | 0.90 | 0.68 | 0.94 | |
| spend-weighted | | 0.73 | 0.72 | 0.58 | 0.77 | |
| spend-weighted, largest descriptor removed | | 0.92 | 0.90 | 0.72 | 0.97 | |
| Realistic list (one-off merchants new), per descriptor | | 0.60 | 0.56 | 0.81 | 0.62 | 0.86 |
| per row | | 0.65 | 0.58 | 0.83 | 0.68 | 0.87 |
| per merchant (macro) | | 0.21 | 0.22 | 0.68 | 0.23 | 0.71 |
| spend-weighted, largest descriptor removed | | 0.34 | 0.33 | 0.72 | 0.36 | 0.75 |

- On the realistic list 202 merchants are new. Jev answered `none_of_these` for 95.5% of
  descriptors whose merchant was off the list and 1.7% of those on it. On off-list
  merchants alone the SLM scores 0.66.
- The largest descriptor by spend, `WWW.III.CO.UK DE` (an investment-platform transfer,
  27 rows), carries 19.1% of all debit spend and is missed by every method
  (`counts.top_spend_descriptor`).
- An embedding-similarity gate (accept the embedding's name above cosine 0.7, else the
  SLM) sends 67% of descriptors to the SLM.

## 6. Category on DoDataThings

`reports/real/ddt_summary.json`, 1,000 test descriptions, 17 categories.

| | accuracy | ±95% | macro F1 | invalid answers |
|---|---|---|---|---|
| embedding (trained on this dataset) | 0.90 | 0.02 | 0.91 | 0 |
| Jev (zero-shot) | 0.81 | 0.03 | 0.81 | 0 |
| SLM (zero-shot) | 0.62 | 0.03 | 0.34 | 31 |

## 7. Routing the category by Jev's merchant answer

`reports/category_routing.json`. Rule: Jev's category when Jev's merchant answer is
`none_of_these`, otherwise the embedding's category.

| | embedding | Jev | rule | rule - Jev (95% CI) |
|---|---|---|---|---|
| random | 0.790 | 0.784 | 0.769 | -0.015 (-0.038 to +0.009) |
| unseen | 0.552 | 0.829 | 0.816 | -0.013 (-0.031 to 0.000) |

| Rows | random: embedding / Jev | unseen: embedding / Jev |
|---|---|---|
| merchant-less (salary, transfers, generic providers) | 0.847 / 0.804 | 0.842 / 0.791 |
| Jev picked a known merchant | 0.771 / 0.796 | 0.287 / 0.649 |

## 8. How often a descriptor is new

`reports/cache_sim.json`.

- MoneyData, one person, share of debit rows whose exact descriptor had not appeared
  before, by year: 2015 0.210, 2016 0.105, 2017 0.130, 2018 0.164, 2019 0.229, 2020 0.190,
  2021 0.164, 2022 0.248.
- Synthetic feed, bank-wide share of rows whose descriptor appeared in an earlier month:
  0.52 in the first month, rising to 0.93 by the twelfth. Dates are uniform and the repeat
  structure is a generator setting, so this illustrates the mechanism rather than
  measuring it.
