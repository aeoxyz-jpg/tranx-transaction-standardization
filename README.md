# Tranx: Transaction Standardization with Small Language Models

Tranx turns noisy bank transaction descriptions into **direction**, **category** and
**canonical merchant**, then rolls up per-customer spend by merchant. A customer can then
see one total for McDonald's or BP across lines like `McDonald's #111`,
`SQ *MCDONALDS F1234 ATLANTA GA` and `BP on Buford Hwy`.

Four method families are compared under one evaluation harness: fuzzy rules, sentence
embeddings, a local small language model (SLM) prompted few-shot, and TypeSafe's hosted
Jev model. Two baselines anchor them: string cleaning alone, and a category guess from
metadata alone.

## Results in brief

- **Most transactions need no model.** A bank sees the same descriptor again and again.
  In one person's real statements 730 distinct descriptors cover 4,043 debit rows (18%);
  a descriptor cache answers the rest. Models only matter for cache misses, and every
  synthetic score below is measured on cache misses.
- **Known merchants:** Jev picks the right merchant from a known list most often (0.94 on
  synthetic cache misses, 0.96 on real UK statements), ahead of embeddings (0.85 / 0.88).
- **New merchants:** only the SLM can name a merchant that is not on the list (0.76 on
  brand-disjoint synthetic merchants). It is much better on well-known brands (0.82) than
  on fictional local shops (0.52) and on the one-off merchants of real statements (0.65).
- **Category:** on new brands Jev is best without any training (0.83, embeddings 0.55,
  metadata alone 0.70). With labelled data and familiar merchants, embeddings plus
  logistic regression tie with Jev on synthetic cache misses and beat it on an independent
  dataset (0.90 vs 0.81).
- **Combining them** (Jev picks from the list, its "none of these" answers go to the SLM)
  is the pre-set primary comparison, and it points in opposite directions on the two
  datasets. On real statements the cascade beats the SLM alone (+0.044, 95% CI +0.014 to
  +0.094). On synthetic new merchants it loses (-0.041, CI -0.071 to -0.017), because Jev
  files 6.5% of new merchants under a known one with high confidence. The cascade is worth
  it only with a clean brand-only list.

Every number below comes from a file in `reports/`, produced by one run whose inputs are
pinned in `reports/run/` (see [Reproducing](#reproducing)).

## Data and labels

| Source | What it is | Labels | Used for |
|---|---|---|---|
| Synthetic hard feed | 100k transactions built from the public HF dataset `mitulshah/transaction-categorization`, with customers, amounts, payment methods, noisy MCCs, card-network-style dirty descriptors, fictional local merchants and repeat visits | Silver: merchant from the clean source description or the fictional name; category from the source | Main leaderboard, both splits |
| MoneyData (real) | One person's anonymized UK bank statements, 2015-2022 ([Firat et al. 2023](https://github.com/thevisgroup/MoneyVis)); 547 labelled card and direct-debit descriptors, 3,521 rows | Merchant labels drafted by an LLM, audited by a second model; **not human-verified**; alias table for legitimate alternative names | Merchant normalization on real noise |
| DoDataThings v2 | Independently generated synthetic US descriptors, 17 categories ([HF](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2)) | Category only | Category on noise that Tranx's own generator did not produce |

**What the generator adds** (`tranx/synth/`, settings in `tranx/config.py`):

- **Fictional local merchants.** A stable-hash quarter of merchant rows in eight
  categories moves to 800 invented local names ("Rustic Stag Bookshop", "Kingsbury
  Physiotherapy") that share no word with any source label; 17,060 of 75,515 merchant
  rows end up local (`reports/run/synth_stats.json`). The source merchants are mostly
  national chains the SLM saw in pretraining; the locals test recovering a name the model
  has never seen.
- **Abbreviation.** 15% of descriptors drop interior vowels (`BLUE HERON BAKERY` becomes
  `BLUE HRN BKRY`). For a fictional name the full name is then not in the input, so those
  rows are left out of the headline merchant score and reported separately.
- **Repeat visits.** Each merchant has a number of locations that grows with its row
  count, each location has one fixed descriptor, and visits favour a few locations.
  This gives 0.159 distinct descriptors per row; MoneyData has 0.181 over all debit rows
  and 0.155 over its labelled descriptors (`counts` in
  `reports/real/moneydata_summary_high-medium_aliased.json`). Both the local share and the
  repeat rate are generator parameters, not findings.

**Merchant-less rows.** About 25% of synthetic rows have no merchant: transaction types
(salary, transfer, loan, donation, fees) and labels that name what was bought rather than
who was paid (MRI, Toll, Broadband). They are scored for category only.

**Splits.** `random` shares merchants between train and eval. `unseen` holds out whole
brand families by a stable hash, so no eval brand, and no sibling label such as Walmart
for Walmart Pharmacy, appears in training; the held-out set is 231 merchants, most of them
fictional locals. For MoneyData the equivalent is a realistic merchant list: merchants
seen in at least two descriptors are on the list, and the 202 merchants seen once count
as new.

## How the synthetic feed is scored

A production system would answer a descriptor it has seen before from a cache, so the
harness separates two views (`tranx/cli.py`, `eval_rows`):

- **Model view** (the headline): eval rows whose exact descriptor appears in the train
  split are removed, the rest are reduced to one row per descriptor, and descriptors that
  map to two different merchants are dropped. On the random split 91.7% of eval rows are
  cache hits, leaving 1,496 descriptors; on the unseen split 22.9% are hits (all
  merchant-less), leaving 2,833 (`reports/run/eval_stats.json`). These are rare
  descriptors, so the scores are lower than scores over all rows and are not comparable to
  earlier versions of this README.
- **Ideal-cache view:** all eval rows; hits take the stored train label, misses take the
  route's prediction. It is an upper bound for a cache (a real cache stores predictions,
  which can be wrong) and carries the spend rollup KPI.

## Routes

| Route | Merchant | Category | Needs | ms/txn |
|---|---|---|---|---|
| `cleaner` | the description with processor prefixes, store numbers and locations stripped (baseline) | none | nothing | 0.01 |
| `metadata` | none | logistic regression on type code, MCC and amount; never reads the description (baseline) | labelled rows | 0.1 |
| `rules` | rapidfuzz match against the training merchant list; falls back to the cleaned text | MCC and merchant priors learned from training data | labelled rows | 0.3 |
| `embedding` | nearest merchant name by MiniLM cosine similarity | logistic regression on description embeddings | merchant list; labelled rows for category | 6.5 |
| `slm_fewshot` | Qwen2.5-3B (Ollama) reads the description and writes the merchant | same prompt | nothing trained; runs locally | 500-520 |
| `jev_merchant` | Jev chooses among the fuzzy top-20 known merchants or `none_of_these` (then the cleaned text) | Jev chooses one of the categories | merchant list; `TYPESAFE_API_KEY` | 320-330 |
| `jev_slm` | as `jev_merchant`, but `none_of_these` goes to the SLM | Jev | both | 410-780 |

Jev is TypeSafe's hosted "System One" model: it answers typed questions (here a Choice
over named options) with a probability per option and a confidence score, and never
writes free text. `rules` and `metadata` see the MCC; the other routes see the description
only. Latency is serial, one request at a time, on an Apple Silicon laptop; Jev's figure is
mostly the network round trip.

## Results

### Synthetic hard feed, model view

`reports/leaderboard_hard_jev.md`, category / merchant (normalized match). Merchant 95%
intervals are merchant-cluster bootstrap half-widths from the same file.

| Route | random: category | random: merchant | unseen: category | unseen: merchant |
|---|---|---|---|---|
| cleaner (baseline) | - | 0.17 | - | 0.21 |
| metadata (baseline) | 0.69 | - | 0.70 | - |
| rules | 0.75 | 0.70 ±0.04 | 0.48 | 0.19\* |
| embedding | 0.79 | 0.85 ±0.02 | 0.55 | 0.00 |
| slm_fewshot | 0.71 | 0.77 ±0.03 | 0.74 | **0.76** ±0.05 |
| jev_merchant | **0.80** | 0.94 ±0.01 | **0.83** | 0.20\* |
| jev_slm | 0.79 | **0.95** ±0.01 | **0.83** | 0.71 ±0.06 |

\* On unseen merchants `rules` and `jev_merchant` fall back to the cleaned text, so their
score is at or below the cleaner's (0.21), not generalization. Embeddings can only return a known
name and score 0 by construction.

![Merchant accuracy per route on the model view, random vs unseen split, with confidence intervals](reports/figures/merchant_norm.png)

**New merchants by origin** (`subsets` in `reports/leaderboard_hard_jev.json`, unseen
split, abbreviated rows included):

| Route | national brands | fictional locals | name intact | name abbreviated |
|---|---|---|---|---|
| slm_fewshot | 0.82 | 0.52 | 0.83 | 0.17 |
| jev_slm | 0.76 | 0.51 | 0.79 | 0.14 |

The gap between brands and locals is the pretraining advantage the old unseen split
rewarded. The locals' 0.52 sits closer to MoneyData's one-off merchants (0.65) than the
brands' 0.82 does.

**Ideal-cache view** (same file): with a cache in front, random-split merchant accuracy is
0.97-1.00 for every route except the cleaner (0.93), because 92% of rows never reach a
model. On the unseen split the cache only answers merchant-less rows, so the scores stay
close to the model view.

### Significance

`reports/significance.md`. Paired merchant-cluster bootstrap, 4,000 resamples, both
routes on the same resample. One comparison was fixed before the rerun as primary; the
rest are exploratory. The intervals describe merchants like these (the same generator, or
the same person's statements), not other customers.

| Comparison (A - B) | Data | diff | 95% CI |
|---|---|---|---|
| **jev_slm - slm_fewshot, merchant (primary)** | synthetic unseen | -0.041 | -0.071 to -0.017 |
| **jev_slm - slm, merchant (primary)** | MoneyData, realistic list | +0.044 | +0.014 to +0.094 |
| jev_merchant - embedding, merchant | synthetic random | +0.090 | +0.069 to +0.113 |
| jev - embedding, merchant | MoneyData, full list | +0.080 | +0.014 to +0.126 |
| embedding - jev_merchant, category | synthetic random | -0.009 | -0.038 to +0.021 |
| embedding - jev_merchant, category | synthetic unseen | -0.281 | -0.344 to -0.217 |
| embedding - jev, category | DoDataThings | +0.096 | +0.067 to +0.126 |

For MoneyData, Amazon is 34% of rows; weighting by rows gives +0.043 with Amazon and
+0.065 (CI +0.018 to +0.130) without it. Per descriptor, dropping any single merchant
leaves the diff between +0.037 and +0.063. DoDataThings rows are paired by descriptor, not clustered by
merchant.

![Forest plot of the paired differences with confidence intervals](reports/figures/significance.png)

### Jev as a gate for new merchants

On the unseen split no gold merchant is on the list. Jev answered `none_of_these` for
93.5% of those rows and picked a known merchant for the other 6.5%, confidently
(`reports/jev_confidence_summary.json`). Where the gold is on the list, confidence
separates right from wrong picks well: the probability that a right pick has higher
confidence than a wrong one (AUROC) is 0.98 on the synthetic random split and 0.93 to 1.00
on MoneyData. A confidence threshold trades one error for the other
(`reports/jev_threshold_cascade.json`). This table combines the `jev_merchant` route's
picks with the `slm_fewshot` route's answers rather than rerunning `jev_slm`, and keeps
abbreviated local rows, so its unseen numbers sit below the leaderboard's. Its MoneyData
row uses a separately saved set of Jev answers that differs from the main MoneyData
predictions on 2 of 547 descriptors (0.850 here, 0.848 in the table below).

| Merchant accuracy of the Jev to SLM cascade | no threshold | threshold 0.95 | SLM only |
|---|---|---|---|
| Synthetic, known merchants (random) | 0.949 | 0.877 | 0.747 |
| Synthetic, new merchants (unseen) | 0.683 | 0.720 | 0.723 |
| MoneyData, realistic list | 0.850 | 0.834 | 0.804 |

### Real UK statements (MoneyData)

`reports/real/moneydata_summary_high-medium_aliased.json`, merchant accuracy; 168
low-confidence labels are excluded.

| Setting | fuzzy | embedding | SLM | Jev | Jev, none to SLM |
|---|---|---|---|---|---|
| Every merchant on the list, per descriptor | 0.87 | 0.88 | 0.80 | **0.96** | |
| Realistic list (one-off merchants are new), per descriptor | 0.60 | 0.55 | 0.80 | 0.62 | **0.85** |
| Every merchant on the list, spend-weighted | 0.72 | 0.72 | 0.57 | **0.77** | |
| Same, without the largest descriptor | 0.91 | 0.90 | 0.71 | **0.97** | |
| Realistic list, spend-weighted, without the largest descriptor | | | 0.71 | | **0.73** |

With the realistic list, Jev answered `none_of_these` for 95.5% of descriptors whose
merchant was off the list and for 1.7% of those on it. On the off-list merchants alone the
SLM gets 0.65.

Spend weighting (debit amounts per descriptor) matches the product's rollup KPI, but one
descriptor dominates it: an investment-platform transfer (`WWW.III.CO.UK DE`, 27 rows)
carries 19% of all debit spend, and every route misses it (`counts.top_spend_descriptor`). Without it, spend-weighted
accuracy returns close to the per-descriptor numbers. Averaged per merchant instead of per
descriptor, the realistic-list scores fall to 0.70 or below, because most merchants in one
person's statements appear only once.

![MoneyData merchant accuracy per descriptor, spend-weighted, and spend-weighted without the largest descriptor](reports/figures/moneydata.png)

### Category on an independent generator (DoDataThings v2)

`reports/real/ddt_summary.json`, 1000 test descriptions after removing duplicates shared
with training, 17 categories:

| embedding (trained on this dataset) | Jev (zero-shot) | SLM (zero-shot) |
|---|---|---|
| **0.90** | 0.81 | 0.62 |

The SLM returned an unparseable or off-list category for 31 of the 1000 rows.

## How to choose

Put an exact-descriptor cache first; then the choice depends on whether the merchant is on
your list:

![Decision tree for merchant normalization: reuse a cached descriptor; for known merchants use Jev (0.94 synthetic cache misses, 0.96 real, about 320 ms) or embeddings when speed matters (0.85 synthetic, 0.88 real, 6.5 ms); for new merchants use the SLM (0.76 synthetic unseen: 0.82 on brands, 0.52 on fictional locals; 0.65 on real one-off merchants); for mixed traffic the Jev-then-SLM cascade, which helps on real statements (0.85 vs 0.80) and hurts on synthetic new merchants (0.71 vs 0.76) and needs a clean brand-only list.](reports/figures/merchant_decision.png)

For the category, the deciding questions are whether you have labelled data and whether
the merchants are familiar:

![Decision tree for category: with labelled rows and familiar merchants use embeddings plus logistic regression (0.90 on DoDataThings vs Jev 0.81; a tie with Jev on synthetic cache misses, 0.79 vs 0.80); without labelled rows or for new brands use Jev's zero-shot category choice (0.83 synthetic unseen, 0.81 DoDataThings). Reference: metadata alone 0.69 to 0.70, the SLM 0.71 to 0.74.](reports/figures/category_decision.png)

The cascade for mixed traffic:

![Flow of the cascade: a cached descriptor reuses its stored merchant; otherwise fuzzy retrieval of the top 20 known merchants, a Jev Choice among them or none_of_these; a pick becomes the merchant, none_of_these sends the description to the SLM.](reports/figures/jev_slm_cascade.png)

The cascade is only as good as Jev's "not on the list" answer. With generic entries such as
"Pharmacy" or parent brands such as "Hilton" on the list, Jev files some new merchants
under them; on the synthetic unseen split it did so for 6.5% of rows. Diagram sources are
in `docs/diagrams/` (Mermaid).

## Cost at bank scale (a scenario, not a measurement)

For a large US regional bank at about 5 million card and ACH transactions a day (an
estimate: Regions reported about 700 million debit card transactions in 2010, and US debit
volume roughly tripled to 120.6 billion in 2024 per the Federal Reserve Payments Study),
the inference bill is small:

| Setup | per day | per year |
|---|---|---|
| Every transaction through Jev | $134 | $49k |
| Jev to SLM cascade, no cache | $516 | $188k |
| Same, 20% cache misses | $103 | $38k |
| Same, 5% cache misses | $26 | $9k |

Inputs: Jev at $42 per billion input tokens, output free (typesafe.ai); 638 input tokens
per Jev call, measured in this run (`input_tokens` in `reports/leaderboard_hard_jev.json`
over the model-view rows); 36.4% of cache misses escalated to the SLM (MoneyData
realistic list); the SLM tier priced as Claude Haiku 4.5 on the Batch API (about 295 input
and 25 output tokens per call; its accuracy on this task was not measured). The cache miss
rate at bank scale is unknown: one person's statements show 10.5% to 24.8% new
descriptors per year (`reports/cache_sim.json`), and many customers share merchants.
At this scale the deciding costs are people (keeping the merchant list clean, labelling)
and the review needed before sending descriptors, which can contain names, to an external
API.

## Limitations

- **One real person.** The only real merchant data is one person's UK statements. The
  intervals above do not cover other customers, other countries or other banks.
- **Model-drafted labels.** MoneyData labels were drafted and audited by models, not by
  a person; 168 low-confidence labels are excluded, which leaves the easier descriptors.
- **Synthetic by construction.** The local-merchant share, abbreviation rate and repeat
  structure are generator parameters. Fictional names test recovering an unseen name from
  a noisy descriptor, not knowledge of real local businesses.
- **Cleaner vs generator.** The description cleaner and the generator share part of their
  noise vocabulary (processor prefixes), so the synthetic feed is kinder to string cleaning
  than real statements are.
- **Direction** is recoverable from the amount sign by construction and is not used to
  compare routes.
- **Run-to-run variance.** The SLM varies by about ±1 point between runs (Ollama on GPU is
  not bit-deterministic at temperature 0).

## Reproducing

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login              # accept the dataset terms on its HF page first
ollama pull qwen2.5:3b-instruct
./run.sh                           # tests, hard-mode synth, eval; TRANX_N=5000 ./run.sh for a quick pass
```

`run.sh` scores the local routes into `reports/leaderboard_hard.md`. With
`TYPESAFE_API_KEY` set, or the key in the macOS keychain under `jev-api-key`, it also scores
the Jev routes into `reports/leaderboard_hard_jev.md` (about two hours locally). The
default caps (5,000 per split) keep every model-view descriptor; a smaller cap samples
descriptors away and the ideal-cache view then omits the rest, which `eval` warns about.

`eval` writes `reports/run/manifest.json` (hashes of the feed, gold, prompt and candidate
lists, and the exact eval row ids) and per-row predictions under `reports/preds/`. The
scripts in `scripts/` read those predictions and refuse any file scored against a
different manifest. Run them from the repo root with `PYTHONPATH=.`:
`significance.py`, `jev_confidence.py --synthetic-only`, `jev_threshold_cascade.py
--synthetic-only`, `cache_sim.py`, `make_figures.py`, and for the real data
`eval_moneydata.py --methods ""` (rescores saved predictions; without the flag it calls
the models again) and `eval_ddt.py`. MoneyData's raw file, labels and aliases are kept
locally under `data/real/` and are not in this repository, because the source repository
carries no licence and the labels are unverified.

## History

An earlier version compared a LoRA fine-tune of the same SLM (`slm_lora`) and reported
results on a standard (clean) feed and on all eval rows rather than cache misses. Those
results predate the label audit and the realism fixes and are not comparable;
`reports/leaderboard.md` and `docs/findings/` are kept as history.
