# Tranx — Transaction Standardization with Small Language Models

Standardize noisy bank transaction descriptions into **direction**, **category**,
and **canonical merchant**, then roll up per-customer spend by merchant — so a
customer can see their total at McDonald's or BP across messy lines like
`McDonald's #111`, `McDonald's #121`, `BP on Buford Hwy`, `BP @ Pleasant Hills`.

## Why

Real bank feeds carry cryptic descriptions, transaction type codes, and partial
merchant codes (MCC). This project simulates that feed from a public dataset and
compares three locally-runnable approaches under one evaluation harness.

## Architecture

```
raw dataset ──▶ synth ──▶ bank feed + gold/silver labels
  (4 columns)            (customer, amount, direction, payment_method,
                          type_code, partial & noisy MCC; canonical merchant)

bank feed ──▶ clean ──▶ [ route: merchant_norm + classify ] ──▶ aggregate ──▶ rollups
                          rules | embedding | slm_fewshot

eval ──▶ random split + unseen-merchant split ──▶ leaderboard + money-shot plot
```

`clean` and `aggregate` are shared pipeline stages; each route implements
merchant normalization and category classification internally behind a common
`standardize(txn) -> {canonical_merchant, category, direction}` interface.

## Routes

| Route | Merchant normalization | Category | Cost |
|---|---|---|---|
| `rules` | rapidfuzz vs learned canonical list | learned MCC & merchant priors | cheapest |
| `embedding` | nearest canonical (MiniLM) | logistic regression | mid |
| `slm_fewshot` | local Qwen2.5-3B few-shot | LLM | highest |
| `slm_lora` | Qwen2.5-3B LoRA fine-tuned (MLX, Apple Silicon) | LLM | highest |

The `slm_lora` route fine-tunes the base model on the hard feed via MLX
(`scripts/train_lora.sh`). It beats few-shot on category and seen merchants but
**regresses on unseen merchants** — fine-tuning trades base-model generalization
for in-distribution accuracy. See `docs/findings/lora-vs-fewshot.md`.

All category priors in `rules` are **learned from training data**, never read
from the synthetic generative tables.

## Results

Every route is scored on two splits (**random** and **unseen-merchant**) and two
feeds: standard (`reports/leaderboard.md`) and hard mode (`reports/leaderboard_hard.md`,
dirty card-network descriptors). The money shot is `reports/spend_<customer>.png` —
spend-by-merchant for one customer, gold vs predicted.

Merchant normalization (`Merchant Norm`, case/punctuation-insensitive), the metric
that matches the headline goal:

| feed | split | rules | embedding | slm_fewshot |
|---|---|---|---|---|
| standard | random | 1.00\* | 0.99 | 0.81 |
| standard | unseen | 0.94\* | 0.00 | 0.77 |
| hard | random | 0.74 | **0.90** | 0.81 |
| hard | unseen | 0.03 | 0.00 | **0.78** |

\* tautological: `rules` falls back to the same `derive_canonical` that generated
the silver label (see caveats below). On hard descriptors the tautology is gone:
`rules` token-set matching makes it competitive on the random split (0.74) but its
unseen-merchant score collapses to ~0 — hand-rules cannot generalize to merchants
they never saw, which is the whole reason the SLM earns its cost.

**Reading it:**
- The **unseen-merchant split** is the honest test — training sees zero eval
  merchants, so memorizing a canonical list cannot coast.
- **embedding** wins on seen/clean data (robust retrieval) but is vocabulary-locked:
  **0.00** on every unseen split — it cannot emit a merchant it never trained on.
- On the realistic case — **dirty descriptors + unseen merchants** — only the local
  **SLM generalizes** (0.78), because it parses the string instead of matching a
  vocabulary. Category tells the same story: embedding leads on clean data, the SLM
  leads on hard+unseen (0.72 vs 0.68).
- The trade-off: the SLM costs ~570 ms/txn vs embedding's ~8 ms and rules' ~0.05 ms —
  suited to offline batch or hard-case fallback, not per-txn real time.

## Quickstart

```bash
pip install -r requirements.txt
huggingface-cli login              # accept dataset terms on the HF page first
ollama pull qwen2.5:3b-instruct
./run.sh                           # full 100k run; TRANX_N=5000 ./run.sh for a quick pass
```

## Synthetic-data caveats & leakage control

The bank feed is synthesized on top of a public dataset, so the evaluation is
only as honest as its controls. The known leakage paths and how they are handled:

- **Silver merchant labels are circular by construction.** The gold
  `canonical_merchant` is `derive_canonical(description)`, and a rules route can
  reuse the same stripping. On a random split this makes `merchant_acc` look
  near-perfect. **Control:** the unseen-merchant split — training never sees the
  eval merchants, so the score reflects generalization, not memorization.
  **Important caveat:** even on the unseen split, the `rules` route's no-match
  fallback re-derives the silver label with the same `derive_canonical`, so its
  unseen `merchant_acc` (~0.95) is a tautology, not generalization. The honest
  contrast is `embedding` at 0.00 (vocabulary-locked — it cannot emit a merchant
  it never saw) and `slm_fewshot` at ~0.87 (genuinely parses unseen strings).
  That spread, not the rules number, is the real merchant-normalization story.
- **MCC could be a category oracle.** If each category mapped to a disjoint MCC
  set, inverting MCC would hand back the label. **Control:** the synthetic MCCs
  overlap across categories and ~15% are drawn from a wrong category, so MCC is a
  strong-but-imperfect prior; and `rules` *learns* the MCC→category map from
  training data rather than reading the generative table.
- **Direction is trivial.** It is recoverable from the amount sign by
  construction, so it is not used to differentiate routes.

## Notes

- All randomness is seeded (`SEED = 42`); runs are reproducible.
- `eval` scores every route on a common capped subset of the held-out split
  (`--eval-cap`, default 1000) so the leaderboard is a fair like-for-like
  comparison and the slow local-SLM route stays feasible. `rules`/`embedding`
  could run on the full split, but all routes share the same rows for fairness.
- The source dataset has real label noise (the same merchant appears under
  different categories), which caps achievable category accuracy — merchant
  normalization is evaluated independently for this reason.
- Fine-tuning is Apple-Silicon native via MLX (no CUDA): the `slm_lora` route
  trains a Qwen2.5-3B LoRA adapter with `scripts/train_lora.sh`. See the
  generalization trade-off in `docs/findings/lora-vs-fewshot.md`.
