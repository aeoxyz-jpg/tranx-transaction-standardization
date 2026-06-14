# Finding: hard-mode (dirty descriptors) results

**Date:** 2026-06-13
**Setup:** 100k feed, `--eval-cap 1000`, both splits. `synth --hard` dirties the
feed descriptors (aggregator prefixes, uppercasing, embedded geo/store-id,
truncation, space collapsing); gold labels stay derived from the clean source.

## Standard vs hard (Merchant Acc / Category Acc)

| split | route | Merchant std → hard | Category std → hard |
|---|---|---|---|
| random | rules | 1.00 → **0.22** | 0.86 → 0.56 |
| random | embedding | 0.99 → **0.77** | 0.98 → 0.92 |
| random | slm_fewshot | 0.86 → **0.26*** | 0.72 → 0.59 |
| unseen | rules | 0.94 → **0.04** | 0.42 → 0.41 |
| unseen | embedding | 0.00 → 0.00 | 0.81 → 0.68 |
| unseen | slm_fewshot | 0.86 → **0.24*** | 0.69 → 0.62 |

## What hard mode revealed

1. **The silver-label tautology is destroyed.** `rules` unseen merchant collapses
   0.94 → 0.04: its `derive_canonical` fallback can no longer recover the clean
   gold name from a dirty descriptor. The clean-data 0.94 really was an artifact.

2. **Embedding retrieval is the most robust exact-match merchant route on the
   random split (0.77).** Semantic nearest-neighbor against the train vocabulary
   tolerates dirt — but it is still vocabulary-locked (0.00 on unseen).

3. **Exact-match Merchant Acc badly understates the SLM (the `*`).** A sample of
   SLM predictions shows most "misses" are case/punctuation near-misses that are
   semantically correct:
   - `PARAMEDIC` vs gold `Paramedic`, `SOCIAL SECURITY` vs `Social Security`
   - `Raising Canes` vs `Raising Cane's`, `CANES` vs `Cane's`
   - genuinely parsed-hard wins: `SunTrustSEATTLEAZ` → `SunTrust`,
     `POSDEBITGREENPEACE` → `Greenpeace`, `HERTZ 9022898740` → `Hertz`
   A normalized (case-insensitive, punctuation-stripped, or fuzzy) merchant
   match would credit these and is the fairer metric for generative output.
   Embedding, which emits exact train-vocabulary strings, would barely change.

4. **The genuine residual is the aggregator-prefix problem.** When the descriptor
   is `AMZN MKTP HOME DEPOT ...` or `PAYPAL *CAPITAL ONE ...`, the SLM tends to
   return the *processor* (`AMZN`, `PAYPAL`) instead of the merchant. This is the
   real industry-hard case (Square/Toast/PayPal route many merchants) and cannot
   be solved from the string alone — it needs a merchant database / search for
   grounding, which is exactly the moat pure-play enrichment vendors build.

## Implications / next steps

- Add a **normalized merchant-match metric** alongside exact-match to fairly
  score generative output. Highest-value, low-effort.
- Consider an **aggregator-prefix-aware** few-shot example or a retrieval-grounded
  SLM variant to address residual (4).
- Category classification via embedding+LR stays the most robust under dirt
  (0.92 random / 0.68 unseen), because category is learnable from the surviving
  tokens even when the exact merchant is mangled.

## Update — after the two fixes (normalized metric + processor-prefix stripping)

Added a normalized merchant metric (case/punctuation-insensitive) and a shared
processor-prefix stripper (`AMZN MKTP X` -> `X`) used by all routes, plus an
aggregator-aware SLM prompt. Hard-mode merchant scores, before -> after:

| split | route | Merchant exact | Merchant norm |
|---|---|---|---|
| random | rules | 0.22 -> 0.36 | 0.36 |
| random | embedding | 0.77 -> 0.90 | 0.90 |
| random | slm_fewshot | 0.26 -> **0.70** | **0.81** |
| unseen | rules | 0.04 -> 0.06 | 0.26 |
| unseen | embedding | 0.00 -> 0.00 | 0.00 |
| unseen | slm_fewshot | 0.24 -> **0.68** | **0.78** |

Conclusions:
- The SLM was the biggest beneficiary: exact merchant ~3x (0.26 -> 0.70 random),
  and the normalized metric recovers another ~10pt (0.81) of casing/punctuation
  near-misses the exact metric had been discarding.
- On the realistic case — **dirty descriptors + unseen merchants** — the SLM is
  the only route that generalizes: Merchant Norm 0.78 and Category 0.72, versus
  embedding's 0.00 merchant (vocabulary-locked) and rules' 0.26. This is the
  payoff the whole evaluation was built to expose, now measured fairly.
- Embedding remains the random-split champion (retrieval is robust to dirt) and
  the category champion on random (0.92), but cannot emit unseen merchants.
- exact == norm for embedding confirms it emits exact vocabulary strings; the gap
  exists only for the generative SLM.

## Update — giving `rules` a fair hard-mode scorer (token_set_ratio)

`rules` originally used `fuzz.ratio`, which over-penalizes a short merchant name
against a long dirty string. Switching to `fuzz.token_set_ratio` (no synth noise
vocabulary enumerated — purely a more tolerant string scorer) on hard mode:

| split | metric | ratio -> token_set_ratio |
|---|---|---|
| random | Category | 0.62 -> 0.78 |
| random | Merchant Norm | 0.36 -> 0.74 |
| unseen | Category | 0.41 -> 0.57 |
| unseen | Merchant Norm | 0.26 -> **0.03** |

- On the **random** split this is a large, fair win: embedded city/state/store-id
  tokens no longer sink the match, so `rules` becomes competitive (0.74) with
  embedding (0.90) on in-distribution dirty data.
- On the **unseen** split the merchant score *drops* to ~0. The more tolerant
  scorer now confidently matches dirty queries to a *wrong* known merchant instead
  of falling back to `derive_canonical`, removing the last of the tautological
  crutch. This is the honest result: a learned-vocabulary matcher cannot produce a
  merchant it never saw. Only the SLM (Norm 0.78) generalizes.
- Lesson for the baseline: tuning the scorer buys in-distribution robustness but
  not generalization — exactly the ceiling the unseen split exists to expose.
