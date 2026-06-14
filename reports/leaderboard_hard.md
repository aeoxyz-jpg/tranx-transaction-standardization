# Route Leaderboard — HARD mode (dirty card-network descriptors)

Feed synthesized with `synth --hard`: aggregator prefixes (`SQ */AMZN MKTP/PAYPAL *`),
uppercasing, embedded city/state/store-id, truncation, space collapsing. Includes
the processor-prefix preprocessing, the normalized merchant metric, and the
token-set fuzzy scorer for `rules`.

`Merchant Acc` is exact string match; `Merchant Norm` ignores case/punctuation.
`rules` now tolerates embedded tokens on the random split (token_set_ratio), but
its unseen-merchant score is ~0 — you cannot hand-rule your way to generalization.
Only the SLM holds up on the unseen split (parses instead of matching a vocabulary).

| route | split | Category Acc | Macro F1 | Merchant Acc | Merchant Norm | Dedup Ratio | Spend KPI | ms/txn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rules | random | 0.78 | 0.78 | 0.72 | 0.74 | 0.57 | 0.72 | 0.50 |
| embedding | random | 0.92 | 0.92 | 0.90 | 0.90 | 0.48 | 0.90 | 8.25 |
| slm_fewshot | random | 0.71 | 0.72 | 0.70 | 0.81 | 0.55 | 0.70 | 570.89 |
| rules | unseen | 0.57 | 0.61 | 0.00 | 0.03 | 0.80 | 0.00 | 0.81 |
| embedding | unseen | 0.68 | 0.69 | 0.00 | 0.00 | 0.31 | 0.00 | 8.13 |
| slm_fewshot | unseen | 0.72 | 0.68 | 0.67 | 0.78 | 0.28 | 0.67 | 581.88 |
