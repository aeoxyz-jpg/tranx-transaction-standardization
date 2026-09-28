# Significance

These CIs describe a hypothetical population of merchants like these (the same generator, or the same one person's statements); they are not a claim about other customers.

MoneyData: 168 low-confidence labels excluded (kept confidence: high, medium).

## Primary

| comparison | diff | lo | hi | n_clusters | n_rows |
| --- | --- | --- | --- | --- | --- |
| jev_slm - slm_fewshot, merchant, synthetic unseen | -0.018 | -0.045 | +0.001 | 213 | 1963 |
| jev_slm - slm, merchant, MoneyData realistic list | +0.049 | +0.019 | +0.101 | 261 | 547 |

MoneyData row-weighted point estimate (Amazon included): +0.044.
MoneyData row-weighted CI (Amazon excluded): +0.067 [+0.020, +0.132].
MoneyData leave-one-merchant-out range: [+0.042, +0.071].

## Exploratory

Not pre-registered; DDT clusters by row (descriptor-paired), not by merchant, so template correlation across rows is not controlled.

| comparison | diff | lo | hi | n_clusters | n_rows |
| --- | --- | --- | --- | --- | --- |
| jev_merchant - embedding, merchant, synthetic random | +0.095 | +0.068 | +0.122 | 393 | 866 |
| jev_merchant - embedding, merchant, synthetic unseen | +0.207 | +0.190 | +0.225 | 213 | 1963 |
| embedding - jev_merchant, category, synthetic random | +0.006 | -0.024 | +0.037 | 586 | 1483 |
| embedding - jev_merchant, category, synthetic unseen | -0.277 | -0.342 | -0.211 | 392 | 2665 |
| jev - embedding, merchant, MoneyData full list | +0.080 | +0.014 | +0.126 | 261 | 547 |
| embedding - jev, category, DoDataThings | +0.096 | +0.067 | +0.126 | 1000 | 1000 |
