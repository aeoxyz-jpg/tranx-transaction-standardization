# Significance

These CIs describe a hypothetical population of merchants like these (the same generator, or the same one person's statements); they are not a claim about other customers.

MoneyData: 168 low-confidence labels excluded (kept confidence: high, medium).

## Primary

| comparison | diff | lo | hi | n_clusters | n_rows |
| --- | --- | --- | --- | --- | --- |
| jev_slm - slm_fewshot, merchant, synthetic unseen | -0.041 | -0.071 | -0.017 | 231 | 2374 |
| jev_slm - slm, merchant, MoneyData realistic list | +0.044 | +0.014 | +0.094 | 261 | 547 |

MoneyData row-weighted point estimate (Amazon included): +0.043.
MoneyData row-weighted CI (Amazon excluded): +0.065 [+0.018, +0.130].
MoneyData leave-one-merchant-out range: [+0.037, +0.063].

## Exploratory

Not pre-registered; DDT clusters by row (descriptor-paired), not by merchant, so template correlation across rows is not controlled.

| comparison | diff | lo | hi | n_clusters | n_rows |
| --- | --- | --- | --- | --- | --- |
| jev_merchant - embedding, merchant, synthetic random | +0.090 | +0.069 | +0.113 | 501 | 1139 |
| jev_merchant - embedding, merchant, synthetic unseen | +0.201 | +0.183 | +0.220 | 231 | 2374 |
| embedding - jev_merchant, category, synthetic random | -0.009 | -0.038 | +0.021 | 619 | 1496 |
| embedding - jev_merchant, category, synthetic unseen | -0.281 | -0.344 | -0.217 | 334 | 2833 |
| jev - embedding, merchant, MoneyData full list | +0.080 | +0.014 | +0.126 | 261 | 547 |
| embedding - jev, category, DoDataThings | +0.096 | +0.067 | +0.126 | 1000 | 1000 |
