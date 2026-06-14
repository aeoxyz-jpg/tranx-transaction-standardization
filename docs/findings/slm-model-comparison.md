# Finding: SLM model comparison (qwen2.5:3b vs gemma2:2b)

**Date:** 2026-06-13
**Setup:** same 100k synthetic feed, `--eval-cap 1000`, both splits, local Ollama, temperature 0.

| split | model | Category Acc | Merchant Acc | Spend KPI | ms/txn |
|---|---|---|---|---|---|
| random | qwen2.5:3b-instruct | 0.72 | 0.86 | 0.86 | 556 |
| random | gemma2:2b | 0.715 | 0.81 | 0.81 | 659 |
| unseen | qwen2.5:3b-instruct | 0.69 | 0.86 | 0.86 | 561 |
| unseen | gemma2:2b | 0.67 | 0.81 | 0.81 | 807 |

## Takeaways

1. **qwen2.5:3b wins on this task.** Better merchant normalization (~+5pt on both
   splits) and marginally better category; the 3B model is the stronger local choice.
2. **Smaller is not faster here.** gemma2:2b (2B) is *slower* than qwen2.5:3b (3B):
   659/807 ms vs 556/561 ms. Latency is dominated by **output token count**, not
   parameter count — gemma2:2b tends to emit more text around the JSON, so it
   decodes more tokens per transaction. Lesson: benchmark latency on the actual
   output behaviour, do not assume fewer params means faster.
3. **The generalization advantage is model-agnostic.** Both SLMs hold their
   merchant accuracy nearly flat from the random to the unseen-merchant split
   (qwen 0.86→0.86, gemma 0.81→0.81). The "parse, don't memorize a vocabulary"
   property is a trait of the few-shot SLM approach, not of one specific model.

## Decision

Keep qwen2.5:3b-instruct as the default SLM. gemma2:2b is supported via
`run --route slm_fewshot --slm-model gemma2:2b` but is not a permanent
leaderboard row. The `--slm-model` flag makes swapping models a one-line change.
