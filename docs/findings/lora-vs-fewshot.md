# Finding: LoRA fine-tune vs few-shot (same base, hard feed)

**Date:** 2026-06-13
**Setup:** Qwen2.5-3B-Instruct (4-bit), hard feed, `--eval-cap 1000`, both splits.
LoRA trained with MLX on Apple M4 (`scripts/train_lora.sh`): 3600 examples from the
**unseen-split train** portion (so eval-unseen merchants are never seen in training),
600 iters, lr 2e-4, 8 layers, mask-prompt. Val loss 2.56 -> 0.048, peak mem 4.0 GB.

| split | metric | few-shot | LoRA | Δ |
|---|---|---:|---:|---:|
| random | Category | 0.71 | **0.82** | +0.11 |
| random | Merchant Norm | 0.81 | **0.87** | +0.06 |
| unseen | Category | 0.72 | **0.76** | +0.04 |
| unseen | Merchant Norm | 0.78 | **0.69** | **−0.09** |
| — | latency (ms/txn) | ~575 | ~712 | +impl. overhead |

## The non-obvious result

LoRA improves everything **except** unseen-merchant normalization, where it
**regresses** below few-shot (0.69 vs 0.78). Fine-tuning is not a free win.

Why:
- **Category is learnable and generalizes** — the LoRA gains on category hold on
  both splits (the mapping from surviving tokens to one of 10 classes is exactly
  what supervised fine-tuning is good at).
- **Seen-merchant normalization is memorized** — random-split merchants overlap
  the LoRA training merchants, so random Merchant Norm jumps to 0.87.
- **Unseen-merchant normalization is base-model world knowledge** — few-shot
  leans on Qwen's pretrained knowledge of real brands (Home Depot, Greenpeace…).
  Fine-tuning on 3600 of *our* merchants (val loss 0.048 = somewhat overfit)
  narrows the model toward the training distribution and erodes the very
  out-of-distribution generalization that made the SLM valuable. A mild
  catastrophic-forgetting effect.

## Takeaways

- **If the merchant set is stable / in-distribution:** LoRA is the better choice —
  higher category and seen-merchant accuracy at the same model size.
- **If new merchants keep appearing (the real bank scenario):** few-shot preserves
  the base model's generalization; LoRA can trade it away.
- **To keep both:** lighter fine-tuning (fewer iters, lower LR, more merchant
  diversity, or mixing in general data) should reduce the unseen regression — the
  0.048 val loss signals over-fitting headroom to give back.
- **Latency** (712 vs 575 ms) is implementation overhead (in-process `mlx_lm.generate`
  vs the optimized Ollama server), not fundamental; fusing the adapter to GGUF and
  serving via Ollama would close the gap.

## Reproduce

```bash
tranx synth --n 100000 --hard
tranx finetune-export --cap 4000          # data/ft from the unseen-split train
./scripts/train_lora.sh                   # adapters/qwen-hard
tranx run --route slm_lora --split unseen --eval-cap 1000
tranx run --route slm_lora --split random --eval-cap 1000
```
