#!/usr/bin/env bash
set -euo pipefail
# Fine-tune Qwen2.5-3B (4-bit) with LoRA on the hard-mode standardization task.
# Apple Silicon native via MLX. Run `tranx finetune-export` first to build data/ft.
#
# Env overrides: BASE, DATA, ADAPTERS, ITERS.
BASE="${BASE:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
DATA="${DATA:-data/ft}"
ADAPTERS="${ADAPTERS:-adapters/qwen-hard}"
ITERS="${ITERS:-600}"

python3.11 -m mlx_lm lora \
  --model "$BASE" \
  --train \
  --data "$DATA" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers 8 \
  --batch-size 4 \
  --max-seq-length 512 \
  --iters "$ITERS" \
  --learning-rate 2e-4 \
  --steps-per-report 25 \
  --steps-per-eval 150 \
  --adapter-path "$ADAPTERS" \
  --seed 42
echo "adapter written to $ADAPTERS"
