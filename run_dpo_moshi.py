"""
Moshi DPO training script — audio-only (Option A, text stream = all PAD).

Key differences from standard text DPO:
  1. Custom collator encodes raw audio → Mimi codes (not text tokenization).
  2. MoshiDPOTrainer inherits from transformers.Trainer (no TRL dependency).
  3. compute_loss is overridden to handle dual-stream audio + depth decoder.
  4. remove_unused_columns=False is CRITICAL — otherwise HF Trainer drops the
     audio columns before the collator ever sees them.
  5. Full fine-tuning (no LoRA). Ref model created via deepcopy inside trainer.

Memory budget (full fine-tuning, bf16, H200 141GB):
  Model weights (bf16):           ~14 GB
  Ref model (bf16, frozen):       ~14 GB
  Gradients (bf16):               ~14 GB
  8-bit Adam states:              ~14 GB  (standard AdamW would be ~56 GB — OOM!)
  Activations (depth decoder):    ~40-50 GB  (8 codebooks × T frames)
  -----------------------------------------------
  Total:                          ~96-106 GB  ← fits on H200 141GB with ~35GB headroom

Usage (local):
  python run_dpo_moshi.py

Usage (sweep via env vars on Modal):
  MOSHI_DPO_LR=1e-6 MOSHI_DPO_BETA=0.1 modal run --detach infra/modal_app.py

Tunable env vars (all optional, with sensible defaults):
  MOSHI_DPO_OUTPUT_DIR          default: ./moshi-dpo-checkpoints
  MOSHI_DPO_LR                 default: 1e-6
  MOSHI_DPO_BETA               default: 0.1
  MOSHI_DPO_SEMANTIC_WEIGHT    default: 7.0
  MOSHI_DPO_EPOCHS             default: 2
  MOSHI_DPO_MAX_STEPS          default: (none — use epochs instead)
  MOSHI_DPO_BATCH_SIZE         default: 1
  MOSHI_DPO_GRAD_ACCUM         default: 8
  MOSHI_DPO_TRAIN_TAKE         default: (none — use full dataset)
  MOSHI_DPO_VAL_TAKE           default: (none)

Metrics logged (visible in W&B or TensorBoard):
  rewards/chosen                mean reward of chosen completions
  rewards/rejected              mean reward of rejected completions
  rewards/margins               chosen - rejected (should increase > 0)
  rewards/accuracies            fraction where chosen > rejected (should → 1)
  logps/chosen                  mean log-prob of chosen completions
  logps/rejected                mean log-prob of rejected completions
"""

from __future__ import annotations

import logging
import os
import sys
import torch

from datasets import load_dataset, Audio
from transformers import AutoTokenizer, TrainingArguments

from moshi_dpo_collator import (
    SpokenSwagMoshiCollator,
    MOSHI_TEXT_PAD_ID,
)
from moshi_dpo_trainer import MoshiDPOTrainer

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("moshi_dpo")


# ========================== helpers ==========================

def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

def _env_int_or_none(key: str) -> int | None:
    v = os.environ.get(key)
    return int(v) if v is not None else None


# ========================== config ==========================

MOSHI_REPO      = "kmhf/hf-moshiko"
DATASET_REPO    = "slprl/SpokenSwag"
OUTPUT_DIR      = os.environ.get("MOSHI_DPO_OUTPUT_DIR", "./moshi-dpo-checkpoints")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
TRAIN_TAKE      = _env_int_or_none("MOSHI_DPO_TRAIN_TAKE")
VAL_TAKE        = _env_int_or_none("MOSHI_DPO_VAL_TAKE")
MAX_BLEU        = 0.3

# Audio truncation (None = no truncation, use full audio)
MAX_PROMPT_SEC      = None
MAX_COMPLETION_SEC  = None

# Training
BATCH_SIZE      = _env_int("MOSHI_DPO_BATCH_SIZE", 1)
GRAD_ACCUM      = _env_int("MOSHI_DPO_GRAD_ACCUM", 8)     # effective batch = 8
LEARNING_RATE   = _env_float("MOSHI_DPO_LR", 1e-6)
NUM_EPOCHS      = _env_int("MOSHI_DPO_EPOCHS", 2)
MAX_STEPS       = _env_int_or_none("MOSHI_DPO_MAX_STEPS")  # optional override
WARMUP_RATIO    = 0.1
WEIGHT_DECAY    = 0.1
MAX_GRAD_NORM   = 1.0
BF16            = True

# DPO
BETA            = _env_float("MOSHI_DPO_BETA", 0.1)

# Loss weights (Option A: audio-only, text_loss_weight is irrelevant)
SEMANTIC_LOSS_WEIGHT = _env_float("MOSHI_DPO_SEMANTIC_WEIGHT", 7.0)
ACOUSTIC_LOSS_WEIGHT = 1.0

# Monitoring
REPORT_TO = os.environ.get("MOSHI_DPO_REPORT_TO", "wandb")


def main():
    logger.info("=" * 60)
    logger.info("Moshi DPO Config:")
    logger.info("  lr=%.2e, beta=%.2f, semantic_w=%.1f", LEARNING_RATE, BETA, SEMANTIC_LOSS_WEIGHT)
    logger.info("  batch=%d, grad_accum=%d (eff_batch=%d)", BATCH_SIZE, GRAD_ACCUM, BATCH_SIZE * GRAD_ACCUM)
    logger.info("  epochs=%d, max_steps=%s, max_grad_norm=%.1f", NUM_EPOCHS, MAX_STEPS or "auto", MAX_GRAD_NORM)
    logger.info("  optimizer=adamw_bnb_8bit")
    logger.info("  report_to=%s", REPORT_TO)
    logger.info("  output_dir=%s", OUTPUT_DIR)
    logger.info("=" * 60)

    # ========================== 1. Load model ==========================
    logger.info("Loading model: %s", MOSHI_REPO)
    from transformers import MoshiForConditionalGeneration

    model = MoshiForConditionalGeneration.from_pretrained(
        MOSHI_REPO,
        torch_dtype=torch.bfloat16,
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    # ========================== 2. Extract Mimi for collator ==========================
    # Mimi freezing and ref model creation happen inside MoshiDPOTrainer.__init__
    mimi = model.audio_encoder

    # ========================== 3. Load dataset ==========================
    logger.info("Loading dataset: %s", DATASET_REPO)
    ds = load_dataset(DATASET_REPO)

    audio_feat = Audio(sampling_rate=24000)
    for col in ("prompt", "chosen", "rejected"):
        ds = ds.cast_column(col, audio_feat)

    ds = ds.filter(
        lambda x: x["auto_bleu2"] is None or x["auto_bleu2"] < MAX_BLEU,
        num_proc=4,
    )

    if TRAIN_TAKE is not None and "train" in ds:
        ds["train"] = ds["train"].select(range(min(TRAIN_TAKE, len(ds["train"]))))
    if VAL_TAKE is not None and "validation" in ds:
        ds["validation"] = ds["validation"].select(range(min(VAL_TAKE, len(ds["validation"]))))

    for split, d in ds.items():
        logger.info("  %s: %d examples", split, len(d))

    # ========================== 4. Collator ==========================
    collator = SpokenSwagMoshiCollator(
        mimi_model=mimi,
        sampling_rate=24000,
        max_prompt_seconds=MAX_PROMPT_SEC,
        max_completion_seconds=MAX_COMPLETION_SEC,
        use_text_alignment=False,
        text_pad_id=MOSHI_TEXT_PAD_ID,
        device=DEVICE,
    )

    # ========================== 5. Training args ==========================
    tokenizer = AutoTokenizer.from_pretrained(MOSHI_REPO)
    has_eval = "validation" in ds and ds.get("validation") is not None

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Batch / steps
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS is not None else -1,

        # Optimizer — 8-bit Adam to fit full FT + ref model in H200 141GB
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_bnb_8bit",

        # Precision
        bf16=BF16,

        # Logging / saving
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch" if has_eval else "no",
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,

        # CRITICAL
        remove_unused_columns=False,
        dataloader_num_workers=0,       # collator runs Mimi on GPU
        dataloader_pin_memory=False,    # tensors already on GPU from Mimi

        # Monitoring
        report_to=REPORT_TO,

        # Misc
        seed=42,
    )

    # ========================== 6. Trainer ==========================
    trainer = MoshiDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=collator,
        processing_class=tokenizer,

        # DPO
        beta=BETA,

        # Moshi-specific
        use_text_alignment=False,
        text_loss_weight=1.0,
        semantic_loss_weight=SEMANTIC_LOSS_WEIGHT,
        acoustic_loss_weight=ACOUSTIC_LOSS_WEIGHT,
    )

    # ========================== 7. Train ==========================
    logger.info("Starting training...")
    trainer.train()

    # ========================== 8. Save ==========================
    final_dir = os.path.join(OUTPUT_DIR, "final")
    logger.info("Saving final model to %s", final_dir)
    trainer.save_model(final_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()