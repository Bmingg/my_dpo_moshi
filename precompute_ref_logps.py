"""
Precompute reference model log-probabilities for DPO.

Runs the base Moshi model over the entire SpokenSwag dataset once,
saves per-example (chosen_logp, rejected_logp) to disk.

The training script loads these and uses them instead of running the
reference model during training.

Usage:
    python precompute_ref_logps.py

Tunables (env vars):
    MOSHI_REPO                  (default kmhf/hf-moshiko)
    REF_LOGPS_PATH              (default /workspace/ref_logps.pt)
    PRECOMPUTE_BATCH_SIZE       (default 2)
    MAX_PROMPT_SEC              (MUST match training)
    MAX_COMPLETION_SEC          (MUST match training)
"""

from __future__ import annotations

import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset, Audio
from transformers import MoshiForConditionalGeneration

from moshi_dpo_collator import SpokenSwagMoshiCollator
from moshi_dpo_trainer import compute_side_logps

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute_ref")


# ======== Config (MUST match training script) ========
MOSHI_REPO          = os.environ.get("MOSHI_REPO", "kmhf/hf-moshiko")
DATASET_REPO        = "slprl/SpokenSwag"
OUTPUT_PATH         = os.environ.get("REF_LOGPS_PATH", "/workspace/ref_logps.pt")
BATCH_SIZE          = int(os.environ.get("PRECOMPUTE_BATCH_SIZE", 2))

# These two values MUST match the training script exactly.
MAX_PROMPT_SEC      = float(os.environ.get("MAX_PROMPT_SEC", 3.0))
MAX_COMPLETION_SEC  = float(os.environ.get("MAX_COMPLETION_SEC", 5.0))

# These must also match the training script's MoshiDPOTrainer args.
SEMANTIC_LOSS_WEIGHT = 7.0
ACOUSTIC_LOSS_WEIGHT = 1.0
TEXT_LOSS_WEIGHT     = 1.0
USE_TEXT_ALIGNMENT   = False

# Dataset filtering (should also match training)
MAX_BLEU = 0.3


def main():
    device = "cuda"
    logger.info("Config: prompt=%.1fs, completion=%.1fs, batch=%d", 
                MAX_PROMPT_SEC, MAX_COMPLETION_SEC, BATCH_SIZE)
    logger.info("Output: %s", OUTPUT_PATH)

    # ---- 1. Load model ----
    logger.info("Loading model %s", MOSHI_REPO)
    model = MoshiForConditionalGeneration.from_pretrained(
        MOSHI_REPO, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # Resolve audio_bos consistently with the trainer
    audio_bos = model.config.audio_vocab_size
    logger.info("Resolved audio_bos = %d", audio_bos)

    # ---- 2. Load dataset (same filtering as training) ----
    logger.info("Loading dataset %s", DATASET_REPO)
    ds = load_dataset(DATASET_REPO)

    audio_feat = Audio(sampling_rate=24000)
    for col in ("prompt", "chosen", "rejected"):
        ds = ds.cast_column(col, audio_feat)

    ds = ds.filter(
        lambda x: x["auto_bleu2"] is None or x["auto_bleu2"] < MAX_BLEU,
        num_proc=4,
    )
    for split, d in ds.items():
        logger.info("  %s: %d examples after filter", split, len(d))

    # ---- 3. Build collator ----
    collator = SpokenSwagMoshiCollator(
        mimi_model=model.audio_encoder,
        sampling_rate=24000,
        max_prompt_seconds=MAX_PROMPT_SEC,
        max_completion_seconds=MAX_COMPLETION_SEC,
        use_text_alignment=USE_TEXT_ALIGNMENT,
    )

    # ---- 4. Run precompute for each split ----
    results = {}
    for split_name, split_ds in ds.items():
        logger.info("=" * 60)
        logger.info("Precomputing split: %s (%d examples)", split_name, len(split_ds))

        loader = DataLoader(
            split_ds,
            batch_size=BATCH_SIZE,
            collate_fn=collator,
            shuffle=False,       # CRITICAL: must not shuffle
            num_workers=0,       # collator uses GPU Mimi
        )

        chosen_logps = []
        rejected_logps = []

        for batch in tqdm(loader, desc=f"  {split_name}"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            with torch.no_grad():
                c = compute_side_logps(
                    model,
                    moshi_codes=batch["chosen_moshi_audio_codes"],
                    user_codes=batch["chosen_user_audio_codes"],
                    text_ids=batch["chosen_input_ids"],
                    attention_mask=batch.get("chosen_attention_mask"),
                    completion_mask=batch["chosen_completion_mask"],
                    audio_bos=audio_bos,
                    semantic_loss_weight=SEMANTIC_LOSS_WEIGHT,
                    acoustic_loss_weight=ACOUSTIC_LOSS_WEIGHT,
                    text_loss_weight=TEXT_LOSS_WEIGHT,
                    use_text_alignment=USE_TEXT_ALIGNMENT,
                )
                r = compute_side_logps(
                    model,
                    moshi_codes=batch["rejected_moshi_audio_codes"],
                    user_codes=batch["rejected_user_audio_codes"],
                    text_ids=batch["rejected_input_ids"],
                    attention_mask=batch.get("rejected_attention_mask"),
                    completion_mask=batch["rejected_completion_mask"],
                    audio_bos=audio_bos,
                    semantic_loss_weight=SEMANTIC_LOSS_WEIGHT,
                    acoustic_loss_weight=ACOUSTIC_LOSS_WEIGHT,
                    text_loss_weight=TEXT_LOSS_WEIGHT,
                    use_text_alignment=USE_TEXT_ALIGNMENT,
                )

            chosen_logps.extend(c.float().cpu().tolist())
            rejected_logps.extend(r.float().cpu().tolist())

        assert len(chosen_logps) == len(split_ds), (
            f"Length mismatch: got {len(chosen_logps)} logps for {len(split_ds)} examples"
        )
        results[split_name] = {
            "chosen": chosen_logps,
            "rejected": rejected_logps,
        }

    # ---- 5. Save ----
    logger.info("Saving to %s", OUTPUT_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(results, OUTPUT_PATH)
    logger.info("Done.")


if __name__ == "__main__":
    main()