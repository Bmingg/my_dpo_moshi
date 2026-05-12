"""
Moshi DPO training script — audio-only (Option A, text stream = all PAD).

Key differences from standard text DPO:
  1. Custom collator encodes raw audio → Mimi codes (not text tokenization).
  2. MoshiDPOTrainer inherits from transformers.Trainer (no TRL dependency).
  3. compute_loss is overridden to handle dual-stream audio + depth decoder.
  4. remove_unused_columns=False is CRITICAL — otherwise HF Trainer drops the
     audio columns before the collator ever sees them.
  5. Full fine-tuning (no LoRA). Ref model created via deepcopy inside trainer.


Usage (sweep via env vars on Modal):
  MOSHI_DPO_LR=1e-6 MOSHI_DPO_BETA=0.1 modal run --detach infra/modal_app.py
  MOSHI_DPO_MAX_STEPS=200 MOSHI_DPO_BATCH_SIZE=4 MOSHI_DPO_LR=1e-6 MOSHI_DPO_BETA=0.1 modal run --detach infra/modal_app.py
  MOSHI_DPO_OUTPUT_DIR=/vol/moshi-dpo-checkpoints/lr5e-5_b1e-1_sem100 MODAL_GPU_TYPE=H200 MODAL_NUM_GPUS=4 MOSHI_DPO_BATCH_SIZE=4 MOSHI_DPO_GRAD_ACCUM=4 MOSHI_DPO_LR=5e-5 MOSHI_DPO_EPOCHS=3 MOSHI_DPO_BETA=0.1 MOSHI_DPO_SEMANTIC_WEIGHT=100.0 modal run --detach infra/modal_app.py
  MOSHI_DPO_OUTPUT_DIR=/vol/moshi-dpo-checkpoints/lr5e-6_b1e-1_sem100 MODAL_GPU_TYPE=H200 MODAL_NUM_GPUS=4 MOSHI_DPO_BATCH_SIZE=4 MOSHI_DPO_GRAD_ACCUM=4 MOSHI_DPO_LR=5e-6 MOSHI_DPO_EPOCHS=3 MOSHI_DPO_BETA=0.1 MOSHI_DPO_SEMANTIC_WEIGHT=100.0 modal run --detach infra/modal_app.py

Tunable env vars (all optional, with sensible defaults):
  MOSHI_DPO_OUTPUT_DIR          default: ./moshi-dpo-checkpoints

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
import torch.nn.functional as F
from transformers.models.moshi.modeling_moshi import MoshiFlexibleLinear

from datasets import load_dataset, Audio
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model


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

# Override MoshiFlexibleLinear.forward with a memory-efficient implementation that avoids large intermediate tensors. This is a workaround for OOM errors in the depth decoder's flexible linears
def patch_moshi_flexible_linear():
    def memory_safe_forward(self, x, layer_idx=None):
        """
        Memory-safe replacement for MoshiFlexibleLinear.forward.

        Supports both:
          1. x: [B, K, D], weights: [K, O, D] -> [B, K, O]
          2. x: [B, 1, D], weights: [K, O, D] -> [B, K, O]
             This is the broadcast case used by Moshi input_projections.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, S, D], got {tuple(x.shape)}")

        B, S, D = x.shape

        # Select weights.
        # self.weight: [num_layers/codebooks, output_size, input_size]
        if layer_idx is None:
            weights = self.weight
        elif isinstance(layer_idx, int):
            return F.linear(x, self.weight[layer_idx])
        elif isinstance(layer_idx, torch.Tensor) and layer_idx.dim() == 0:
            return F.linear(x, self.weight[int(layer_idx.item())])
        else:
            layer_idx = layer_idx.to(device=self.weight.device, dtype=torch.long)
            weights = torch.index_select(self.weight, 0, layer_idx)

        # weights: [K, O, D]
        K, O, D_weight = weights.shape
        if D != D_weight:
            raise ValueError(
                f"Input dim mismatch: x.shape={tuple(x.shape)}, "
                f"weights.shape={tuple(weights.shape)}"
            )

        # Case A: one x vector per codebook.
        # Original behavior: each x[:, i, :] uses weights[i].
        if S == K:
            outs = [
                F.linear(x[:, i, :], weights[i])
                for i in range(K)
            ]
            return torch.stack(outs, dim=1)  # [B, K, O]

        # Case B: one x vector broadcast to all codebooks.
        # This is your failing case: x=(T, 1, 4096), weights=(8, 1024, 4096).
        if S == 1:
            x0 = x[:, 0, :]  # [B, D]
            outs = [
                F.linear(x0, weights[i])
                for i in range(K)
            ]
            return torch.stack(outs, dim=1)  # [B, K, O]

        raise ValueError(
            f"Unsupported MoshiFlexibleLinear shape combination: "
            f"x.shape={tuple(x.shape)}, weights.shape={tuple(weights.shape)}, "
            f"layer_idx={layer_idx}"
        )

    MoshiFlexibleLinear.forward = memory_safe_forward
    print("Patched MoshiFlexibleLinear.forward with memory-safe implementation.")


patch_moshi_flexible_linear()


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
MAX_PROMPT_SEC      = _env_int_or_none("MAX_PROMPT_SEC")
MAX_COMPLETION_SEC  = _env_int_or_none("MAX_COMPLETION_SEC")

# Training
_TARGET_EFF_BATCH = 64
_world_size = int(os.environ.get("WORLD_SIZE", 1))
_per_device = _env_int("MOSHI_DPO_BATCH_SIZE", 1)
_DEFAULT_GRAD_ACCUM = max(1, _TARGET_EFF_BATCH // (_world_size * _per_device))
GRAD_ACCUM = _env_int("MOSHI_DPO_GRAD_ACCUM", _DEFAULT_GRAD_ACCUM)
BATCH_SIZE = _per_device

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
    # DDP rank detection (set by torchrun). 0 in single-GPU runs.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = local_rank == 0
    
    # Silence non-rank-0 logs to avoid 4× duplicates.
    if not is_main_process:
        logger.setLevel(logging.WARNING)
    
    if is_main_process:
        logger.info("DDP setup: local_rank=%d, world_size=%d", local_rank, world_size)
    
    logger.info("=" * 60)
    logger.info("Moshi DPO Config:")
    logger.info("  lr=%.2e, beta=%.2f, semantic_w=%.1f", LEARNING_RATE, BETA, SEMANTIC_LOSS_WEIGHT)

    # ===== W&B init: hardcoded project, name derived from actual hyperparams =====
    # Must run BEFORE TrainingArguments is constructed, so HF Trainer reuses
    # this run instead of creating its own.
    if is_main_process and REPORT_TO == "wandb":
        import wandb
        run_name = f"text-less_{LEARNING_RATE:.0e}_{BETA}_{SEMANTIC_LOSS_WEIGHT}"
        wandb.init(
            project="moshi_DPO_1",
            name=run_name,
            config={
                "learning_rate": LEARNING_RATE,
                "beta": BETA,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "epochs": NUM_EPOCHS,
                "semantic_loss_weight": SEMANTIC_LOSS_WEIGHT,
                "world_size": world_size,
                "effective_batch_size": BATCH_SIZE * world_size * GRAD_ACCUM,
            },
        )
        logger.info("W&B initialized: project=moshi_DPO_1, name=%s", run_name)


    global_eff_batch = BATCH_SIZE * world_size * GRAD_ACCUM
    logger.info(
        "  per_device_batch=%d, world_size=%d, grad_accum=%d "
        "(global_eff_batch=%d, per_device_eff_batch=%d)",
        BATCH_SIZE,
        world_size,
        GRAD_ACCUM,
        global_eff_batch,
        BATCH_SIZE * GRAD_ACCUM,
    )

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
    # GRADIENT CHECKPOINTING
    # # Enable gradient checkpointing on base model FIRST, before LoRA wrapping.
    # model.gradient_checkpointing_enable(
    #     gradient_checkpointing_kwargs={"use_reentrant": False}
    # )
    # model.config.use_cache = False

    # # Disable checkpointing on the depth decoder (causes CUDA assertion).
    # def _disable_checkpointing_recursive(module):
    #     if hasattr(module, "gradient_checkpointing"):
    #         module.gradient_checkpointing = False
    #     for child in module.children():
    #         _disable_checkpointing_recursive(child)

    # _disable_checkpointing_recursive(model.depth_decoder)

    # # Disable checkpointing on Mimi (frozen, no backward, wasteful).
    # _disable_checkpointing_recursive(model.audio_encoder)

    # # Verify.
    # remaining = [n for n, m in model.named_modules()
    #             if getattr(m, "gradient_checkpointing", False)]
    # logger.info("After surgical disable: %d modules still checkpointed", len(remaining))
    # logger.info("First few: %s", remaining[:5])
    # logger.info("PATCH_APPLIED_v1")

    # LoRA - uncomment if ever desired
    # ========================== Apply LoRA ==========================
    # Moshi's temporal transformer wraps attention projections in MoshiLinear,
    # which contains a plain nn.Linear as its .linear child.
    # The depth decoder also uses MoshiLinear, but its inner .linear is a
    # MoshiFlexibleLinear (also unsupported by PEFT), so we must restrict LoRA
    # to the temporal transformer only.
    # We use a regex string (re.fullmatch) anchored to "model.*" — the temporal
    # transformer lives under model.*, while depth_decoder.* and audio_encoder.*
    # are excluded automatically.
    # GC must be set on the base model before get_peft_model(), then
    # enable_input_require_grads() makes GC backprop through frozen base weights.
    # lora_config = LoraConfig(
    #     r=64,
    #     lora_alpha=128,       # scaling = 2 * r
    #     target_modules=r"decoder\.model\..*\.(q|k|v|o)_proj\.linear",  # regex: temporal transformer only
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type=None,       # MoshiForConditionalGeneration is not a PEFT-registered task type
    # )
    # model = get_peft_model(model, lora_config)
    # model.enable_input_require_grads()  # required: lets GC backprop through frozen base weights
    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total = sum(p.numel() for p in model.parameters())
    # logger.info("LoRA applied: %d trainable / %d total (%.2f%%)", trainable, total, trainable / total * 100)

    # Diagnostic: what's on GPU before training even starts?
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        torch.cuda.synchronize()
        logger.info("MEM[after_model_load] alloc=%.1fGB", 
                    torch.cuda.memory_allocated() / 1e9)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        mimi_trainable = sum(p.numel() for p in model.audio_encoder.parameters() if p.requires_grad)
        logger.info("Params: %.2fB total, %.2fB trainable, mimi_trainable=%d",
                    total/1e9, trainable/1e9, mimi_trainable)

    # ========================== 2. Extract Mimi for collator ==========================
    # Mimi freezing and ref model creation happen inside MoshiDPOTrainer.__init__
    mimi = model.audio_encoder
    for param in mimi.parameters():
        param.requires_grad = False


    # trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    # for i, (name, param) in enumerate(trainable):
    #     if i in [274]:
    #         print(
    #             f"DDP param index {i}: {name}, "
    #             f"shape={tuple(param.shape)}, "
    #             f"requires_grad={param.requires_grad}"
    #         )

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

    ## No longer precompute ref logps — compute on the fly inside the trainer's compute_loss
    # # ========== 3.5 Attach precomputed reference logps ==========
    # REF_LOGPS_PATH = os.environ.get("REF_LOGPS_PATH", "/vol/ref_logps.pt")
    # logger.info("Loading precomputed ref logps from %s", REF_LOGPS_PATH)
    # ref_logps = torch.load(REF_LOGPS_PATH)

    # for split_name in list(ds.keys()):
    #     split_ds = ds[split_name]
    #     assert len(split_ds) == len(ref_logps[split_name]["chosen"]), (
    #         f"Precompute/dataset size mismatch for {split_name}: "
    #         f"dataset has {len(split_ds)}, ref file has {len(ref_logps[split_name]['chosen'])}. "
    #         "Did filtering differ between precompute and training?"
    #     )
    #     split_ds = split_ds.add_column("ref_chosen_logp",   ref_logps[split_name]["chosen"])
    #     split_ds = split_ds.add_column("ref_rejected_logp", ref_logps[split_name]["rejected"])
    #     ds[split_name] = split_ds
    #     logger.info("  %s: attached %d ref logp pairs", split_name, len(split_ds))

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

        # fsdp="full_shard auto_wrap",
        # fsdp_config={
        #     "fsdp_transformer_layer_cls_to_wrap": ["MoshiDecoderLayer"],
        #     "fsdp_offload_params": False,
        #     "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        # },

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

        ddp_find_unused_parameters=False, # avoid DDP overhead since we know all params are used

        # Monitoring
        report_to=REPORT_TO,

        # Misc
        seed=13,
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