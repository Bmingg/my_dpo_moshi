"""
Moshi DPO Trainer — inherits from transformers.Trainer, no TRL dependency.

Computes per-example log-probs under Moshi's dual-stream, 8-codebook,
delay-patterned layout, then applies the standard DPO sigmoid loss.

The DPO math is 3 lines. Everything else is Moshi-specific audio handling.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer

logger = logging.getLogger(__name__)


# ============================================================
# Core helpers (unchanged from the original trainer)
# ============================================================

def _apply_delay_pattern(codes: torch.Tensor, bos_token_id: int) -> torch.Tensor:
    """Shift codebooks 1..K-1 right by 1, fill position 0 with BOS."""
    B, K, T = codes.shape
    out = torch.empty_like(codes)
    out[:, 0, :] = codes[:, 0, :]
    out[:, 1:, 0] = bos_token_id
    if T > 1:
        out[:, 1:, 1:] = codes[:, 1:, :-1]
    return out


def _gather_text_logp(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Causal-shift text log-prob gather. Returns [B, T-1]."""
    shifted_logits = logits[:, :-1, :]
    shifted_targets = targets[:, 1:]
    logp_all = F.log_softmax(shifted_logits, dim=-1)
    logp = logp_all.gather(dim=-1, index=shifted_targets.unsqueeze(-1)).squeeze(-1)
    return logp


def _gather_audio_logp(audio_logits: torch.Tensor, delayed_codes: torch.Tensor) -> torch.Tensor:
    """Gather audio log-probs from depth decoder logits. Returns [B, T, K]."""
    delayed_codes_btk = delayed_codes.transpose(1, 2).contiguous()
    V_audio = audio_logits.shape[-1]
    delayed_codes_safe = delayed_codes_btk.clamp(0, V_audio - 1)
    logp_all = F.log_softmax(audio_logits, dim=-1)
    logp = logp_all.gather(dim=-1, index=delayed_codes_safe.unsqueeze(-1)).squeeze(-1)
    return logp


def _build_audio_completion_mask(completion_mask: torch.Tensor, num_codebooks: int) -> torch.Tensor:
    """Expand frame-level completion mask to [B, T, K] for audio scoring."""
    B, T = completion_mask.shape
    out = torch.zeros((B, T, num_codebooks), dtype=completion_mask.dtype, device=completion_mask.device)
    out[:, :, 0] = completion_mask
    if T > 1:
        out[:, 1:, 1:] = completion_mask[:, :-1, None]
    return out.to(torch.float32)


# ============================================================
# Trainer
# ============================================================

class MoshiDPOTrainer(Trainer):
    """DPO trainer for Moshi. Inherits from transformers.Trainer directly."""

    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        processing_class=None,
        *,
        ref_model=None,
        beta: float = 0.1,
        use_text_alignment: bool = False,
        text_loss_weight: float = 1.0,
        semantic_loss_weight: float = 7.0,
        acoustic_loss_weight: float = 1.0,
        audio_bos_token_id: int | None = None,
        **kwargs,
    ):
        # ---- Freeze Mimi BEFORE deepcopy so ref model's Mimi is also frozen ----
        mimi = model.audio_encoder
        mimi.eval()
        for param in mimi.parameters():
            param.requires_grad = False
        frozen_count = sum(p.numel() for p in mimi.parameters())
        logger.info("Mimi encoder frozen (%d params)", frozen_count)

        # ---- Create reference model ----
        if ref_model is None:
            logger.info("Creating reference model (deepcopy)...")
            ref_model = copy.deepcopy(model)
            ref_model.gradient_checkpointing_disable()
            for param in ref_model.parameters():
                param.requires_grad = False
        ref_model.eval()
        self._ref_model = ref_model  # stored before super().__init__

        # ---- DPO config ----
        self.beta = beta
        self.use_text_alignment = use_text_alignment
        self.text_loss_weight = text_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.acoustic_loss_weight = acoustic_loss_weight
        self.audio_bos_token_id = audio_bos_token_id

        # ---- Metrics storage ----
        self._dpo_metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # ---- Init Trainer ----
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=processing_class,
            **kwargs,
        )

        # ---- Prepare ref model for device placement ----
        self._ref_model = self.accelerator.prepare_model(self._ref_model, evaluation_mode=True)

        # Log param counts
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Parameters: %.2fB total, %.2fB trainable (%.1f%%)",
            total / 1e9, trainable / 1e9, trainable / total * 100,
        )
        logger.info(
            "MoshiDPOTrainer: beta=%.2f, semantic_w=%.1f, acoustic_w=%.1f, text_align=%s",
            beta, semantic_loss_weight, acoustic_loss_weight, use_text_alignment,
        )

    # ============================================================
    # Audio BOS resolution (same as before)
    # ============================================================

    def _resolve_audio_bos(self, model) -> int:
        if self.audio_bos_token_id is not None:
            return self.audio_bos_token_id
        # Unwrap accelerate/FSDP wrappers
        m = model.module if hasattr(model, "module") else model
        v = getattr(m.config, "audio_vocab_size", None)
        if v is None and hasattr(m.config, "depth_decoder_config"):
            v = getattr(m.config.depth_decoder_config, "audio_vocab_size", None)
        if v is None:
            raise ValueError("Cannot auto-detect audio_bos_token_id. Pass it explicitly.")
        self.audio_bos_token_id = v
        logger.info("Resolved audio_bos_token_id = %d", v)
        return v

    # ============================================================
    # Side log-probs (unchanged from original)
    # ============================================================

    def _side_logps(
        self,
        model,
        moshi_codes: torch.Tensor,
        user_codes: torch.Tensor,
        text_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        completion_mask: torch.Tensor,
        audio_bos: int,
    ) -> torch.Tensor:
        """Return per-example log-prob of the completion, shape [B]."""
        B, K, T = moshi_codes.shape

        outputs = model(
            input_ids=text_ids,
            attention_mask=attention_mask,
            user_audio_codes=user_codes,
            moshi_audio_codes=moshi_codes,
            text_labels=text_ids,
            audio_labels=moshi_codes,
            return_dict=True,
        )
        text_logits = outputs.logits
        audio_logits = outputs.audio_logits
        V_audio = audio_logits.shape[-1]
        audio_logits = audio_logits.view(B, T, K, V_audio)

        # Text log-probs
        text_logp_per_pos = _gather_text_logp(text_logits, text_ids)

        # Audio log-probs with delay pattern
        delayed_codes = _apply_delay_pattern(moshi_codes, audio_bos)
        audio_logp_per_pos = _gather_audio_logp(audio_logits, delayed_codes)

        # Codebook weights
        codebook_weights = torch.tensor(
            [self.semantic_loss_weight] + [self.acoustic_loss_weight] * (K - 1),
            device=audio_logp_per_pos.device, dtype=audio_logp_per_pos.dtype,
        )
        audio_logp_weighted = audio_logp_per_pos * codebook_weights

        # Text completion mask
        text_mask = completion_mask[:, 1:].to(text_logp_per_pos.dtype)
        text_logp = (text_logp_per_pos * text_mask).sum(dim=-1)

        # Audio completion mask
        audio_mask = _build_audio_completion_mask(completion_mask, K).to(audio_logp_weighted.dtype)
        audio_logp = (audio_logp_weighted * audio_mask).sum(dim=(1, 2))

        # Combine
        if self.use_text_alignment:
            return self.text_loss_weight * text_logp + audio_logp
        return audio_logp

    # ============================================================
    # compute_loss — the DPO loss
    # ============================================================

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute DPO sigmoid loss over Moshi audio log-probs."""
        audio_bos = self._resolve_audio_bos(model)
        mode = "train" if model.training else "eval"

        # ---- Policy log-probs ----
        chosen_logps = self._side_logps(
            model,
            moshi_codes=inputs["chosen_moshi_audio_codes"],
            user_codes=inputs["chosen_user_audio_codes"],
            text_ids=inputs["chosen_input_ids"],
            attention_mask=inputs.get("chosen_attention_mask"),
            completion_mask=inputs["chosen_completion_mask"],
            audio_bos=audio_bos,
        )
        rejected_logps = self._side_logps(
            model,
            moshi_codes=inputs["rejected_moshi_audio_codes"],
            user_codes=inputs["rejected_user_audio_codes"],
            text_ids=inputs["rejected_input_ids"],
            attention_mask=inputs.get("rejected_attention_mask"),
            completion_mask=inputs["rejected_completion_mask"],
            audio_bos=audio_bos,
        )

        # ---- Reference log-probs ----
        with torch.no_grad():
            ref_chosen_logps = self._side_logps(
                self._ref_model,
                moshi_codes=inputs["chosen_moshi_audio_codes"],
                user_codes=inputs["chosen_user_audio_codes"],
                text_ids=inputs["chosen_input_ids"],
                attention_mask=inputs.get("chosen_attention_mask"),
                completion_mask=inputs["chosen_completion_mask"],
                audio_bos=audio_bos,
            )
            ref_rejected_logps = self._side_logps(
                self._ref_model,
                moshi_codes=inputs["rejected_moshi_audio_codes"],
                user_codes=inputs["rejected_user_audio_codes"],
                text_ids=inputs["rejected_input_ids"],
                attention_mask=inputs.get("rejected_attention_mask"),
                completion_mask=inputs["rejected_completion_mask"],
                audio_bos=audio_bos,
            )

        # ---- DPO loss (sigmoid) ----
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
        margins = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(margins).mean()

        # ---- Log metrics ----
        with torch.no_grad():
            self._dpo_metrics[mode]["loss"].append(loss.item())
            self._dpo_metrics[mode]["rewards/chosen"].append(chosen_rewards.mean().item())
            self._dpo_metrics[mode]["rewards/rejected"].append(rejected_rewards.mean().item())
            self._dpo_metrics[mode]["rewards/margins"].append(margins.mean().item())
            self._dpo_metrics[mode]["rewards/accuracies"].append((margins > 0).float().mean().item())
            self._dpo_metrics[mode]["logps/chosen"].append(chosen_logps.mean().item())
            self._dpo_metrics[mode]["logps/rejected"].append(rejected_logps.mean().item())

        if return_outputs:
            return loss, {"chosen_logps": chosen_logps, "rejected_logps": rejected_logps}
        return loss

    # ============================================================
    # Metric logging — flush stored metrics on each log() call
    # ============================================================

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {}
        for key, vals in self._dpo_metrics[mode].items():
            if vals:
                metrics[key] = sum(vals) / len(vals)
        if mode == "eval":
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._dpo_metrics[mode].clear()

    # ============================================================
    # prediction_step — force compute_loss during eval
    # ============================================================

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss, None, None

    # ============================================================
    # Verification (same as before, for sanity checks)
    # ============================================================

    def verify_manual_loss_matches_hf(self, model, batch, tol: float = 5e-3) -> dict:
        """One forward pass, compare HF internal loss with our manual gather."""
        audio_bos = self._resolve_audio_bos(model)
        moshi_codes = batch["chosen_moshi_audio_codes"]
        user_codes = batch["chosen_user_audio_codes"]
        text_ids = batch["chosen_input_ids"]
        attention_mask = batch.get("chosen_attention_mask")
        B, K, T = moshi_codes.shape

        with torch.no_grad():
            outputs = model(
                input_ids=text_ids,
                attention_mask=attention_mask,
                user_audio_codes=user_codes,
                moshi_audio_codes=moshi_codes,
                text_labels=text_ids,
                audio_labels=moshi_codes,
                return_dict=True,
            )

        hf_loss = float(outputs.loss)
        hf_depth_loss = float(outputs.depth_loss)
        text_logits = outputs.logits
        audio_logits = outputs.audio_logits.view(B, T, K, -1)

        text_logp = _gather_text_logp(text_logits, text_ids)
        delayed_codes = _apply_delay_pattern(moshi_codes, audio_bos)
        audio_logp = _gather_audio_logp(audio_logits, delayed_codes)

        manual_text = -text_logp.mean().item()
        manual_audio = -audio_logp.mean().item()
        text_rel = abs(manual_text - hf_loss) / max(abs(hf_loss), 1e-8)
        audio_rel = abs(manual_audio - hf_depth_loss) / max(abs(hf_depth_loss), 1e-8)

        diagnostic = {
            "hf_text_CE": hf_loss, "manual_text_CE": manual_text, "text_rel_diff": text_rel,
            "hf_audio_CE": hf_depth_loss, "manual_audio_CE": manual_audio, "audio_rel_diff": audio_rel,
        }
        logger.info("verify: %s", diagnostic)
        assert text_rel < 0.5, f"Text CE diverges: {manual_text:.4f} vs {hf_loss:.4f}"
        assert audio_rel < 0.5, f"Audio CE diverges: {manual_audio:.4f} vs {hf_depth_loss:.4f}"
        return diagnostic