"""
DPO trainer for Moshi with BATCHED log-prob computation.

Overrides `trl.DPOTrainer.concatenated_forward` to compute per-example log-probs
under Moshi's dual-stream, 8-codebook, delay-patterned layout.

Two modes controlled by `use_text_alignment`:
  - False (Option A, audio-only): completion text stream is all-PAD. Only the
    8 Mimi audio codebooks contribute to the DPO log-prob.
  - True  (Option B, aligned Inner Monologue): completion text stream is
    frame-aligned via Whisper timestamps. Text CE joins audio CE in the log-prob.

Design notes:
  - The depth decoder only populates `outputs.audio_logits` when BOTH
    `text_labels` and `audio_labels` are passed. We pass dummy labels (the
    ground-truth codes themselves) to trigger it, then ignore the returned
    scalar loss and gather from the raw logits ourselves.
  - We replicate HF's `build_delay_pattern_mask` for audio labels, because
    the gather has to target the delayed positions. See `_apply_delay_pattern`.
  - Per-example log-probs fall out from batched gather → batched mask → sum.
    No per-example loop needed.

Expected batch keys (produced by SpokenSwagMoshiCollator):
  {side}_moshi_audio_codes     LongTensor [B, K, T]   pre-encoded Mimi codes
  {side}_user_audio_codes      LongTensor [B, K, T]   user-side Mimi codes (silent)
  {side}_input_ids             LongTensor [B, T]      frame-aligned text stream
  {side}_attention_mask        BoolTensor [B, T]      True for prompt+completion, False for right-pad
  {side}_completion_mask       BoolTensor [B, T]      True for completion frames only

where side is either "chosen" or "rejected", T is the per-side padded frame count.

Reference model:
  Pass `ref_model=None` and `precompute_ref_log_probs=True` in the TRL config
  to avoid holding two copies of the 7B model.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from trl import DPOTrainer

logger = logging.getLogger(__name__)


# ======================================================================
# Core module-level helpers (testable without the full trainer).
# ======================================================================

def _apply_delay_pattern(
    codes: torch.Tensor, bos_token_id: int
) -> torch.Tensor:
    """Replicate HF's `build_delay_pattern_mask` for Moshi's τ=1 acoustic delay.

    Given Mimi codes in "actual-time" layout [B, K, T] where codes[:, q, t] is
    the code for codebook q at actual frame t, return a delayed version where:
      - Codebook 0 (semantic): unchanged.
      - Codebooks 1..K-1 (acoustic): shifted right by 1, with BOS at position 0.

    So the returned tensor `out` has:
      out[:, 0, t] = codes[:, 0, t]                  for all t
      out[:, q, 0] = bos_token_id                    for q > 0
      out[:, q, t] = codes[:, q, t-1]                for q > 0, t >= 1

    This exactly matches the paper's V_{s,1+q} = A_{s-τ, q} with τ=1, and
    matches what HF's depth decoder is predicting at each (depth-step, codebook)
    position when we pass audio_labels=codes.

    NOTE: HF's version also overwrites the last position of codebook 0 with PAD
    and extends max_length by 1. For our DPO scoring we stay within the input's
    time dimension T, which is fine as long as we handle the boundary in the
    completion mask.

    Args:
        codes: [B, K, T] long tensor of actual-time audio codes.
        bos_token_id: BOS value placed at position 0 of shifted codebooks. For
            Moshi this is typically audio_vocab_size (a sentinel value outside
            the normal code range).

    Returns:
        [B, K, T] long tensor of delayed codes.
    """
    B, K, T = codes.shape
    out = torch.empty_like(codes)
    out[:, 0, :] = codes[:, 0, :]          # codebook 0 unchanged
    out[:, 1:, 0] = bos_token_id           # BOS at position 0 for q > 0
    if T > 1:
        out[:, 1:, 1:] = codes[:, 1:, :-1]  # right-shift by 1 for q > 0
    return out


def _gather_text_logp(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Causal-shift text log-prob gather.

    logits[:, t, :] predicts the text token at position t+1 given context ≤ t.
    So log p(targets[:, t+1] | context_{<=t}) = log_softmax(logits[:, t, :])[targets[:, t+1]].

    Args:
        logits:  [B, T, V_text]
        targets: [B, T]

    Returns:
        [B, T-1] tensor of per-position log-probs. Position i in the output
        corresponds to predicting targets[:, i+1].
    """
    shifted_logits  = logits[:, :-1, :]                # [B, T-1, V]
    shifted_targets = targets[:, 1:]                   # [B, T-1]
    logp_all = F.log_softmax(shifted_logits, dim=-1)   # [B, T-1, V]
    logp = logp_all.gather(
        dim=-1, index=shifted_targets.unsqueeze(-1)
    ).squeeze(-1)                                      # [B, T-1]
    return logp


def _gather_audio_logp(
    audio_logits: torch.Tensor,
    delayed_codes: torch.Tensor,
) -> torch.Tensor:
    """Gather audio log-probs from the depth decoder's logits.

    The depth decoder at (model-step s, codebook k) predicts the delayed label
    at that position. So the target for position (s, k) is delayed_codes[:, k, s],
    i.e., the value that emerged from `_apply_delay_pattern`.

    Args:
        audio_logits:  [B, T, K, V_audio]  — already reshaped from HF's
            [B*T, K, V_audio] output. audio_logits[b, s, k, :] is the depth
            decoder's prediction at model-step s, codebook k for example b.
        delayed_codes: [B, K, T]  — output of `_apply_delay_pattern` on the
            ground-truth codes.

    Returns:
        [B, T, K] tensor of per-(step, codebook) log-probs.
    """
    # Transpose delayed_codes to [B, T, K] to match audio_logits layout.
    # BOS sentinel positions contain id = audio_vocab_size (e.g., 2048), which
    # is out-of-range for .gather into audio_logits (vocab size 2048 → valid
    # indices 0..2047). Clamp before gathering; those positions are masked out
    # downstream in _build_audio_completion_mask anyway.
    delayed_codes_btk = delayed_codes.transpose(1, 2).contiguous()
    V_audio = audio_logits.shape[-1]
    delayed_codes_safe = delayed_codes_btk.clamp(0, V_audio - 1)

    logp_all = F.log_softmax(audio_logits, dim=-1)
    logp = logp_all.gather(
        dim=-1, index=delayed_codes_safe.unsqueeze(-1)
    ).squeeze(-1)
    return logp


def _build_audio_completion_mask(
    completion_mask: torch.Tensor, num_codebooks: int
) -> torch.Tensor:
    """Expand the frame-level completion mask to an audio-position mask.

    For codebook 0 (semantic, no delay): the model at step s predicts codebook 0
    for actual-frame s, so the mask at (s, 0) equals completion_mask[b, s].

    For codebooks q > 0 (acoustic, delayed by 1): the model at step s predicts
    the code for actual-frame s-1. So we want to score only if s-1 is in the
    completion range — equivalently, shift completion_mask right by 1.

    The position (s=0, q>0) carries the BOS token and should never be scored
    (it's a placeholder, not a real code). The right-shift naturally excludes it.

    Args:
        completion_mask: [B, T] bool. True for completion frames.
        num_codebooks:   K (8 for Moshi).

    Returns:
        [B, T, K] float tensor (not bool, to allow weighted sums). 1.0 at
        positions that should contribute to the log-prob, 0.0 otherwise.
    """
    B, T = completion_mask.shape
    out = torch.zeros((B, T, num_codebooks), dtype=completion_mask.dtype,
                      device=completion_mask.device)
    # Codebook 0: no shift.
    out[:, :, 0] = completion_mask
    # Codebooks 1..K-1: right-shift by 1.
    if T > 1:
        out[:, 1:, 1:] = completion_mask[:, :-1, None]
    return out.to(torch.float32)  # float for use in multiplication


# ======================================================================
# Trainer.
# ======================================================================

class SLAMMoshiDPOTrainer(DPOTrainer):
    """DPO over Moshi with batched log-prob computation."""

    def __init__(
        self,
        *args: Any,
        use_text_alignment: bool = False,
        text_loss_weight: float = 1.0,
        semantic_loss_weight: float = 1.0,
        acoustic_loss_weight: float = 1.0,
        audio_bos_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args (beyond TRL's DPOTrainer):
            use_text_alignment: True → include text CE in the log-prob (Option B).
                False → audio-only DPO (Option A). Must match the collator.
            text_loss_weight: scales the text contribution. 1.0 is neutral.
            semantic_loss_weight: scales codebook 1 (semantic) log-prob.
                Moshi's pretraining uses 100; we default to 1.0 (equal weight).
            acoustic_loss_weight: scales codebooks 2..K log-probs.
            audio_bos_token_id: value placed at position 0 of delayed codebooks.
                For Moshi this is typically audio_vocab_size (e.g., 2048), used
                as a sentinel outside the normal code range. If None, we'll
                detect it from the model config at first forward.
        """
        super().__init__(*args, **kwargs)
        self.use_text_alignment = use_text_alignment
        self.text_loss_weight = text_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.acoustic_loss_weight = acoustic_loss_weight
        self.audio_bos_token_id = audio_bos_token_id  # may be None; resolved lazily

        logger.info(
            "SLAMMoshiDPOTrainer: use_text_alignment=%s, text_w=%.2f, "
            "semantic_w=%.2f, acoustic_w=%.2f",
            use_text_alignment, text_loss_weight,
            semantic_loss_weight, acoustic_loss_weight,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length,
                     max_completion_length, add_special_tokens):
        """Bypass TRL's per-row tokenisation; the collator handles all prep."""
        return features

    # ------------------------------------------------------------------
    def _resolve_audio_bos(self, model) -> int:
        """Return the BOS token id used for delay-pattern shifting."""
        if self.audio_bos_token_id is not None:
            return self.audio_bos_token_id
        # Kyutai convention: use audio_vocab_size as the BOS sentinel.
        # (Same as what HF's get_unconditional_inputs does.)
        v = getattr(model.config, "audio_vocab_size", None)
        if v is None and hasattr(model.config, "depth_decoder_config"):
            v = getattr(model.config.depth_decoder_config, "audio_vocab_size", None)
        if v is None:
            raise ValueError(
                "Cannot auto-detect audio_bos_token_id. Pass it explicitly to "
                "SLAMMoshiDPOTrainer."
            )
        self.audio_bos_token_id = v
        logger.info("Resolved audio_bos_token_id = %d", v)
        return v

    # ------------------------------------------------------------------
    def concatenated_forward(self, model, batch):  # type: ignore[override]
        """Return per-example log π(completion | prompt) for chosen and rejected.

        Returns a dict matching TRL's contract:
          {
            "chosen_logps":    FloatTensor [B],
            "rejected_logps":  FloatTensor [B],
            "chosen_logits":   placeholder (None-equivalent),
            "rejected_logits": placeholder,
            "chosen_nll_loss": FloatTensor scalar (SFT auxiliary; TRL's DPO
                               uses it only if rpo_alpha is set),
          }
        """
        audio_bos = self._resolve_audio_bos(model)

        chosen_logps = self._side_logps(
            model,
            moshi_codes=batch["chosen_moshi_audio_codes"],
            user_codes=batch["chosen_user_audio_codes"],
            text_ids=batch["chosen_input_ids"],
            attention_mask=batch.get("chosen_attention_mask"),
            completion_mask=batch["chosen_completion_mask"],
            audio_bos=audio_bos,
        )
        rejected_logps = self._side_logps(
            model,
            moshi_codes=batch["rejected_moshi_audio_codes"],
            user_codes=batch["rejected_user_audio_codes"],
            text_ids=batch["rejected_input_ids"],
            attention_mask=batch.get("rejected_attention_mask"),
            completion_mask=batch["rejected_completion_mask"],
            audio_bos=audio_bos,
        )

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            # Placeholders — TRL sometimes reads these for logging.
            "chosen_logits": torch.zeros(1, device=chosen_logps.device),
            "rejected_logits": torch.zeros(1, device=rejected_logps.device),
            # SFT auxiliary loss for DPO+SFT variants (rpo_alpha).
            "chosen_nll_loss": -chosen_logps.mean(),
        }

    # ------------------------------------------------------------------
    def _side_logps(
        self,
        model,
        moshi_codes: torch.Tensor,      # [B, K, T] long
        user_codes: torch.Tensor,       # [B, K, T] long
        text_ids: torch.Tensor,         # [B, T]    long
        attention_mask: torch.Tensor | None,   # [B, T] bool or None
        completion_mask: torch.Tensor,  # [B, T]    bool
        audio_bos: int,
    ) -> torch.Tensor:
        """Return per-example log-prob of the completion, shape [B].

        Steps:
          1. One forward pass on (prompt ⊕ completion) with dummy labels to
             trigger the depth decoder and populate audio_logits.
          2. Text gather: standard causal shift + log_softmax + gather → [B, T-1].
          3. Audio gather: apply delay pattern to ground-truth codes, then
             log_softmax + gather against delayed positions → [B, T, K].
          4. Apply completion masks (frame-level for text, delay-shifted for audio).
          5. Sum and combine.
        """
        B, K, T = moshi_codes.shape
        assert text_ids.shape == (B, T), (
            f"text_ids shape {text_ids.shape} != (B={B}, T={T}). "
            "Text stream length must match audio frame count."
        )

        # ===== 1. Forward pass =====
        # We pass dummy labels (the codes themselves) to trigger the depth
        # decoder. Without labels, audio_logits comes back as None.
        outputs = model(
            input_ids=text_ids,
            attention_mask=attention_mask,
            user_audio_codes=user_codes,
            moshi_audio_codes=moshi_codes,
            text_labels=text_ids,
            audio_labels=moshi_codes,
            return_dict=True,
        )
        text_logits  = outputs.logits          # [B, T, V_text]
        audio_logits = outputs.audio_logits    # [B*T, K, V_audio]

        # Reshape audio_logits to [B, T, K, V_audio].
        V_audio = audio_logits.shape[-1]
        audio_logits = audio_logits.view(B, T, K, V_audio)

        # ===== 2. Text log-probs (standard causal shift) =====
        text_logp_per_pos = _gather_text_logp(text_logits, text_ids)  # [B, T-1]

        # ===== 3. Audio log-probs with delay pattern =====
        delayed_codes = _apply_delay_pattern(moshi_codes, audio_bos)  # [B, K, T]
        audio_logp_per_pos = _gather_audio_logp(
            audio_logits, delayed_codes
        )                                                              # [B, T, K]

        # ===== 4. Apply codebook weights =====
        codebook_weights = torch.tensor(
            [self.semantic_loss_weight] + [self.acoustic_loss_weight] * (K - 1),
            device=audio_logp_per_pos.device,
            dtype=audio_logp_per_pos.dtype,
        )                                                              # [K]
        audio_logp_weighted = audio_logp_per_pos * codebook_weights    # [B, T, K]

        # ===== 5. Apply completion masks =====
        # Text mask: frame t of text_logp_per_pos scores text at frame t+1.
        # So we mask with completion_mask[:, 1:] to only score completion text.
        text_mask = completion_mask[:, 1:].to(text_logp_per_pos.dtype)  # [B, T-1]
        text_logp = (text_logp_per_pos * text_mask).sum(dim=-1)         # [B]

        # Audio mask: (model-step s, codebook k) scores:
        #   - codebook 0: frame s
        #   - codebook q > 0: frame s-1 (right-shifted)
        audio_mask = _build_audio_completion_mask(
            completion_mask, K
        ).to(audio_logp_weighted.dtype)                                # [B, T, K]
        audio_logp = (audio_logp_weighted * audio_mask).sum(dim=(1, 2))  # [B]

        # ===== 6. Combine =====
        if self.use_text_alignment:
            logps = self.text_loss_weight * text_logp + audio_logp
        else:
            logps = audio_logp
        return logps

    # ------------------------------------------------------------------
    # Verification: our batched manual gather vs HF's internal loss.
    # ------------------------------------------------------------------
    def verify_manual_loss_matches_hf(
        self, model, batch, tol: float = 5e-3
    ) -> dict:
        """One forward pass, extract both HF's loss and our manual gather."""
        audio_bos = self._resolve_audio_bos(model)
    
        moshi_codes    = batch["chosen_moshi_audio_codes"]
        user_codes     = batch["chosen_user_audio_codes"]
        text_ids       = batch["chosen_input_ids"]
        attention_mask = batch.get("chosen_attention_mask")
    
        B, K, T = moshi_codes.shape
    
        # One forward — gives us both HF's internal losses AND the raw logits
        # we need for manual gather.
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
    
        hf_loss       = float(outputs.loss)
        hf_depth_loss = float(outputs.depth_loss)
    
        text_logits  = outputs.logits
        audio_logits = outputs.audio_logits.view(B, T, K, -1)
    
        text_logp_per_pos  = _gather_text_logp(text_logits, text_ids)
        delayed_codes      = _apply_delay_pattern(moshi_codes, audio_bos)
        audio_logp_per_pos = _gather_audio_logp(audio_logits, delayed_codes)
    
        manual_text_mean  = -text_logp_per_pos.mean().item()
        manual_audio_mean = -audio_logp_per_pos.mean().item()
    
        text_diff  = abs(manual_text_mean  - hf_loss)
        audio_diff = abs(manual_audio_mean - hf_depth_loss)
        text_rel   = text_diff  / max(abs(hf_loss),       1e-8)
        audio_rel  = audio_diff / max(abs(hf_depth_loss), 1e-8)
    
        diagnostic = {
            "hf_text_mean_CE":      hf_loss,
            "manual_text_mean_CE":  manual_text_mean,
            "text_diff":            text_diff,
            "text_rel_diff":        text_rel,
            "hf_audio_mean_CE":     hf_depth_loss,
            "manual_audio_mean_CE": manual_audio_mean,
            "audio_diff":           audio_diff,
            "audio_rel_diff":       audio_rel,
        }
        logger.info("verify: %s", diagnostic)
    
        assert text_rel < 0.5, (
            f"Text CE diverges: manual={manual_text_mean:.4f}, hf={hf_loss:.4f}. "
            f"Check _gather_text_logp shift."
        )
        assert audio_rel < 0.5, (
            f"Audio CE diverges: manual={manual_audio_mean:.4f}, hf={hf_depth_loss:.4f}. "
            f"Check _apply_delay_pattern or _gather_audio_logp."
        )
        return diagnostic