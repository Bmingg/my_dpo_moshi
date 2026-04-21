"""
DPO trainer for Moshi.

Subclasses `trl.DPOTrainer` and overrides `concatenated_forward` to compute
per-example log-probs under Moshi's dual-stream, 8-codebook layout.

Two modes via `use_text_alignment`:
  - False (Option A, audio-only): completion text stream is all-PAD. Only the
    8 Mimi audio codebooks contribute to the DPO log-prob.
  - True  (Option B, aligned Inner Monologue): text stream is frame-aligned
    via Whisper word-level timestamps. Text CE joins audio CE in the log-prob.

Expected batch keys (produced by SpokenSwagMoshiCollator):
  chosen_moshi_audio_codes   LongTensor [B, K, T_chosen]   pre-encoded Mimi codes
  chosen_user_audio_codes    LongTensor [B, K, T_chosen]   silent user codes (all-zero or mimi-of-silence)
  chosen_input_ids           LongTensor [B, T_chosen]      frame-aligned text stream
  chosen_attention_mask      BoolTensor [B, T_chosen]      1 = real frame, 0 = right-pad
  chosen_completion_mask     BoolTensor [B, T_chosen]      1 = completion frame, 0 = prompt or pad

  (same five for rejected)

Separately the collator should expose the raw waveforms too (under keys like
`chosen_moshi_input_values`) in case you want to encode on the fly instead of
pre-encoding; this trainer prefers the pre-encoded path for speed and clarity.

Reference model:
  Pass `ref_model=None` and `precompute_ref_log_probs=True` in the TRL config to
  avoid holding two copies of the 7B model. Alternatively use LoRA on the policy
  and disable the adapter to compute the reference pass.

Expected output of `concatenated_forward` (TRL contract):
  {
    "chosen_logps":    FloatTensor [B],
    "rejected_logps":  FloatTensor [B],
    "chosen_logits":   placeholder (None or dummy tensor — TRL sometimes reads it),
    "rejected_logits": placeholder,
    "chosen_nll_loss": FloatTensor scalar,   # SFT auxiliary for DPO+SFT variants
  }
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from trl import DPOTrainer

logger = logging.getLogger(__name__)


class SLAMMoshiDPOTrainer(DPOTrainer):
    """DPO over Moshi with dual-stream + 8-codebook log-prob computation."""

    def __init__(
        self,
        *args: Any,
        use_text_alignment: bool = False,
        text_loss_weight: float = 1.0,
        semantic_loss_weight: float = 1.0,
        acoustic_loss_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Args (beyond TRL's DPOTrainer):
            use_text_alignment: True → include text CE in log-prob (Option B).
                False → audio-only DPO (Option A). Must match the collator.
            text_loss_weight: scales the text CE term in the log-prob. The Moshi
                paper weights text=1 and semantic=100 and acoustic=1 in the
                PRETRAINING loss (Eq. 7), but for DPO preference signal this
                asymmetry isn't obviously right. Default 1.0 is neutral.
            semantic_loss_weight: scales codebook 1 (semantic) log-prob. Set to
                100.0 to mirror pretraining weighting if desired.
            acoustic_loss_weight: scales codebooks 2..K (acoustic) log-probs.
        """
        super().__init__(*args, **kwargs)
        self.use_text_alignment = use_text_alignment
        self.text_loss_weight = text_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.acoustic_loss_weight = acoustic_loss_weight

        logger.info(
            "SLAMMoshiDPOTrainer: use_text_alignment=%s, "
            "text_w=%.2f, semantic_w=%.2f, acoustic_w=%.2f",
            use_text_alignment, text_loss_weight,
            semantic_loss_weight, acoustic_loss_weight,
        )

    # ------------------------------------------------------------------
    # Row tokenization — bypassed (the collator does all prep).
    # ------------------------------------------------------------------
    @staticmethod
    def tokenize_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens,
    ):
        """No-op pass-through. All processing happens in SpokenSwagMoshiCollator."""
        return features

    # ------------------------------------------------------------------
    # Core: per-example log-prob of the completion.
    # ------------------------------------------------------------------
    def concatenated_forward(self, model, batch):  # type: ignore[override]
        """Compute log π(completion | prompt) for chosen and rejected sides."""
        chosen_logps = self._side_logps(
            model,
            moshi_codes=batch["chosen_moshi_audio_codes"],
            user_codes=batch["chosen_user_audio_codes"],
            text_ids=batch["chosen_input_ids"],
            attention_mask=batch.get("chosen_attention_mask"),
            completion_mask=batch["chosen_completion_mask"],
        )
        rejected_logps = self._side_logps(
            model,
            moshi_codes=batch["rejected_moshi_audio_codes"],
            user_codes=batch["rejected_user_audio_codes"],
            text_ids=batch["rejected_input_ids"],
            attention_mask=batch.get("rejected_attention_mask"),
            completion_mask=batch["rejected_completion_mask"],
        )

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            # Placeholders — TRL sometimes reads these for logging.
            "chosen_logits": torch.zeros(1, device=chosen_logps.device),
            "rejected_logits": torch.zeros(1, device=rejected_logps.device),
            # SFT auxiliary (negative mean log-prob of chosen). Only used if
            # you enable DPO+SFT via rpo_alpha in the TRL config.
            "chosen_nll_loss": -chosen_logps.mean(),
        }

    # ------------------------------------------------------------------
    # Log-prob for one side (chosen or rejected).
    # ------------------------------------------------------------------
    def _side_logps(
        self,
        model,
        moshi_codes: torch.Tensor,      # [B, K, T]
        user_codes: torch.Tensor,       # [B, K, T]
        text_ids: torch.Tensor,         # [B, T]
        attention_mask: torch.Tensor | None,  # [B, T] bool or None
        completion_mask: torch.Tensor,  # [B, T] bool
    ) -> torch.Tensor:
        """Return per-example log-prob of the completion, shape [B]."""
        B, K, T = moshi_codes.shape
        assert text_ids.shape == (B, T), (
            f"text_ids shape {text_ids.shape} != (B={B}, T={T}). "
            "The collator must align text stream length to audio frame count."
        )

        # ===== 1. Forward pass =====
        # We pass pre-encoded audio codes (moshi_audio_codes / user_audio_codes)
        # to skip the internal Mimi encoding.

        outputs = model(
            input_ids=text_ids,
            attention_mask=attention_mask,
            user_audio_codes=user_codes,
            moshi_audio_codes=moshi_codes,
            # Pass audio_labels to force the depth decoder to run — otherwise
            # audio_logits comes back as None. The actual loss computed with these
            # labels is ignored; we gather our own log-probs from audio_logits.
            audio_labels=moshi_codes,
            return_dict=True,
        )


        # outputs.logits      — text logits,  [B, T, V_text]
        # outputs.audio_logits — audio logits, [B, T, K, V_audio] (assumed)
        text_logits = outputs.logits
        audio_logits = outputs.audio_logits

        # Defensive shape check for audio_logits. HF docstring is ambiguous;
        # we handle the two most plausible layouts.
        audio_logits = self._normalize_audio_logits_shape(
            audio_logits, B=B, T=T, K=K
        )

        # ===== 2. Text stream log-probs (with standard HF shift) =====
        # logits[:, t, :] predicts the token at position t+1.
        text_logp = _gather_shifted_logp(
            logits=text_logits,                         # [B, T, V]
            targets=text_ids,                           # [B, T]
        )                                                # → [B, T-1]

        # ===== 3. Audio stream log-probs (with standard HF shift) =====
        # audio_logits[:, t, k, :] predicts audio code (t+1, k) given context < t+1.
        # Ground-truth codes are `moshi_codes` [B, K, T]; transpose to [B, T, K].
        audio_codes_btk = moshi_codes.transpose(1, 2).contiguous()   # [B, T, K]
        audio_logp = _gather_shifted_audio_logp(
            logits=audio_logits,                        # [B, T, K, V_a]
            targets=audio_codes_btk,                    # [B, T, K]
        )                                                # → [B, T-1, K]

        # ===== 4. Apply per-stream weights =====
        # Codebook 1 is semantic, 2..K are acoustic.
        codebook_weights = torch.tensor(
            [self.semantic_loss_weight] + [self.acoustic_loss_weight] * (K - 1),
            device=audio_logp.device, dtype=audio_logp.dtype,
        )                                                # [K]
        weighted_audio = (audio_logp * codebook_weights).sum(dim=-1)  # [B, T-1]

        if self.use_text_alignment:
            per_frame_logp = self.text_loss_weight * text_logp + weighted_audio
        else:
            per_frame_logp = weighted_audio

        # ===== 5. Mask to completion frames and sum =====
        # Shift completion_mask to align with the shifted logp: logp at index t
        # corresponds to predicting frame t+1, so mask[b, 1:] tells us whether
        # the PREDICTED frame is a completion frame.
        shifted_mask = completion_mask[:, 1:].to(per_frame_logp.dtype)   # [B, T-1]
        logps = (per_frame_logp * shifted_mask).sum(dim=-1)              # [B]

        return logps

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_audio_logits_shape(
        audio_logits: torch.Tensor, *, B: int, T: int, K: int
    ) -> torch.Tensor:
        """Reshape audio_logits to a canonical [B, T, K, V_audio] layout.

        HF's Moshi docstring is ambiguous on the exact shape. Common possibilities:
          (B, T, K, V)      — already canonical
          (B*T, K, V)       — depth decoder flattens frames into batch dim
          (B, K, T, V)      — transposed
        """
        V = audio_logits.shape[-1]
        if audio_logits.shape == (B, T, K, V):
            return audio_logits
        if audio_logits.shape == (B * T, K, V):
            return audio_logits.view(B, T, K, V)
        if audio_logits.shape == (B, K, T, V):
            return audio_logits.transpose(1, 2).contiguous()
        raise ValueError(
            f"Unexpected audio_logits shape {tuple(audio_logits.shape)}; "
            f"expected one of (B={B}, T={T}, K={K}, V), (B*T={B*T}, K={K}, V), "
            f"(B={B}, K={K}, T={T}, V). Inspect the output of a dummy forward."
        )

    # ------------------------------------------------------------------
    # Smoke test — run this once to verify our manual loss matches HF's.
    # ------------------------------------------------------------------
    def verify_manual_loss_matches_hf(
        self, model, batch, tol: float = 1e-3
    ) -> dict:
        """Sanity check: compare our manual log-probs against HF's internal loss.

        Run this on one small batch BEFORE starting real training. If it fails,
        the shift convention or logits shape assumption is wrong, and the
        training gradients will be silently wrong too.

        Procedure:
          1. Compute our manual sum log-prob over ALL non-pad frames for one side.
          2. Run the same forward with text_labels/audio_labels → HF returns a
             (mean) loss over the same positions.
          3. Compare: manual_loss = -manual_logp / num_predicted_tokens should
             ≈ HF's loss, up to a small numerical tolerance.

        Returns a dict with the diagnostic quantities. Raises AssertionError
        if they disagree by more than `tol`.
        """
        moshi_codes = batch["chosen_moshi_audio_codes"]
        user_codes = batch["chosen_user_audio_codes"]
        text_ids = batch["chosen_input_ids"]
        attention_mask = batch.get("chosen_attention_mask")

        B, K, T = moshi_codes.shape

        # ---- HF internal loss path ----
        # Mask out pad positions with -100 so HF excludes them.
        text_labels = text_ids.clone()
        if attention_mask is not None:
            text_labels[~attention_mask.bool()] = -100
        audio_labels = moshi_codes.clone()
        if attention_mask is not None:
            audio_mask = attention_mask.bool().unsqueeze(1).expand(-1, K, -1)
            audio_labels[~audio_mask] = -100

        hf_out = model(
            input_ids=text_ids,
            attention_mask=attention_mask,
            user_audio_codes=user_codes,
            moshi_audio_codes=moshi_codes,
            text_labels=text_labels,
            audio_labels=audio_labels,
            return_dict=True,
        )
        hf_text_loss = hf_out.loss                       # scalar, mean over text positions
        hf_audio_loss = hf_out.depth_loss                # scalar, mean over audio positions

        # ---- Manual path (full-length, no completion masking) ----
        full_mask = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(text_ids, dtype=torch.bool)
        )

        
        # Must pass audio_labels to get audio_logits (depth decoder only runs with labels).
        outputs = model(
            input_ids=text_ids,
            attention_mask=attention_mask,
            user_audio_codes=user_codes,
            moshi_audio_codes=moshi_codes,
            audio_labels=moshi_codes,   # force depth decoder to populate audio_logits
            return_dict=True,
        )
        text_logits = outputs.logits
        audio_logits = self._normalize_audio_logits_shape(
            outputs.audio_logits, B=B, T=T, K=K
        )

        # Text: manual mean CE over non-pad shifted positions.
        text_logp = _gather_shifted_logp(text_logits, text_ids)  # [B, T-1]
        text_shift_mask = full_mask[:, 1:]                        # [B, T-1]
        manual_text_loss = -(text_logp * text_shift_mask).sum() / text_shift_mask.sum().clamp(min=1)

        # Audio: manual mean CE over non-pad shifted positions, averaged over K codebooks.
        audio_codes_btk = moshi_codes.transpose(1, 2).contiguous()
        audio_logp = _gather_shifted_audio_logp(audio_logits, audio_codes_btk)  # [B, T-1, K]
        audio_shift_mask = full_mask[:, 1:].unsqueeze(-1).expand(-1, -1, K)     # [B, T-1, K]
        manual_audio_loss = -(audio_logp * audio_shift_mask).sum() / audio_shift_mask.sum().clamp(min=1)

        diagnostic = {
            "hf_text_loss": float(hf_text_loss),
            "manual_text_loss": float(manual_text_loss),
            "text_diff": float((hf_text_loss - manual_text_loss).abs()),
            "hf_audio_loss": float(hf_audio_loss) if hf_audio_loss is not None else None,
            "manual_audio_loss": float(manual_audio_loss),
            "audio_diff": (
                float((hf_audio_loss - manual_audio_loss).abs())
                if hf_audio_loss is not None else None
            ),
        }
        logger.info("verify_manual_loss_matches_hf: %s", diagnostic)

        assert diagnostic["text_diff"] < tol, (
            f"Text loss mismatch: manual={manual_text_loss} vs hf={hf_text_loss}. "
            f"Shift convention or logits shape is likely wrong."
        )
        if hf_audio_loss is not None:
            assert diagnostic["audio_diff"] < tol, (
                f"Audio loss mismatch: manual={manual_audio_loss} vs hf={hf_audio_loss}. "
                f"Audio logits shape or shift convention is likely wrong."
            )
        return diagnostic


# ======================================================================
# Module-level gather helpers (testable, no state).
# ======================================================================

def _gather_shifted_logp(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Per-position log-prob under standard causal-LM shift.

    logits[:, t, :] is assumed to predict the token at position t+1.
    Returns log p(targets[:, t+1] | context_{<=t}), i.e. shape [B, T-1].

    Args:
        logits: [B, T, V]
        targets: [B, T]

    Returns:
        [B, T-1] tensor of per-position log-probs.
    """
    shifted_logits = logits[:, :-1, :]                           # [B, T-1, V]
    shifted_targets = targets[:, 1:]                             # [B, T-1]
    logp_all = F.log_softmax(shifted_logits, dim=-1)             # [B, T-1, V]
    logp = logp_all.gather(dim=-1, index=shifted_targets.unsqueeze(-1)).squeeze(-1)
    return logp                                                  # [B, T-1]


def _gather_shifted_audio_logp(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Audio-stream version of _gather_shifted_logp.

    logits: [B, T, K, V_a]   — logits[:, t, k, :] predicts code (t+1, k)
    targets: [B, T, K]        — ground-truth codes
    Returns: [B, T-1, K]      — per-frame per-codebook log-probs

    The same frame-level shift applies to every codebook: the temporal
    transformer's hidden state at frame t is conditioned on frames 0..t-1 and
    feeds into the depth decoder's prediction of every codebook at frame t+1.
    """
    shifted_logits = logits[:, :-1, :, :]                        # [B, T-1, K, V_a]
    shifted_targets = targets[:, 1:, :]                          # [B, T-1, K]
    logp_all = F.log_softmax(shifted_logits, dim=-1)             # [B, T-1, K, V_a]
    logp = logp_all.gather(dim=-1, index=shifted_targets.unsqueeze(-1)).squeeze(-1)
    return logp                                                  # [B, T-1, K]