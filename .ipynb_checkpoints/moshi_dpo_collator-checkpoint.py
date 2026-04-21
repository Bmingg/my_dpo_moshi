"""
Data collator + dataset loader for Moshi DPO training on SpokenSwag.

Pairs with `moshi_dpo_trainer.py` (SLAMMoshiDPOTrainer).

Two modes via `use_text_alignment` (must match the trainer's flag):
  - False: completion text stream is all-PAD (Option A, audio-only DPO).
  - True:  completion text stream is frame-aligned from word-level timestamps
           (Option B, full Inner Monologue DPO).

The collator:
  1. Extracts and truncates raw audio for prompt / chosen / rejected.
  2. Pads each clip to an exact multiple of SAMPLES_PER_FRAME so frame counts
     are unambiguous.
  3. Concatenates (prompt ⊕ completion) per example, because DPO scores the
     completion CONDITIONAL ON the prompt.
  4. Encodes audio to Mimi codes via `mimi_encoder` (pre-provided) — or leaves
     the raw waveforms in the batch if `mimi_encoder` is None.
  5. Builds a frame-aligned text stream (PAD-only for Option A, per-paper Eq. 5
     for Option B).
  6. Returns tensors keyed by stream + a completion mask.

Output keys (for each of {chosen, rejected}):
  {side}_moshi_audio_codes     LongTensor [B, K, T]   pre-encoded Mimi codes
  {side}_user_audio_codes      LongTensor [B, K, T]   silent-user codes
  {side}_input_ids             LongTensor [B, T]      frame-aligned text stream
  {side}_attention_mask        BoolTensor [B, T]      1 = real frame
  {side}_completion_mask       BoolTensor [B, T]      1 = completion frame
  (if mimi_encoder is None, also: {side}_moshi_input_values / {side}_user_input_values)

For Option B, each dataset example must contain:
  prompt_alignment, chosen_alignment, rejected_alignment:
    list of {"word": str, "start": float, "end": float}  (times in seconds)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from datasets import Audio, DatasetDict, load_dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ======================================================================
# Constants
# ======================================================================

SAMPLING_RATE = 24_000
FRAME_RATE = 12.5
SAMPLES_PER_FRAME = int(SAMPLING_RATE / FRAME_RATE)   # 1920
NUM_CODEBOOKS = 8                                     # Mimi: 1 semantic + 7 acoustic

SPOKEN_SWAG_REPO = "slprl/SpokenSwag"


# ======================================================================
# Dataset loading (unchanged from scaffold apart from docstring)
# ======================================================================

def load_spoken_swag(cfg: DictConfig, *, sampling_rate: int = SAMPLING_RATE) -> DatasetDict:
    """Load SpokenSwag and resample audio columns to 24 kHz (Moshi's rate)."""
    repo_id = cfg.get("repo_id", SPOKEN_SWAG_REPO)
    cache_dir = cfg.get("cache_dir", None)
    num_proc = cfg.get("num_proc", 4)

    logger.info("Loading %s (cache_dir=%s)", repo_id, cache_dir)
    ds: DatasetDict = load_dataset(repo_id, cache_dir=cache_dir)

    audio_feat = Audio(sampling_rate=sampling_rate)
    for col in ("prompt", "chosen", "rejected"):
        ds = ds.cast_column(col, audio_feat)

    if cfg.get("repetition_filter", False):
        max_auto_bleu = float(cfg.get("max_auto_bleu", 0.3))

        def keep(x):
            return x["auto_bleu2"] is None or x["auto_bleu2"] < max_auto_bleu

        before = {k: len(v) for k, v in ds.items()}
        ds = ds.filter(keep, num_proc=num_proc)
        after = {k: len(v) for k, v in ds.items()}
        logger.info("repetition_filter %s -> %s", before, after)

    train_take = cfg.get("train_take", None)
    val_take = cfg.get("val_take", None)
    if train_take is not None and "train" in ds:
        ds["train"] = ds["train"].select(range(min(train_take, len(ds["train"]))))
    if val_take is not None and "validation" in ds:
        ds["validation"] = ds["validation"].select(
            range(min(val_take, len(ds["validation"])))
        )

    return ds


def init_spoken_swag_dpo_dataset(cfg: DictConfig) -> DatasetDict:
    """Hydra entrypoint."""
    return load_spoken_swag(cfg, sampling_rate=int(cfg.get("sampling_rate", SAMPLING_RATE)))


# ======================================================================
# Frame-aligned text stream (Moshi paper Eq. 5)
# ======================================================================

def build_aligned_text_stream(
    words_with_timestamps: list[dict],
    tokenizer,
    n_frames: int,
    *,
    frame_rate: float = FRAME_RATE,
    pad_id: int = 0,
    epad_id: int = 1,
) -> list[int]:
    """Build a frame-aligned Inner Monologue text stream of length n_frames.

    Algorithm (Moshi paper §3.4.4):
      - Start with all-PAD buffer of length n_frames.
      - For each word with start_time, compute t_i = round(start_time * 12.5).
      - Tokenize the word → [w_1, ..., w_{n_i}] (Moshi's SentencePiece vocab).
      - Place EPAD at t_i - 1 (or at index 0 if t_i == 0).
      - Place word tokens starting at t_i.
      - Don't overwrite slots already holding a non-PAD token.
    """
    stream = [pad_id] * n_frames
    words_sorted = sorted(words_with_timestamps, key=lambda w: w["start"])

    for word_info in words_sorted:
        start_time = word_info["start"]
        word = word_info["word"]

        word_token_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_token_ids:
            continue

        t_i = int(round(start_time * frame_rate))
        if t_i >= n_frames:
            break  # word starts after this audio segment ends

        if t_i == 0:
            # Paper special case: shift word right by 1 to make room for EPAD.
            if stream[0] == pad_id:
                stream[0] = epad_id
            t_i = 1
        else:
            # Insert EPAD only if that slot is still free.
            if stream[t_i - 1] == pad_id:
                stream[t_i - 1] = epad_id

        for j, tok in enumerate(word_token_ids):
            pos = t_i + j
            if pos >= n_frames:
                break
            if stream[pos] == pad_id:
                stream[pos] = tok
            # else: previous word's tokens are still here; skip silently.

    return stream


# ======================================================================
# Waveform → frame-boundary-aligned waveform
# ======================================================================

def pad_waveform_to_frame_boundary(
    wav: np.ndarray, samples_per_frame: int = SAMPLES_PER_FRAME
) -> np.ndarray:
    """Right-pad with zeros so len(wav) is a multiple of samples_per_frame.

    Guarantees `len(wav) // samples_per_frame` is the exact frame count.
    """
    n = len(wav)
    remainder = n % samples_per_frame
    if remainder == 0:
        return wav
    pad = samples_per_frame - remainder
    return np.concatenate([wav, np.zeros(pad, dtype=wav.dtype)])


# ======================================================================
# Collator
# ======================================================================

@dataclass
class SpokenSwagMoshiCollator:
    """Turn SpokenSwag rows into a Moshi DPO batch.

    REQUIRED arguments (no defaults — must be passed by the caller):
      tokenizer:      Moshi's text tokenizer (from AutoTokenizer.from_pretrained).
      pad_token_id:   the text-stream PAD id. For `kmhf/hf-moshiko` this is 3,
                      matching Moshi's config `existing_text_padding_id`.
                      NOTE: the tokenizer's own `pad_token_id` is None because
                      `<pad>` is an added-special token that was never
                      registered in `special_tokens_map`. Pass 3 explicitly.
      epad_token_id:  the "end-of-padding, next word starts" marker. For
                      `kmhf/hf-moshiko` this is 0 (the `<unk>` token, which
                      Kyutai repurposed as EPAD — see the HF Moshi docs
                      example that renders "Hello, I'm Moshi" as
                      `Hello,<pad><unk>I'm Moshi`).

    OPTIONAL arguments:
      mimi_encoder:        Mimi audio codec. If provided, the collator encodes
                           audio to RVQ codes here. If None, raw waveforms are
                           emitted in the batch and the trainer must encode.
      feature_extractor:   kept for interface compatibility; currently unused.
      sampling_rate:       24000 for Moshi.
      max_prompt_seconds/max_completion_seconds: optional truncation caps.
      use_text_alignment:  False → PAD-only text stream (Option A, audio-only).
                           True  → frame-aligned from word timestamps (Option B).
    """

    # Required (no defaults — dataclass ordering: required fields must come
    # before fields with defaults).
    tokenizer: Any
    pad_token_id: int
    epad_token_id: int

    # Optional.
    mimi_encoder: Any = None           # kyutai/mimi MimiModel
    feature_extractor: Any = None      # kept for interface compat; unused here
    sampling_rate: int = SAMPLING_RATE
    max_prompt_seconds: Optional[float] = None
    max_completion_seconds: Optional[float] = None
    use_text_alignment: bool = False

    def __post_init__(self):
        # Strict validation: the common cause of silent bugs in this pipeline
        # is writing the wrong special-token id into the text stream, so we
        # refuse to construct without explicit, validated values.
        if self.tokenizer is None:
            raise ValueError("SpokenSwagMoshiCollator: `tokenizer` is required.")
        if not isinstance(self.pad_token_id, int):
            raise TypeError(
                f"SpokenSwagMoshiCollator: `pad_token_id` must be int, got "
                f"{type(self.pad_token_id).__name__}. For kmhf/hf-moshiko "
                f"use 3 (Moshi config `existing_text_padding_id`)."
            )
        if not isinstance(self.epad_token_id, int):
            raise TypeError(
                f"SpokenSwagMoshiCollator: `epad_token_id` must be int, got "
                f"{type(self.epad_token_id).__name__}. For kmhf/hf-moshiko "
                f"use 0 (<unk>, repurposed as EPAD in the HF port)."
            )
        if self.pad_token_id == self.epad_token_id:
            logger.warning(
                "pad_token_id == epad_token_id == %d. This is valid but "
                "collapses the end-of-padding signal — word onsets will not "
                "be distinguishable from ordinary padding. Only do this if "
                "you deliberately want to skip the EPAD signal.",
                self.pad_token_id,
            )

        logger.info(
            "Collator init: use_text_alignment=%s, pad_id=%d, epad_id=%d, "
            "mimi_encoder=%s",
            self.use_text_alignment, self.pad_token_id, self.epad_token_id,
            "yes" if self.mimi_encoder is not None else "no",
        )

    # ------------------------------------------------------------------
    def __call__(self, features: list[dict]) -> dict:
        # ---- 1. Extract & truncate audio ----
        def _truncate(wav, max_sec):
            if max_sec is None:
                return wav
            return wav[: int(max_sec * self.sampling_rate)]

        p_wav = [_truncate(f["prompt"]["array"], self.max_prompt_seconds) for f in features]
        c_wav = [_truncate(f["chosen"]["array"], self.max_completion_seconds) for f in features]
        r_wav = [_truncate(f["rejected"]["array"], self.max_completion_seconds) for f in features]

        # ---- 2. Pad each clip to whole-frame length ----
        p_wav = [pad_waveform_to_frame_boundary(w) for w in p_wav]
        c_wav = [pad_waveform_to_frame_boundary(w) for w in c_wav]
        r_wav = [pad_waveform_to_frame_boundary(w) for w in r_wav]

        # ---- 3. Concatenate prompt ⊕ completion ----
        concat_chosen = [np.concatenate([p, c]) for p, c in zip(p_wav, c_wav)]
        concat_rejected = [np.concatenate([p, r]) for p, r in zip(p_wav, r_wav)]

        p_nframes = [len(w) // SAMPLES_PER_FRAME for w in p_wav]
        c_nframes = [len(w) // SAMPLES_PER_FRAME for w in c_wav]
        r_nframes = [len(w) // SAMPLES_PER_FRAME for w in r_wav]

        max_frames_c = max(pn + cn for pn, cn in zip(p_nframes, c_nframes))
        max_frames_r = max(pn + rn for pn, rn in zip(p_nframes, r_nframes))

        # ---- 4. Batch-pad waveforms to common length ----
        c_moshi_wav = self._batch_pad(concat_chosen, max_frames_c)
        r_moshi_wav = self._batch_pad(concat_rejected, max_frames_r)

        # User-side: silent (SpokenSwag is monologue continuation).
        c_user_wav = torch.zeros_like(c_moshi_wav)
        r_user_wav = torch.zeros_like(r_moshi_wav)

        # ---- 5. Build attention + completion masks ----
        c_attention_mask = self._build_attention_mask(p_nframes, c_nframes, max_frames_c)
        r_attention_mask = self._build_attention_mask(p_nframes, r_nframes, max_frames_r)
        c_completion_mask = self._build_completion_mask(p_nframes, c_nframes, max_frames_c)
        r_completion_mask = self._build_completion_mask(p_nframes, r_nframes, max_frames_r)

        # ---- 6. Build text streams ----
        if self.use_text_alignment:
            c_text_ids = self._build_aligned_text(
                features, "prompt_alignment", "chosen_alignment",
                p_nframes, c_nframes, max_frames_c,
            )
            r_text_ids = self._build_aligned_text(
                features, "prompt_alignment", "rejected_alignment",
                p_nframes, r_nframes, max_frames_r,
            )
        else:
            c_text_ids = torch.full((len(features), max_frames_c), self.pad_token_id, dtype=torch.long)
            r_text_ids = torch.full((len(features), max_frames_r), self.pad_token_id, dtype=torch.long)

        batch = {
            "chosen_input_ids": c_text_ids,
            "rejected_input_ids": r_text_ids,
            "chosen_attention_mask": c_attention_mask,
            "rejected_attention_mask": r_attention_mask,
            "chosen_completion_mask": c_completion_mask,
            "rejected_completion_mask": r_completion_mask,
        }

        # ---- 7. Mimi encode (or leave raw) ----
        if self.mimi_encoder is not None:
            with torch.no_grad():
                # Match Mimi's device AND dtype. The model is loaded in bf16.
                mimi_param = next(self.mimi_encoder.parameters())
                mimi_device = mimi_param.device
                mimi_dtype = mimi_param.dtype
        
                def _encode(wav):
                    # wav is [B, T_samples] float32 on CPU. Mimi wants [B, 1, T_samples] on its device+dtype.
                    return self.mimi_encoder.encode(
                        wav.unsqueeze(1).to(device=mimi_device, dtype=mimi_dtype),
                        num_quantizers=NUM_CODEBOOKS,   # restrict to 8 (Moshi's setting)
                    ).audio_codes
        
            c_moshi_codes = _encode(c_moshi_wav)
            c_user_codes = _encode(c_user_wav)
            r_moshi_codes = _encode(r_moshi_wav)
            r_user_codes = _encode(r_user_wav)
            batch["chosen_moshi_audio_codes"] = c_moshi_codes.cpu()
            batch["chosen_user_audio_codes"] = c_user_codes.cpu()
            batch["rejected_moshi_audio_codes"] = r_moshi_codes.cpu()
            batch["rejected_user_audio_codes"] = r_user_codes.cpu()
        else:
            batch["chosen_moshi_input_values"] = c_moshi_wav
            batch["chosen_user_input_values"] = c_user_wav
            batch["rejected_moshi_input_values"] = r_moshi_wav
            batch["rejected_user_input_values"] = r_user_wav

        return batch

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    @staticmethod
    def _batch_pad(waves: list[np.ndarray], target_frames: int) -> torch.Tensor:
        out = np.zeros((len(waves), target_frames * SAMPLES_PER_FRAME), dtype=np.float32)
        for i, w in enumerate(waves):
            out[i, : len(w)] = w
        return torch.from_numpy(out)

    @staticmethod
    def _build_attention_mask(
        prompt_ns: list[int], comp_ns: list[int], max_frames: int
    ) -> torch.Tensor:
        """1 for real (prompt or completion) frames, 0 for right-pad."""
        mask = torch.zeros((len(prompt_ns), max_frames), dtype=torch.bool)
        for i, (pn, cn) in enumerate(zip(prompt_ns, comp_ns)):
            mask[i, : pn + cn] = True
        return mask

    @staticmethod
    def _build_completion_mask(
        prompt_ns: list[int], comp_ns: list[int], max_frames: int
    ) -> torch.Tensor:
        """1 for completion frames only."""
        mask = torch.zeros((len(prompt_ns), max_frames), dtype=torch.bool)
        for i, (pn, cn) in enumerate(zip(prompt_ns, comp_ns)):
            mask[i, pn : pn + cn] = True
        return mask

    def _build_aligned_text(
        self,
        features: list[dict],
        prompt_align_key: str,
        comp_align_key: str,
        prompt_ns: list[int],
        comp_ns: list[int],
        max_frames: int,
    ) -> torch.Tensor:
        B = len(features)
        out = torch.full((B, max_frames), self.pad_token_id, dtype=torch.long)

        for i, f in enumerate(features):
            if prompt_align_key not in f or comp_align_key not in f:
                raise KeyError(
                    f"use_text_alignment=True but example missing "
                    f"'{prompt_align_key}' or '{comp_align_key}'. Run an alignment "
                    f"preprocessing pass (see moshi-finetune/annotate.py)."
                )

            prompt_stream = build_aligned_text_stream(
                f[prompt_align_key], self.tokenizer,
                n_frames=prompt_ns[i],
                pad_id=self.pad_token_id, epad_id=self.epad_token_id,
            )
            out[i, : prompt_ns[i]] = torch.tensor(prompt_stream, dtype=torch.long)

            comp_stream = build_aligned_text_stream(
                f[comp_align_key], self.tokenizer,
                n_frames=comp_ns[i],
                pad_id=self.pad_token_id, epad_id=self.epad_token_id,
            )
            out[i, prompt_ns[i] : prompt_ns[i] + comp_ns[i]] = torch.tensor(
                comp_stream, dtype=torch.long
            )

        return out