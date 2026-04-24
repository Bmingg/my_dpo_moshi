"""
Data loading for Moshi DPO training on `slprl/SpokenSwag`.

SpokenSwag schema (from https://huggingface.co/datasets/slprl/SpokenSwag):

    {
        "speaker":      str,        # one of 4 Kokoro voices
        "prompt_text":  str,
        "chosen_text":  str,
        "rejected_text":str,
        "prompt":       Audio,      # {"array": np.ndarray, "sampling_rate": int, "path": str}
        "chosen":       Audio,
        "rejected":     Audio,
        "auto_bleu2":   float,
    }

    splits: train (47,900) / validation (20,000)

This module provides:

1. `load_spoken_swag` — thin `datasets.load_dataset` wrapper that re-samples
   audio to Moshi's expected rate and (optionally) drops repetitive examples
   using the existing `auto_bleu2` field.

2. `SpokenSwagMoshiCollator` — a SCAFFOLD collator that turns a batch of
   {prompt, chosen, rejected} audio + text into the (text + audio waveform)
   tensors that `MoshiForConditionalGeneration` actually consumes. The
   non-trivial bits — Mimi pre-tokenisation, dual-stream alignment, and
   producing the `(prompt_*, chosen_*, rejected_*)` keys that
   `trl.DPOTrainer` expects — are explicitly left as TODOs.
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


SPOKEN_SWAG_REPO = "slprl/SpokenSwag"
MIMI_HOP_LENGTH = 1920
MIMI_NUM_CODEBOOKS = 8
MOSHI_TEXT_PAD_ID  = 3
MOSHI_TEXT_EPAD_ID = 0

def load_spoken_swag(
    cfg: DictConfig,
    *,
    sampling_rate: int = 24000,
) -> DatasetDict:
    """Load the SpokenSwag DPO dataset.

    Args:
        cfg: a Hydra config (see `config/data/spoken_swag.yaml`). Recognised
            keys:

            - `repo_id`: HF repo id (default `slprl/SpokenSwag`).
            - `cache_dir`: optional HF cache override.
            - `num_proc`: parallelism for `.filter` / `.cast_column`.
            - `repetition_filter` (bool): drop rows where `auto_bleu2`
              exceeds `max_auto_bleu`.
            - `max_auto_bleu` (float): threshold for the above.
            - `train_take` / `val_take` (int | None): optional truncation,
              handy for smoke tests.
        sampling_rate: target sample rate for the Audio columns. Moshi /
            Mimi expects 24 kHz, which is also SpokenSwag's native rate.

    Returns:
        A `DatasetDict` with `train` and `validation` splits.
    """
    repo_id = cfg.get("repo_id", SPOKEN_SWAG_REPO)
    cache_dir = cfg.get("cache_dir", None)
    num_proc = cfg.get("num_proc", 4)

    logger.info("Loading %s (cache_dir=%s)", repo_id, cache_dir)
    ds: DatasetDict = load_dataset(repo_id, cache_dir=cache_dir)

    # Re-sample audio columns to Moshi's expected rate.
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
        logger.info("repetition_filter %s -> %s (max_auto_bleu=%s)",
                    before, after, max_auto_bleu)

    train_take = cfg.get("train_take", None)
    val_take = cfg.get("val_take", None)
    if train_take is not None and "train" in ds:
        ds["train"] = ds["train"].select(range(min(train_take, len(ds["train"]))))
    if val_take is not None and "validation" in ds:
        ds["validation"] = ds["validation"].select(
            range(min(val_take, len(ds["validation"])))
        )

    return ds


@dataclass
class SpokenSwagMoshiCollator:
    """SCAFFOLD collator: SpokenSwag rows → Moshi DPO batch.

    The TRL `DPOTrainer` (and our `SLAMMoshiDPOTrainer` subclass) ultimately
    calls `concatenated_forward` on a batch of dicts that look like:

        {
            "prompt_input_ids":   LongTensor [B, Tp],
            "chosen_input_ids":   LongTensor [B, Tc],
            "rejected_input_ids": LongTensor [B, Tr],
            ...attention masks...
        }

    For Moshi this is wrong on two counts:

    1. The "input" is not a single text token stream — it's the dual audio
       stream `(user_input_values, moshi_input_values)` plus the Inner
       Monologue text stream `input_ids`.
    2. The audio stream is itself 8 RVQ codebooks per frame after Mimi.

    For a real implementation, this collator should:

      a. Encode `prompt`, `chosen`, `rejected` raw audio through
         `feature_extractor` → `input_values` at 24 kHz.
      b. Run Mimi to get RVQ codes (or pass `input_values` straight to
         `MoshiForConditionalGeneration`, which encodes internally).
      c. Build the dual-stream layout: user-side audio = SpokenSwag prompt,
         Moshi-side audio = SpokenSwag chosen / rejected. The matching text
         stream is the corresponding `*_text` field tokenised with
         `tokenizer`. Pad both streams to the same temporal length, since
         Moshi requires aligned dual streams.
      d. Emit `prompt_*`, `chosen_*`, `rejected_*` tensors keyed by stream
         (`*_input_ids`, `*_user_input_values`, `*_moshi_input_values`,
         `*_attention_mask`) so that `concatenated_forward` can stack them.

    None of (a)-(d) is implemented here — they are TODOs for the user.
    """
    mimi_model: Any
    sampling_rate: int = 24000
    max_prompt_seconds: Optional[float] = None
    max_completion_seconds: Optional[float] = None
    use_text_alignment: bool = False
    text_pad_id: int = MOSHI_TEXT_PAD_ID
    text_epad_id: int = MOSHI_TEXT_EPAD_ID
    device: str = "cpu"

    def _truncate_audio(
        self, 
        features: list[dict],
    ) -> tuple[list, list, list]:
        prompt_audio   = [np.array(f["prompt"]["array"],   dtype=np.float32) for f in features]
        chosen_audio   = [np.array(f["chosen"]["array"],   dtype=np.float32) for f in features]
        rejected_audio = [np.array(f["rejected"]["array"], dtype=np.float32) for f in features]

        if self.max_prompt_seconds is not None:
            cap = int(self.max_prompt_seconds * self.sampling_rate)
            prompt_audio = [a[:cap] for a in prompt_audio]

        if self.max_completion_seconds is not None:
            cap = int(self.max_completion_seconds * self.sampling_rate)
            chosen_audio   = [a[:cap] for a in chosen_audio]
            rejected_audio = [a[:cap] for a in rejected_audio]

        return prompt_audio, chosen_audio, rejected_audio
    
    def _encode_mimi(
        self,
        waveforms: list[np.ndarray],
    ) -> torch.LongTensor:
        max_len = max(w.shape[0] for w in waveforms)
        padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
        for i, w in enumerate(waveforms):
            padded[i, :w.shape[0]] = w

        # Match Mimi's device AND dtype (model is bf16 in our setup).
        mimi_param = next(self.mimi_model.parameters())
        audio_tensor = torch.from_numpy(padded).unsqueeze(1).to(
            device=mimi_param.device, dtype=mimi_param.dtype
        )

        with torch.no_grad():
            # num_quantizers=8 - Moshi uses 8 codebooks; Mimi defaults to 32.
            codes = self.mimi_model.encode(
                audio_tensor, num_quantizers=MIMI_NUM_CODEBOOKS
            ).audio_codes

        return codes.long()
    
    @staticmethod
    def _make_attention_mask(
        frame_counts: list[int],
        max_frames: int, 
    ) -> torch.BoolTensor:
        mask = torch.zeros(len(frame_counts), max_frames, dtype = torch.bool)
        for i, n in enumerate(frame_counts):
            mask[i, :n] = True
        return mask
    
    @staticmethod
    def _make_completion_mask(
        prompt_frame_counts: list[int],
        completion_frame_counts: list[int],
        max_frames: int, 
    ) -> torch.BoolTensor:
        mask = torch.zeros(
            len(prompt_frame_counts), max_frames, dtype = torch.bool
        )
        for i, (p, c) in enumerate(zip(prompt_frame_counts, completion_frame_counts)):
            mask[i, p : p + c] = True
        return mask
    
    @staticmethod
    def _pad_token_ids(
        sequences: list[list[int]],
        pad_id: int, 
    ) -> torch.LongTensor:
        max_len = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return ids
    
    def _build_side(
        self, 
        prompt_audio: list[np.ndarray],
        completion_audio: list[np.ndarray],
        input_ids_list: list[list[int]],
    ) -> dict:
        prompt_frames     = [len(a) // MIMI_HOP_LENGTH for a in prompt_audio]
        completion_frames = [len(a) // MIMI_HOP_LENGTH for a in completion_audio]
        total_frames      = [p + c for p, c in zip(prompt_frames, completion_frames)]
        max_frames        = max(total_frames)

        concat_audio = [
            np.concatenate([p, c], axis=0)
            for p, c in zip(prompt_audio, completion_audio)
        ]

        moshi_codes = self._encode_mimi(concat_audio)
        moshi_codes = moshi_codes[:, :, :max_frames]

        user_codes = torch.zeros_like(moshi_codes)

        attention_mask = self._make_attention_mask(total_frames, max_frames)

        completion_mask = self._make_completion_mask(
            prompt_frames, completion_frames, max_frames
        )

        input_ids = self._pad_token_ids(input_ids_list, self.text_pad_id)
        if input_ids.shape[1] < max_frames:
            extra = torch.full(
                (input_ids.shape[0], max_frames - input_ids.shape[1]),
                self.text_pad_id,
                dtype=torch.long,
            )
            input_ids = torch.cat([input_ids, extra], dim=1)
        else:
            input_ids = input_ids[:, :max_frames]

        return {
            "moshi_audio_codes": moshi_codes,    
            "user_audio_codes":  user_codes,      
            "input_ids":         input_ids,       
            "attention_mask":    attention_mask,  
            "completion_mask":   completion_mask, 
        }
        
    def __call__(self, features: list[dict]) -> dict:
        # TODO(user): implement the (a)-(d) pipeline above.
        #
        # A minimal sketch (not enough for real training, but enough to see
        # the shape) would look like:
        #
        #     prompt_audio = [f["prompt"]["array"] for f in features]
        #     chosen_audio = [f["chosen"]["array"] for f in features]
        #     rejected_audio = [f["rejected"]["array"] for f in features]
        #
        #     prompt_iv = self.feature_extractor(
        #         raw_audio=prompt_audio,
        #         sampling_rate=self.sampling_rate,
        #         return_tensors="pt",
        #         padding=True,
        #     )
        #     chosen_iv = self.feature_extractor(...)
        #     rejected_iv = self.feature_extractor(...)
        #
        #     prompt_text_ids = self.tokenizer(
        #         [f["prompt_text"] for f in features],
        #         return_tensors="pt", padding=True,
        #     )
        #
        #     return {
        #         "prompt_input_ids":         prompt_text_ids.input_ids,
        #         "prompt_user_input_values": prompt_iv.input_values,
        #         "chosen_moshi_input_values":   chosen_iv.input_values,
        #         "rejected_moshi_input_values": rejected_iv.input_values,
        #         ...
        #     }
        prompt_audio, chosen_audio, rejected_audio = self._truncate_audio(features)
        prompt_frames   = [len(a) // MIMI_HOP_LENGTH for a in prompt_audio]
        chosen_frames   = [len(a) // MIMI_HOP_LENGTH for a in chosen_audio]
        rejected_frames = [len(a) // MIMI_HOP_LENGTH for a in rejected_audio]

        if self.use_text_alignment:
            for key in (
                "prompt_text_token_ids",
                "chosen_text_token_ids",
                "rejected_text_token_ids",
            ):
                if key not in features[0]:
                    raise KeyError(
                        f"'{key}' not found in dataset example. "
                        "Run Minh's preprocessing step before setting "
                        "use_text_alignment=True."
                    )

            chosen_ids = [
                list(f["prompt_text_token_ids"])[:prompt_frames[i]] +
                list(f["chosen_text_token_ids"])[:chosen_frames[i]]
                for i, f in enumerate(features)
            ]
            rejected_ids = [
                list(f["prompt_text_token_ids"])[:prompt_frames[i]] +
                list(f["rejected_text_token_ids"])[:rejected_frames[i]]
                for i, f in enumerate(features)
            ]
        else:
            chosen_ids = [
                [self.text_pad_id] * (prompt_frames[i] + chosen_frames[i])
                for i in range(len(features))
            ]
            rejected_ids = [
                [self.text_pad_id] * (prompt_frames[i] + rejected_frames[i])
                for i in range(len(features))
            ]

        chosen_out   = self._build_side(prompt_audio, chosen_audio,   chosen_ids)
        rejected_out = self._build_side(prompt_audio, rejected_audio, rejected_ids)

        batch = {}
        for key, val in chosen_out.items():
            batch[f"chosen_{key}"] = val
        for key, val in rejected_out.items():
            batch[f"rejected_{key}"] = val

        # Pass precomputed reference logps through, if present.
        if "ref_chosen_logp" in features[0]:
            batch["ref_chosen_logp"] = torch.tensor(
                [f["ref_chosen_logp"] for f in features], dtype=torch.float32,
            )
            batch["ref_rejected_logp"] = torch.tensor(
                [f["ref_rejected_logp"] for f in features], dtype=torch.float32,
            )

        return batch

    

def init_spoken_swag_dpo_dataset(cfg: DictConfig) -> DatasetDict:
    """Hydra entrypoint mirroring `init_preference_optimization_dataset`."""
    return load_spoken_swag(cfg, sampling_rate=int(cfg.get("sampling_rate", 24000)))