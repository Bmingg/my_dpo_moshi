"""
Preprocess SpokenSwag for Moshi DPO with frame-aligned text token IDs.

Adapts the approach from kyutai-labs/moshi-finetune/annotate.py:
  1. whisper_timestamped → word-level (start, end) timestamps per audio
  2. Moshi SentencePiece tokenizer → sub-word token IDs per word
  3. Place token IDs at Mimi frame positions (12.5 Hz) → padded sequence

Produces three new columns per example:
  prompt_text_token_ids    list[int]   len ≈ prompt_audio_samples / 1920
  chosen_text_token_ids    list[int]   len ≈ chosen_audio_samples / 1920
  rejected_text_token_ids  list[int]   len ≈ rejected_audio_samples / 1920

These plug directly into SpokenSwagMoshiCollator(use_text_alignment=True).

Usage:
  # Smoke test (10 examples, local GPU)
  python preprocess_spoken_swag.py \
      --hf_push_repo your-username/SpokenSwag-aligned \
      --train_take 10 --val_take 10

  # Full run (takes ~12-24h on a single GPU depending on whisper model)
  python preprocess_spoken_swag.py \
      --hf_push_repo your-username/SpokenSwag-aligned \
      --whisper_model medium

  # Resume after crash (skips already-processed shards)
  python preprocess_spoken_swag.py \
      --hf_push_repo your-username/SpokenSwag-aligned \
      --checkpoint_dir ./swag_ckpts --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as TAF

# ---------- lazy imports (fail fast with helpful messages) ----------
try:
    import whisper_timestamped as whisper
except ImportError:
    raise ImportError(
        "pip install whisper-timestamped   "
        "(or: uv pip install whisper-timestamped)"
    )

from datasets import Audio, Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_spoken_swag")

# ========================== constants ==========================

SPOKEN_SWAG_REPO    = "slprl/SpokenSwag"
MOSHI_SAMPLE_RATE   = 24_000
WHISPER_SAMPLE_RATE = 16_000
MIMI_HOP_LENGTH     = 1920          # 24000 / 12.5
MIMI_FRAME_RATE     = 12.5          # Hz
TEXT_PAD_ID          = 3             # Moshi's existing_text_padding_id
AUDIO_COLUMNS       = ("prompt", "chosen", "rejected")


# ====================== whisper + alignment ======================

def get_word_timestamps(
    audio_array: np.ndarray,
    sr: int,
    whisper_model,
    language: str = "en",
) -> list[dict]:
    """Run whisper_timestamped → list of {text, start, end} dicts.

    Mirrors the core of moshi-finetune/annotate.py::process_one, but
    operates on a numpy array instead of reading from disk.
    """
    # Resample to 16 kHz for Whisper (SpokenSwag is 24 kHz after cast).
    if sr != WHISPER_SAMPLE_RATE:
        t = torch.from_numpy(audio_array).unsqueeze(0).float()
        t = TAF.resample(t, sr, WHISPER_SAMPLE_RATE)
        audio_16k = t.squeeze(0).numpy()
    else:
        audio_16k = audio_array.copy()

    duration_sec = len(audio_16k) / WHISPER_SAMPLE_RATE
    result = whisper.transcribe(
        whisper_model,
        audio_16k,
        language=language,
        # annotate.py uses auditok VAD for clips > 10 s
        vad="auditok" if duration_sec > 10 else None,
        best_of=5,
        beam_size=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        verbose=None,
    )

    words: list[dict] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            try:
                words.append({
                    "text":  w["text"],
                    "start": float(w["start"]),
                    "end":   float(w["end"]),
                })
            except KeyError:
                logger.warning("Malformed word entry: %s", w)
    return words


def align_tokens_to_frames(
    words: list[dict],
    tokenizer,
    num_frames: int,
    *,
    frame_rate: float = MIMI_FRAME_RATE,
    pad_id: int = TEXT_PAD_ID,
) -> list[int]:
    """Place SentencePiece token IDs at Mimi frame positions.

    For each word with onset at t seconds:
      start_frame = floor(t * 12.5)
      tokens      = tokenizer.encode(word_text)
      aligned[start_frame + j] = tokens[j]   (for j in range(len(tokens)))

    Frames without a token stay at pad_id (3).

    This matches the moshi-finetune data pipeline where text tokens are
    placed at the audio frame corresponding to word onset, and the rest
    of the text stream is padded.
    """
    aligned = [pad_id] * num_frames

    for word_info in words:
        text = word_info["text"]
        start_sec = word_info["start"]
        start_frame = int(start_sec * frame_rate)

        # whisper_timestamped returns words with leading spaces like " hello".
        # SentencePiece handles this naturally — the ▁ prefix encodes the
        # word boundary, which is exactly what Moshi's tokenizer expects.
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        for j, tid in enumerate(token_ids):
            f = start_frame + j
            if f < num_frames:
                aligned[f] = tid
            # If tokens overflow into the next word's frames, they're
            # silently dropped. This is rare (12.5 Hz ≈ 80 ms/frame,
            # and most sub-words take at least that long to speak).

    return aligned


def compute_num_frames(audio_array: np.ndarray, sr: int) -> int:
    """Number of Mimi frames for an audio array at the given sample rate."""
    # Convert to 24 kHz sample count, then divide by hop length.
    num_samples_24k = int(len(audio_array) * MOSHI_SAMPLE_RATE / sr)
    return num_samples_24k // MIMI_HOP_LENGTH


# ====================== dataset processing ======================

def process_example(
    example: dict,
    whisper_model,
    tokenizer,
    language: str = "en",
) -> dict:
    """Add {prompt,chosen,rejected}_text_token_ids to a single example."""
    result = {}

    for col in AUDIO_COLUMNS:
        audio_data = example[col]
        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]

        num_frames = compute_num_frames(audio_array, sr)

        # Skip very short clips (< 2 frames ≈ 160 ms)
        if num_frames < 2:
            logger.warning(
                "Very short audio in column '%s' (%d frames). "
                "Filling with PAD.", col, num_frames,
            )
            result[f"{col}_text_token_ids"] = [TEXT_PAD_ID] * max(num_frames, 1)
            continue

        words = get_word_timestamps(audio_array, sr, whisper_model, language)
        aligned = align_tokens_to_frames(words, tokenizer, num_frames)
        result[f"{col}_text_token_ids"] = aligned

    return result


def process_split(
    ds: Dataset,
    whisper_model,
    tokenizer,
    *,
    language: str = "en",
    checkpoint_dir: Path | None = None,
    split_name: str = "train",
    save_every: int = 500,
    resume: bool = False,
) -> Dataset:
    """Process an entire split, with periodic checkpointing.

    Returns a new Dataset with the three *_text_token_ids columns added.
    """
    n = len(ds)
    # Accumulate new columns.
    new_cols = {
        f"{col}_text_token_ids": [None] * n for col in AUDIO_COLUMNS
    }

    # Resume support: load partial results if available.
    start_idx = 0
    ckpt_file = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = checkpoint_dir / f"{split_name}_progress.json"
        if resume and ckpt_file.exists():
            with open(ckpt_file) as f:
                saved = json.load(f)
            start_idx = saved["next_idx"]
            for col in AUDIO_COLUMNS:
                key = f"{col}_text_token_ids"
                new_cols[key][: start_idx] = saved[key][: start_idx]
            logger.info(
                "Resumed %s from index %d / %d", split_name, start_idx, n
            )

    t0 = time.time()
    for i in range(start_idx, n):
        example = ds[i]
        try:
            result = process_example(example, whisper_model, tokenizer, language)
            for col in AUDIO_COLUMNS:
                key = f"{col}_text_token_ids"
                new_cols[key][i] = result[key]
        except Exception:
            logger.exception("Error processing %s[%d], filling with PAD.", split_name, i)
            for col in AUDIO_COLUMNS:
                audio_data = example[col]
                sr = audio_data["sampling_rate"]
                nf = compute_num_frames(
                    np.array(audio_data["array"], dtype=np.float32), sr
                )
                new_cols[f"{col}_text_token_ids"][i] = [TEXT_PAD_ID] * max(nf, 1)

        # Periodic logging + checkpoint.
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - start_idx) / elapsed
            eta = (n - i - 1) / max(rate, 1e-6)
            logger.info(
                "%s: %d / %d  (%.1f ex/s, ETA %.0f min)",
                split_name, i + 1, n, rate, eta / 60,
            )

        if ckpt_file is not None and (i + 1) % save_every == 0:
            _save_checkpoint(ckpt_file, new_cols, i + 1)

        # Prevent GPU OOM from whisper accumulating graphs.
        if (i + 1) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Final checkpoint.
    if ckpt_file is not None:
        _save_checkpoint(ckpt_file, new_cols, n)

    # Add new columns to the dataset.
    for key, vals in new_cols.items():
        ds = ds.add_column(key, vals)

    return ds


def _save_checkpoint(path: Path, new_cols: dict, next_idx: int):
    """Save processing progress to a JSON file."""
    data = {"next_idx": next_idx}
    for key, vals in new_cols.items():
        data[key] = vals[:next_idx]
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    Path(tmp).rename(path)
    logger.info("Checkpoint saved at index %d → %s", next_idx, path)


# ========================== sanity checks ==========================

def print_alignment_sample(ds: Dataset, tokenizer, idx: int = 0):
    """Print a human-readable view of one aligned example for debugging."""
    ex = ds[idx]
    print(f"\n{'='*60}")
    print(f"Example {idx}")
    print(f"{'='*60}")

    for col in AUDIO_COLUMNS:
        key = f"{col}_text_token_ids"
        ids = ex[key]
        text_field = f"{col}_text"
        original_text = ex.get(text_field, "(no text field)")

        # Decode non-PAD tokens back to text.
        non_pad = [t for t in ids if t != TEXT_PAD_ID]
        decoded = tokenizer.decode(non_pad) if non_pad else "(all PAD)"

        n_frames = len(ids)
        n_text = len(non_pad)
        coverage = n_text / max(n_frames, 1) * 100

        print(f"\n  {col}:")
        print(f"    Original text : {original_text}")
        print(f"    Decoded tokens: {decoded}")
        print(f"    Frames: {n_frames}, text tokens: {n_text} ({coverage:.1f}% coverage)")

        # Show first 20 frames with token details.
        preview_n = min(40, n_frames)
        frame_strs = []
        for fi in range(preview_n):
            tid = ids[fi]
            if tid == TEXT_PAD_ID:
                frame_strs.append(f"  [{fi:3d}] PAD")
            else:
                tok_text = tokenizer.decode([tid])
                frame_strs.append(f"  [{fi:3d}] {tid:5d} → '{tok_text}'")
        print(f"    First {preview_n} frames:")
        for s in frame_strs:
            print(f"      {s}")


# ============================== main ===============================

def main():
    parser = argparse.ArgumentParser(
        description="Add frame-aligned text token IDs to SpokenSwag for Moshi DPO."
    )
    parser.add_argument(
        "--hf_push_repo", type=str, required=True,
        help="HF repo to push the processed dataset (e.g. your-user/SpokenSwag-aligned)."
    )
    parser.add_argument(
        "--source_repo", type=str, default=SPOKEN_SWAG_REPO,
        help="Source SpokenSwag HF repo."
    )
    parser.add_argument(
        "--tokenizer_repo", type=str, default="kmhf/hf-moshiko",
        help="HF repo for Moshi's text tokenizer."
    )
    parser.add_argument(
        "--whisper_model", type=str, default="medium",
        help="Whisper model size. 'medium' recommended (matches annotate.py)."
    )
    parser.add_argument(
        "--language", type=str, default="en",
        help="Language for Whisper."
    )
    parser.add_argument(
        "--checkpoint_dir", type=Path, default=None,
        help="Directory for saving processing progress (enables resume)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint."
    )
    parser.add_argument(
        "--save_every", type=int, default=500,
        help="Checkpoint frequency (in examples)."
    )
    parser.add_argument(
        "--train_take", type=int, default=None,
        help="Process only this many training examples (for smoke tests)."
    )
    parser.add_argument(
        "--val_take", type=int, default=None,
        help="Process only this many validation examples."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="HF datasets cache directory."
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Push as a private HF repo."
    )
    args = parser.parse_args()

    # ---- Load dataset ----
    logger.info("Loading dataset %s ...", args.source_repo)
    ds = load_dataset(args.source_repo, cache_dir=args.cache_dir)

    # Cast audio to 24 kHz (Moshi's rate). SpokenSwag is natively 24 kHz,
    # but the cast ensures consistency if the source rate ever changes.
    audio_feat = Audio(sampling_rate=MOSHI_SAMPLE_RATE)
    for col in AUDIO_COLUMNS:
        ds = ds.cast_column(col, audio_feat)

    # Optional truncation for smoke tests.
    if args.train_take is not None and "train" in ds:
        ds["train"] = ds["train"].select(range(min(args.train_take, len(ds["train"]))))
    if args.val_take is not None and "validation" in ds:
        ds["validation"] = ds["validation"].select(
            range(min(args.val_take, len(ds["validation"])))
        )
    for split_name, split_ds in ds.items():
        logger.info("  %s: %d examples", split_name, len(split_ds))

    # ---- Load models ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Whisper '%s' on %s ...", args.whisper_model, device)
    whisper_model = whisper.load_model(args.whisper_model, device=device)

    logger.info("Loading tokenizer from %s ...", args.tokenizer_repo)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_repo)

    # Quick tokenizer sanity check.
    test_tokens = tokenizer.encode(" hello world", add_special_tokens=False)
    logger.info(
        "Tokenizer test: ' hello world' → %s (decoded: '%s')",
        test_tokens, tokenizer.decode(test_tokens),
    )

    # ---- Process each split ----
    processed = {}
    for split_name, split_ds in ds.items():
        logger.info("Processing split '%s' (%d examples) ...", split_name, len(split_ds))
        processed[split_name] = process_split(
            split_ds,
            whisper_model,
            tokenizer,
            language=args.language,
            checkpoint_dir=args.checkpoint_dir,
            split_name=split_name,
            save_every=args.save_every,
            resume=args.resume,
        )

    result_ds = DatasetDict(processed)

    # ---- Sanity check: print a sample ----
    logger.info("Alignment sample:")
    for split_name in result_ds:
        print_alignment_sample(result_ds[split_name], tokenizer, idx=0)

    # ---- Push to Hub ----
    logger.info("Pushing to %s ...", args.hf_push_repo)
    result_ds.push_to_hub(args.hf_push_repo, private=args.private)
    logger.info("Done! Dataset available at https://huggingface.co/datasets/%s", args.hf_push_repo)


if __name__ == "__main__":
    main()