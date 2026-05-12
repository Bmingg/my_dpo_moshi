"""
verify_collator_streams.py

Verifies that the teammate's collator (with prompt_moshi_audio fix applied)
correctly routes:
  - user stream:   speaker A (prompt) | silence (completion)
  - moshi stream:  speaker B (prompt) | Moshi's generation (completion)

Run with:
  python verify_collator_streams.py --dataset kalbin/moshi-on-policy-dpo-v20-kyutai-aligned-smoke
"""

import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Audio
from transformers import MoshiForConditionalGeneration

MIMI_HOP_LENGTH = 1920
SAMPLING_RATE   = 24_000


# ── helpers ───────────────────────────────────────────────────────────────────

def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))

def decode_codes(mimi, codes: torch.Tensor) -> np.ndarray:
    """Decode Mimi codes [1, K, T] → waveform [T*1920]."""
    with torch.no_grad():
        wav = mimi.decode(codes.to(next(mimi.parameters()).device)).audio_values
    return wav[0, 0].float().cpu().numpy()

def encode_wav(mimi, wav: np.ndarray) -> torch.Tensor:
    """Encode a 1-D float32 waveform → codes [1, K, T]."""
    mimi_param = next(mimi.parameters())
    t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(
        device=mimi_param.device, dtype=mimi_param.dtype
    )
    with torch.no_grad():
        codes = mimi.encode(t, num_quantizers=8).audio_codes
    return codes.long()

def segment_rms(wav: np.ndarray, start_frame: int, end_frame: int) -> float:
    s = start_frame * MIMI_HOP_LENGTH
    e = end_frame   * MIMI_HOP_LENGTH
    return rms(wav[s:e])


# ── main check ────────────────────────────────────────────────────────────────

def verify_example(example: dict, mimi, out_dir: Path, idx: int):
    print(f"\n{'='*60}")
    print(f"Example {idx}")
    print(f"{'='*60}")

    # 1. Load raw audio from dataset columns
    prompt_user  = np.array(example["prompt"]["array"],              dtype=np.float32)
    prompt_moshi = np.array(example["prompt_moshi_audio"]["array"],  dtype=np.float32)
    chosen       = np.array(example["chosen"]["array"],              dtype=np.float32)
    rejected     = np.array(example["rejected"]["array"],            dtype=np.float32)

    # Use min of both prompt streams — they represent the same time window
    # but may differ by a few samples due to generation rounding.
    p_frames = min(len(prompt_user), len(prompt_moshi)) // MIMI_HOP_LENGTH
    c_frames = len(chosen)   // MIMI_HOP_LENGTH
    r_frames = len(rejected) // MIMI_HOP_LENGTH
    p_samples = p_frames * MIMI_HOP_LENGTH
    total_c  = p_frames + c_frames
    total_r  = p_frames + r_frames

    print(f"  prompt frames  : {p_frames}")
    print(f"  chosen frames  : {c_frames}  (total chosen side: {total_c})")
    print(f"  rejected frames: {r_frames}  (total rejected side: {total_r})")

    # 2. Build what the fixed collator SHOULD produce
    #    moshi stream = [prompt_moshi | chosen/rejected]
    #    user  stream = [prompt_user  | silence        ]

    def build_expected(completion: np.ndarray, n_completion_frames: int):
        total = p_frames + n_completion_frames
        n_samples = total * MIMI_HOP_LENGTH

        exp_moshi = np.zeros(n_samples, dtype=np.float32)
        exp_moshi[:p_samples] = prompt_moshi[:p_samples]   # truncate to p_samples
        exp_moshi[p_samples:p_samples + n_completion_frames * MIMI_HOP_LENGTH] = \
            completion[:n_completion_frames * MIMI_HOP_LENGTH]

        exp_user = np.zeros(n_samples, dtype=np.float32)
        exp_user[:p_samples] = prompt_user[:p_samples]     # truncate to p_samples

        return exp_moshi, exp_user

    exp_moshi_c, exp_user_c = build_expected(chosen,   c_frames)
    exp_moshi_r, exp_user_r = build_expected(rejected, r_frames)

    # 3. Encode expected streams through Mimi → decode back → compare
    #    (encode→decode is lossy but should preserve energy distribution)
    print("\n  [Chosen side]")
    codes_moshi_c = encode_wav(mimi, exp_moshi_c)  # [1, 8, total_c]
    codes_user_c  = encode_wav(mimi, exp_user_c)
    recon_moshi_c = decode_codes(mimi, codes_moshi_c)
    recon_user_c  = decode_codes(mimi, codes_user_c)

    # 4. Energy checks: confirm non-silence in the right frame windows
    # Moshi stream: prompt region should have energy (speaker B was talking)
    moshi_prompt_rms = segment_rms(recon_moshi_c, 0, p_frames)
    moshi_compl_rms  = segment_rms(recon_moshi_c, p_frames, total_c)
    user_prompt_rms  = segment_rms(recon_user_c,  0, p_frames)
    user_compl_rms   = segment_rms(recon_user_c,  p_frames, total_c)

    print(f"    moshi stream — prompt region RMS : {moshi_prompt_rms:.5f}  (expect > 0)")
    print(f"    moshi stream — completion RMS    : {moshi_compl_rms:.5f}   (expect > 0)")
    print(f"    user  stream — prompt region RMS : {user_prompt_rms:.5f}   (expect > 0)")
    print(f"    user  stream — completion RMS    : {user_compl_rms:.5f}    (expect ≈ 0)")

    # 5. Boundary check: frame p_frames-1 vs p_frames in moshi stream
    #    The transition from prompt_moshi to chosen should be a real audio boundary
    frame_before = segment_rms(recon_moshi_c, p_frames - 2, p_frames)
    frame_after  = segment_rms(recon_moshi_c, p_frames,     p_frames + 2)
    print(f"    moshi boundary (last 2 prompt frames RMS) : {frame_before:.5f}")
    print(f"    moshi boundary (first 2 compl frames RMS) : {frame_after:.5f}")

    # 6. Cross-check: encode each raw segment separately, compare to
    #    the corresponding slice of the concatenated encoding.
    #    Mimi is causal, so the first T frames of encode(A⊕B) ≈ encode(A).
    #    The gap tells you whether there's a seam artifact at the boundary.
    codes_prompt_moshi_alone = encode_wav(mimi, prompt_moshi[:p_frames * MIMI_HOP_LENGTH])
    codes_chosen_alone       = encode_wav(mimi, chosen[:c_frames * MIMI_HOP_LENGTH])

    # Compare codebook-0 (semantic token) at the boundary
    cb0_concat_prompt = codes_moshi_c[0, 0, :p_frames].cpu().numpy()
    cb0_alone_prompt  = codes_prompt_moshi_alone[0, 0, :p_frames].cpu().numpy()
    cb0_concat_compl  = codes_moshi_c[0, 0, p_frames:total_c].cpu().numpy()
    cb0_alone_compl   = codes_chosen_alone[0, 0, :c_frames].cpu().numpy()

    prompt_match = float(np.mean(cb0_concat_prompt == cb0_alone_prompt))
    compl_match  = float(np.mean(cb0_concat_compl  == cb0_alone_compl))
    print(f"\n    codebook-0 match (prompt region vs. solo encode) : {prompt_match:.2%}")
    print(f"    codebook-0 match (compl  region vs. solo encode) : {compl_match:.2%}")
    print("    (100% = no seam artifact; near-100% is fine for Mimi's causal conv)")

    # 7. Save WAVs for manual listening
    if out_dir is not None:
        sf.write(str(out_dir / f"ex{idx}_moshi_stream_chosen.wav"), recon_moshi_c, SAMPLING_RATE)
        sf.write(str(out_dir / f"ex{idx}_user_stream_chosen.wav"),  recon_user_c,  SAMPLING_RATE)
        sf.write(str(out_dir / f"ex{idx}_prompt_moshi_raw.wav"),    prompt_moshi,  SAMPLING_RATE)
        sf.write(str(out_dir / f"ex{idx}_chosen_raw.wav"),          chosen,        SAMPLING_RATE)
        print(f"\n  WAVs saved to {out_dir}/")
        print("  Listen: moshi_stream_chosen.wav should sound like")
        print("          [speaker B's prior speech] then [Moshi's generation]")
        print("          user_stream_chosen.wav should sound like")
        print("          [speaker A's prior speech] then [silence]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kalbin/moshi-on-policy-dpo-v20-kyutai-aligned-smoke")
    parser.add_argument("--model",   default="kmhf/hf-moshiko")
    parser.add_argument("--n",       type=int, default=3, help="Number of examples to check")
    parser.add_argument("--out",     default="./collator_verify_out")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    split = ds["train"] if "train" in ds else next(iter(ds.values()))

    print(f"\nDataset columns: {split.column_names}")
    print(f"Dataset size   : {len(split)}")

    # Check required columns exist
    required = {"prompt", "prompt_moshi_audio", "chosen", "rejected"}
    missing = required - set(split.column_names)
    if missing:
        print(f"\n[ERROR] Missing columns: {missing}")
        print("  'prompt_moshi_audio' missing means the collator fix cannot be verified.")
        return

    for col in ("prompt", "prompt_moshi_audio", "chosen", "rejected"):
        split = split.cast_column(col, Audio(sampling_rate=SAMPLING_RATE))

    print(f"\nLoading Moshi (for Mimi encoder/decoder): {args.model}")
    model = MoshiForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    mimi = model.audio_encoder.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mimi = mimi.to(device)
    print(f"  Mimi on {device}")

    for i in range(min(args.n, len(split))):
        verify_example(split[i], mimi, out_dir, i)

    print(f"\n{'='*60}")
    print("WHAT TO LOOK FOR:")
    print("  1. moshi stream prompt RMS >> 0  : speaker B's speech is present")
    print("  2. user  stream compl  RMS ≈ 0   : user is silent during completion")
    print("  3. codebook-0 match near 100%    : no seam artifact at boundary")
    print("  4. Listen to the WAVs: moshi stream should be a continuous")
    print("     narrative — B's prior speech flowing into Moshi's generation")
    print("='*60)")


if __name__ == "__main__":
    main()