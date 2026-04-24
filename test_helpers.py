"""
Unit tests for the module-level helpers in moshi_dpo_trainer.py.

Runs on CPU in under a second. Tests the tricky bits (delay pattern,
gather logic, completion masking) against hand-constructed toy tensors
where we can verify the expected output by eye.

Usage:
    python test_helpers.py

Passes: all assertions succeed, prints "All tests passed."
Fails:  raises AssertionError with a descriptive message.

If a test fails, the expected and actual values are printed so you can
see exactly what went wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure we can import the trainer.
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from moshi_dpo_trainer import (
    _apply_delay_pattern,
    _gather_text_logp,
    _gather_audio_logp,
    _build_audio_completion_mask,
)


# ======================================================================
# _apply_delay_pattern
# ======================================================================

def test_delay_pattern_basic():
    """Codebook 0 unchanged, codebooks 1+ shifted right by 1 with BOS at pos 0."""
    # Shape: [B=1, K=3, T=4]. Codes are distinct integers for readability.
    # Codebook 0: [10, 11, 12, 13]
    # Codebook 1: [20, 21, 22, 23]
    # Codebook 2: [30, 31, 32, 33]
    codes = torch.tensor([[
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ]])
    BOS = 999

    out = _apply_delay_pattern(codes, bos_token_id=BOS)
    expected = torch.tensor([[
        [10, 11, 12, 13],     # codebook 0: unchanged
        [999, 20, 21, 22],    # codebook 1: BOS at pos 0, rest shifted right
        [999, 30, 31, 32],    # codebook 2: BOS at pos 0, rest shifted right
    ]])
    assert torch.equal(out, expected), (
        f"_apply_delay_pattern basic case mismatch\n"
        f"expected:\n{expected}\n"
        f"got:\n{out}"
    )
    print("✓ test_delay_pattern_basic")


def test_delay_pattern_batch():
    """Same shift applied independently to each batch element."""
    codes = torch.tensor([
        [[1, 2, 3], [10, 20, 30]],    # example 0
        [[4, 5, 6], [40, 50, 60]],    # example 1
    ])
    BOS = 99

    out = _apply_delay_pattern(codes, bos_token_id=BOS)
    expected = torch.tensor([
        [[1, 2, 3], [99, 10, 20]],
        [[4, 5, 6], [99, 40, 50]],
    ])
    assert torch.equal(out, expected), f"Batch case failed:\n{out}"
    print("✓ test_delay_pattern_batch")


def test_delay_pattern_single_frame():
    """Edge case: T=1. Codebooks 1+ become just BOS with nothing to shift in."""
    codes = torch.tensor([[[5], [15], [25]]])   # B=1, K=3, T=1
    BOS = 7
    out = _apply_delay_pattern(codes, bos_token_id=BOS)
    expected = torch.tensor([[[5], [7], [7]]])
    assert torch.equal(out, expected), f"T=1 case failed:\n{out}"
    print("✓ test_delay_pattern_single_frame")


def test_delay_pattern_preserves_shape():
    """Output shape == input shape."""
    codes = torch.randint(0, 100, (2, 8, 15))
    out = _apply_delay_pattern(codes, bos_token_id=999)
    assert out.shape == codes.shape
    print("✓ test_delay_pattern_preserves_shape")


# ======================================================================
# _gather_text_logp
# ======================================================================

def test_gather_text_logp_shape():
    """Output shape is [B, T-1]."""
    B, T, V = 2, 10, 32000
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    out = _gather_text_logp(logits, targets)
    assert out.shape == (B, T - 1), f"Expected {(B, T-1)}, got {out.shape}"
    print("✓ test_gather_text_logp_shape")


def test_gather_text_logp_values():
    """Verify the gather returns log p(target[t+1] | logits[t]).

    Causal LM convention: logits[:, t, :] predicts token t+1.
    So the returned log-prob at position i should equal
    log_softmax(logits[:, i, :])[targets[:, i+1]].
    """
    B, T, V = 1, 4, 5
    torch.manual_seed(42)
    logits = torch.randn(B, T, V)
    targets = torch.tensor([[1, 3, 0, 2]])   # [B=1, T=4]

    out = _gather_text_logp(logits, targets)   # [B=1, T-1=3]

    # Manual check at position 0: should equal log_softmax(logits[0, 0])[targets[0, 1]]
    # = log_softmax(logits[0, 0])[3]
    expected_pos0 = F.log_softmax(logits[0, 0], dim=-1)[3].item()
    assert abs(out[0, 0].item() - expected_pos0) < 1e-6, (
        f"Position 0: expected {expected_pos0}, got {out[0, 0].item()}"
    )

    # Position 1: log_softmax(logits[0, 1])[targets[0, 2]] = log_softmax(logits[0, 1])[0]
    expected_pos1 = F.log_softmax(logits[0, 1], dim=-1)[0].item()
    assert abs(out[0, 1].item() - expected_pos1) < 1e-6, (
        f"Position 1: expected {expected_pos1}, got {out[0, 1].item()}"
    )
    print("✓ test_gather_text_logp_values")


def test_gather_text_logp_negative():
    """Log-probs must be negative (since probabilities are in (0, 1])."""
    B, T, V = 2, 5, 100
    torch.manual_seed(0)
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    out = _gather_text_logp(logits, targets)
    assert (out < 0).all(), f"Expected all log-probs negative, got max={out.max().item()}"
    print("✓ test_gather_text_logp_negative")


# ======================================================================
# _gather_audio_logp
# ======================================================================

def test_gather_audio_logp_shape():
    """Output shape is [B, T, K]."""
    B, T, K, V = 2, 10, 8, 2048
    audio_logits = torch.randn(B, T, K, V)
    delayed_codes = torch.randint(0, V, (B, K, T))
    out = _gather_audio_logp(audio_logits, delayed_codes)
    assert out.shape == (B, T, K), f"Expected {(B, T, K)}, got {out.shape}"
    print("✓ test_gather_audio_logp_shape")


def test_gather_audio_logp_values():
    """Verify the gather uses delayed_codes as index.

    Output at [b, t, k] should be log_softmax(audio_logits[b, t, k, :])[delayed_codes[b, k, t]].
    Note the k/t transpose between audio_logits ([B, T, K, V]) and delayed_codes ([B, K, T]).
    """
    B, T, K, V = 1, 3, 2, 5
    torch.manual_seed(7)
    audio_logits = torch.randn(B, T, K, V)
    # delayed_codes is [B, K, T]
    delayed_codes = torch.tensor([[[1, 2, 0], [3, 4, 1]]])   # [B=1, K=2, T=3]

    out = _gather_audio_logp(audio_logits, delayed_codes)   # [B=1, T=3, K=2]

    # Manual check at (t=0, k=0): log_softmax(audio_logits[0, 0, 0, :])[delayed_codes[0, 0, 0]]
    # = log_softmax(audio_logits[0, 0, 0, :])[1]
    expected = F.log_softmax(audio_logits[0, 0, 0], dim=-1)[1].item()
    assert abs(out[0, 0, 0].item() - expected) < 1e-6, (
        f"(t=0, k=0): expected {expected}, got {out[0, 0, 0].item()}"
    )

    # At (t=2, k=1): log_softmax(audio_logits[0, 2, 1, :])[delayed_codes[0, 1, 2]]
    # = log_softmax(audio_logits[0, 2, 1, :])[1]
    expected = F.log_softmax(audio_logits[0, 2, 1], dim=-1)[1].item()
    assert abs(out[0, 2, 1].item() - expected) < 1e-6, (
        f"(t=2, k=1): expected {expected}, got {out[0, 2, 1].item()}"
    )
    print("✓ test_gather_audio_logp_values")


# ======================================================================
# _build_audio_completion_mask
# ======================================================================

def test_audio_mask_codebook_0_no_shift():
    """Codebook 0's mask equals the raw completion_mask at every position."""
    # B=1, T=5, completion spans frames 2-4.
    completion_mask = torch.tensor([[False, False, True, True, True]])
    K = 3

    out = _build_audio_completion_mask(completion_mask, num_codebooks=K)   # [B, T, K] float

    # Codebook 0 should match completion_mask directly.
    expected_k0 = torch.tensor([[0., 0., 1., 1., 1.]])
    assert torch.allclose(out[:, :, 0], expected_k0), (
        f"Codebook 0 mask wrong:\nexpected: {expected_k0}\ngot: {out[:, :, 0]}"
    )
    print("✓ test_audio_mask_codebook_0_no_shift")


def test_audio_mask_higher_codebooks_shifted():
    """Codebooks 1+ shifted right by 1 (scoring previous actual-time frame)."""
    completion_mask = torch.tensor([[False, False, True, True, True]])
    K = 3

    out = _build_audio_completion_mask(completion_mask, num_codebooks=K)

    # Codebooks 1 and 2: right-shifted by 1. So completion starts at index 3 instead of 2.
    # Position 0: 0 (never contributes — BOS sentinel lives here)
    # Position 1: completion_mask[0] = 0
    # Position 2: completion_mask[1] = 0
    # Position 3: completion_mask[2] = 1  (first completion frame's q>0 code
    #                                       is predicted at model-step 3)
    # Position 4: completion_mask[3] = 1
    expected_k1 = torch.tensor([[0., 0., 0., 1., 1.]])
    assert torch.allclose(out[:, :, 1], expected_k1), (
        f"Codebook 1 mask wrong:\nexpected: {expected_k1}\ngot: {out[:, :, 1]}"
    )
    assert torch.allclose(out[:, :, 2], expected_k1), (
        f"Codebook 2 mask wrong:\nexpected: {expected_k1}\ngot: {out[:, :, 2]}"
    )
    print("✓ test_audio_mask_higher_codebooks_shifted")


def test_audio_mask_excludes_bos_position():
    """Position 0 for codebooks 1+ is always False (holds BOS, not a real code)."""
    # Even if completion starts at frame 0, codebook q>0 at position 0 is BOS.
    completion_mask = torch.tensor([[True, True, True]])
    out = _build_audio_completion_mask(completion_mask, num_codebooks=3)

    # Codebook 0 at pos 0: True (real frame 0 semantic code)
    assert out[0, 0, 0] == 1.0, "Codebook 0 pos 0 should be 1 when frame 0 is completion"

    # Codebook 1 at pos 0: False (BOS placeholder)
    assert out[0, 0, 1] == 0.0, "Codebook 1 pos 0 should be 0 (BOS)"
    assert out[0, 0, 2] == 0.0, "Codebook 2 pos 0 should be 0 (BOS)"

    # Codebook 1 at pos 1: True (shifted completion_mask[0] = True)
    assert out[0, 1, 1] == 1.0, "Codebook 1 pos 1 should be 1"
    print("✓ test_audio_mask_excludes_bos_position")


def test_audio_mask_t_equals_1():
    """Edge case: T=1. Higher codebooks should be all zero (nothing to shift in)."""
    completion_mask = torch.tensor([[True]])
    out = _build_audio_completion_mask(completion_mask, num_codebooks=3)
    assert out[0, 0, 0] == 1.0, "Codebook 0 pos 0 should be 1"
    assert out[0, 0, 1] == 0.0, "Codebook 1 pos 0 should be 0 (BOS placeholder, T=1)"
    assert out[0, 0, 2] == 0.0, "Codebook 2 pos 0 should be 0 (BOS placeholder, T=1)"
    print("✓ test_audio_mask_t_equals_1")


# ======================================================================
# Integration: helpers work together on a realistic toy
# ======================================================================

def test_integration_toy_example():
    """End-to-end: simulate a tiny 1-example, 5-frame batch and verify all
    helpers compose sensibly.

    Scenario: 5 frames, prompt = frames 0-1, completion = frames 2-4. K=3.
    """
    B, T, K = 1, 5, 3
    V_audio = 10
    torch.manual_seed(123)

    # Fake codes and logits.
    moshi_codes = torch.randint(0, V_audio, (B, K, T))
    audio_logits = torch.randn(B, T, K, V_audio)

    completion_mask = torch.tensor([[False, False, True, True, True]])

    # Apply delay pattern.
    BOS = V_audio  # sentinel outside normal vocab
    delayed = _apply_delay_pattern(moshi_codes, bos_token_id=BOS)
    assert delayed.shape == (B, K, T)
    # Codebook 0 unchanged.
    assert torch.equal(delayed[:, 0, :], moshi_codes[:, 0, :])
    # Codebooks 1+ have BOS at position 0.
    assert (delayed[:, 1:, 0] == BOS).all()
    # Codebooks 1+ at positions 1..T-1 equal original at 0..T-2.
    assert torch.equal(delayed[:, 1:, 1:], moshi_codes[:, 1:, :-1])

    # Gather audio log-probs. But NOTE: audio_logits vocab is [0, V_audio-1],
    # and delayed now contains BOS=V_audio at some positions, which is
    # out-of-range for gather. That's intentional — the mask will zero out
    # those positions.
    # To make the gather not error, we clamp delayed labels into the valid
    # range (they'll be masked out anyway).
    delayed_safe = delayed.clamp(0, V_audio - 1)

    audio_logp = _gather_audio_logp(audio_logits, delayed_safe)   # [B, T, K]
    assert audio_logp.shape == (B, T, K)

    # Build mask and apply.
    audio_mask = _build_audio_completion_mask(completion_mask, num_codebooks=K)
    masked_logp = audio_logp * audio_mask

    # Confirm: positions that should be zero (prompt + BOS) are zero.
    # Position (0, k=0): prompt (completion_mask[0] = False) → mask = 0
    assert masked_logp[0, 0, 0] == 0.0
    # Position (0, k>0): BOS → mask = 0
    assert masked_logp[0, 0, 1] == 0.0
    # Position (2, k=0): completion frame → mask = 1, masked_logp = audio_logp
    assert masked_logp[0, 2, 0] == audio_logp[0, 2, 0]
    # Position (3, k=1): shifted completion → mask = 1
    assert masked_logp[0, 3, 1] == audio_logp[0, 3, 1]

    # Sum should be finite and negative (real log-probs of actual events).
    total = masked_logp.sum()
    assert torch.isfinite(total)
    # Note: with random logits it might not be negative, but with properly
    # trained logits it should be. Here we just check finiteness.

    print("✓ test_integration_toy_example")


# ======================================================================
# Run everything.
# ======================================================================

def run_all_tests():
    test_fns = [
        # _apply_delay_pattern
        test_delay_pattern_basic,
        test_delay_pattern_batch,
        test_delay_pattern_single_frame,
        test_delay_pattern_preserves_shape,
        # _gather_text_logp
        test_gather_text_logp_shape,
        test_gather_text_logp_values,
        test_gather_text_logp_negative,
        # _gather_audio_logp
        test_gather_audio_logp_shape,
        test_gather_audio_logp_values,
        # _build_audio_completion_mask
        test_audio_mask_codebook_0_no_shift,
        test_audio_mask_higher_codebooks_shifted,
        test_audio_mask_excludes_bos_position,
        test_audio_mask_t_equals_1,
        # Integration
        test_integration_toy_example,
    ]

    failures = 0
    for fn in test_fns:
        try:
            fn()
        except AssertionError as e:
            print(f"✗ {fn.__name__} FAILED: {e}")
            failures += 1
        except Exception as e:
            print(f"✗ {fn.__name__} ERRORED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failures += 1

    print()
    if failures == 0:
        print(f"All {len(test_fns)} tests passed.")
        return 0
    else:
        print(f"{failures}/{len(test_fns)} tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())