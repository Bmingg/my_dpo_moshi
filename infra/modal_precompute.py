"""
Modal app for precomputing Moshi DPO ref log-probs, sharded across GPUs.

Usage:
  modal run --detach infra/modal_precompute.py
  NUM_SHARDS=8 PRECOMPUTE_BATCH_SIZE=8 modal run --detach infra/modal_precompute.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "moshi-dpo-precompute"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"

DEFAULT_GPU = "H200"
DEFAULT_CPU = 4.0
DEFAULT_MEMORY_MB = 81920
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 12   # 12h per shard

NUM_SHARDS = int(os.environ.get("NUM_SHARDS", 4))

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

volume = modal.Volume.from_name("moshi-dpo-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    if not modal.is_local():
        return []
    gitignore_path = _PROJECT_ROOT / ".gitignore"
    if not gitignore_path.is_file():
        return []
    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


def _run_subprocess_with_periodic_volume_commits(cmd: list[str], extra_env: dict) -> None:
    full_env = {**os.environ, **extra_env}
    proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, env=full_env)
    returncode: int | None = None
    try:
        while returncode is None:
            try:
                returncode = proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                volume.commit()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        volume.commit()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


# ============ image (same as training) ============
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .run_commands("pip install torch --index-url https://download.pytorch.org/whl/cu130")
    .run_commands(
        "pip install transformers accelerate datasets sentencepiece safetensors "
        "omegaconf tensorboard tqdm huggingface_hub bitsandbytes wandb torchcodec"
    )
)

NETRC_PATH = Path("~/.netrc").expanduser()
if NETRC_PATH.is_file():
    image = image.add_local_file(NETRC_PATH, remote_path="/root/.netrc", copy=True)

image = image.add_local_dir(
    str(_PROJECT_ROOT), remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)

app = modal.App(APP_NAME)

base_env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    "HF_HOME": f"{VOLUME_PATH}/hf",
    "HF_DATASETS_CACHE": f"{VOLUME_PATH}/hf/datasets",
    "TRANSFORMERS_CACHE": f"{VOLUME_PATH}/hf/transformers",
}

PRECOMPUTE_FORWARD_VARS = [
    "MAX_PROMPT_SEC",
    "MAX_COMPLETION_SEC",
    "PRECOMPUTE_BATCH_SIZE",
    "MOSHI_REPO",
    "MOSHI_DPO_SEMANTIC_WEIGHT",
    "MOSHI_DPO_ACOUSTIC_WEIGHT",
    "MOSHI_DPO_TEXT_WEIGHT",
    "MOSHI_DPO_USE_TEXT_ALIGNMENT",
    "CHECKPOINT_EVERY_N_BATCHES",
]
for var in PRECOMPUTE_FORWARD_VARS:
    if os.environ.get(var):
        base_env[var] = os.environ[var]

function_secrets = []
for secret_var in ("WANDB_API_KEY", "HF_TOKEN"):
    if os.environ.get(secret_var):
        function_secrets.append(modal.Secret.from_dict({secret_var: os.environ[secret_var]}))


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,         # ONE GPU per shard — but many shards run in parallel
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def precompute_shard(shard_idx: int, num_shards: int) -> None:
    """One shard. Modal runs many of these in parallel, each on its own H200."""
    shard_env = {
        **base_env,
        "SHARD_IDX": str(shard_idx),
        "NUM_SHARDS": str(num_shards),
        "REF_LOGPS_PATH": f"{VOLUME_PATH}/ref_logps_shard{shard_idx}of{num_shards}.pt",
    }
    cmd = ["python", "-u", f"{PROJECT_DIR}/precompute_ref_logps.py"]
    print(f"[modal] shard {shard_idx}/{num_shards} starting")
    _run_subprocess_with_periodic_volume_commits(cmd, shard_env)
    print(f"[modal] shard {shard_idx}/{num_shards} done")


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=600,
    image=image,
    cpu=2.0,
    memory=8192,
    # No GPU — just file ops.
)
def combine_shards(num_shards: int) -> None:
    """Concatenate per-shard files into the final /vol/ref_logps.pt."""
    import torch

    combined = {}
    metadata = None

    for shard_idx in range(num_shards):
        path = f"{VOLUME_PATH}/ref_logps_shard{shard_idx}of{num_shards}.pt"
        print(f"[combine] loading {path}")
        shard = torch.load(path)

        # Keep metadata once, but do not treat it as a dataset split.
        if "_metadata" in shard:
            if metadata is None:
                metadata = shard["_metadata"]
            else:
                # Optional sanity check: all shards should have same metadata.
                assert shard["_metadata"] == metadata, (
                    f"Metadata mismatch in shard {shard_idx}"
                )

        for split_name, data in shard.items():
            if split_name == "_metadata":
                continue

            if split_name not in combined:
                combined[split_name] = {"chosen": [], "rejected": []}

            combined[split_name]["chosen"].extend(data["chosen"])
            combined[split_name]["rejected"].extend(data["rejected"])

    if metadata is not None:
        combined["_metadata"] = metadata

    out_path = f"{VOLUME_PATH}/ref_logps.pt"
    torch.save(combined, out_path)

    for split_name, data in combined.items():
        if split_name == "_metadata":
            print(f"[combine] metadata: {data}")
            continue
        print(
            f"[combine] {split_name}: "
            f"{len(data['chosen'])} chosen, {len(data['rejected'])} rejected"
        )

    print(f"[combine] saved {out_path}")
    volume.commit()


@app.local_entrypoint()
def main() -> None:
    print(f"[modal] launching {NUM_SHARDS} parallel shards on {DEFAULT_GPU}")
    list(precompute_shard.starmap([(i, NUM_SHARDS) for i in range(NUM_SHARDS)]))
    print("[modal] all shards done; combining…")
    combine_shards.remote(NUM_SHARDS)
    print("[modal] all done.")