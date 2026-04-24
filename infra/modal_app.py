"""
Modal app for Moshi DPO training on H100.

Usage:
  # Full run (detached — keeps running after you close terminal)
  modal run --detach infra/modal_app.py

  # With custom args forwarded to run_moshi_dpo.py
  modal run --detach infra/modal_app.py --max-steps 500

  # Interactive (logs stream to terminal, dies if you disconnect)
  modal run infra/modal_app.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

# ========================== config ==========================

APP_NAME = "moshi-dpo"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"

DEFAULT_GPU = "H200"
DEFAULT_CPU = 4.0
DEFAULT_MEMORY_MB = 81920           # 80 GB — Moshi 7B + LoRA needs headroom
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24  # 24h

# Volume commits every 5 min so you don't lose checkpoints on crash.
DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS = 300

# Resolve project root from this file's location (parent of infra/).
# This way `modal run` works regardless of which directory you're in.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

volume = modal.Volume.from_name("moshi-dpo-volume", create_if_missing=True)

# ========================== .gitignore handling ==========================

def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""
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


# ========================== path helpers ==========================
# (No path rewriting needed — output dir is set via env var directly.)


# ========================== subprocess runner ==========================

def _run_subprocess_with_periodic_volume_commits(cmd: list[str]) -> None:
    """Run a command, committing the volume every 5 min for crash safety."""
    proc = subprocess.Popen(cmd, cwd=PROJECT_DIR)
    returncode: int | None = None
    try:
        while returncode is None:
            try:
                returncode = proc.wait(
                    timeout=DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS
                )
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


# ========================== image (container env) ==========================

# Start with slim Debian + Python 3.12, install system deps.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
)

# Install CUDA-enabled PyTorch (unpinned — let pip resolve latest compatible version).
image = image.run_commands(
    "pip install torch --index-url https://download.pytorch.org/whl/cu130"
)

# Install all Python dependencies for Moshi DPO.
image = image.run_commands(
    "pip install "
    "transformers "
    "accelerate "
    "datasets "
    "sentencepiece "
    "safetensors "
    "omegaconf "
    "tensorboard "
    "tqdm "
    "huggingface_hub "
    "bitsandbytes "       # 8-bit Adam, fallback if switching to H100
    "wandb "              # real-time metric dashboard
    "torchcodec "         # audio decoding for HF datasets
)

# Copy HF credentials so we can push/pull private repos.
# Must come BEFORE add_local_dir (which must be last — it's a lazy mount).
NETRC_PATH = Path("~/.netrc").expanduser()
if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )

# Copy project code into the container — MUST BE LAST in the image build.
image = image.add_local_dir(
    str(_PROJECT_ROOT),
    remote_path=PROJECT_DIR,
    ignore=load_gitignore_patterns(),
)

# ========================== app ==========================

app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    "HF_HOME": f"{VOLUME_PATH}/hf",
    "HF_DATASETS_CACHE": f"{VOLUME_PATH}/hf/datasets",
    "TRANSFORMERS_CACHE": f"{VOLUME_PATH}/hf/transformers",
    "MOSHI_DPO_OUTPUT_DIR": f"{VOLUME_PATH}/moshi-dpo-checkpoints",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
}

# Forward any MOSHI_DPO_* env vars from local machine → Modal container.
# This lets you sweep hyperparams without editing code:
#   MOSHI_DPO_LR=5e-7 MOSHI_DPO_BETA=0.2 modal run --detach infra/modal_app.py
SWEEP_ENV_VARS = [
    "MOSHI_DPO_LR", "MOSHI_DPO_BETA", "MOSHI_DPO_SEMANTIC_WEIGHT",
    "MOSHI_DPO_EPOCHS", "MOSHI_DPO_MAX_STEPS", "MOSHI_DPO_BATCH_SIZE",
    "MOSHI_DPO_GRAD_ACCUM", "MOSHI_DPO_TRAIN_TAKE", "MOSHI_DPO_VAL_TAKE",
    "MOSHI_DPO_OUTPUT_DIR", "MOSHI_DPO_REPORT_TO",
]
for var in SWEEP_ENV_VARS:
    if os.environ.get(var):
        env[var] = os.environ[var]

# Pass W&B key and HF token if available locally.
function_secrets = []
for secret_var in ("WANDB_API_KEY", "HF_TOKEN"):
    if os.environ.get(secret_var):
        function_secrets.append(
            modal.Secret.from_dict({secret_var: os.environ[secret_var]})
        )
    env["WANDB_DIR"] = f"{VOLUME_PATH}/wandb"


# ========================== training function ==========================

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def train_remote() -> None:
    """Run Moshi DPO training on a remote H100."""
    cmd = [
        "python", "-u",
        f"{PROJECT_DIR}/run_dpo_moshi.py",
    ]

    print(f"[modal] Running: {' '.join(cmd)}")
    print(f"[modal] Checkpoints → {VOLUME_PATH}/moshi-dpo-checkpoints")
    _run_subprocess_with_periodic_volume_commits(cmd)


# ========================== entrypoint ==========================

@app.local_entrypoint()
def main() -> None:
    """Default entrypoint: launch training on Modal."""
    print(f"[modal] Launching {APP_NAME} on {DEFAULT_GPU}...")
    print(f"[modal] Volume: moshi-dpo-volume (mounted at {VOLUME_PATH})")
    print(f"[modal] Timeout: {DEFAULT_TIMEOUT_SECONDS // 3600}h")
    train_remote.remote()