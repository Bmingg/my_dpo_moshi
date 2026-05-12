"""
Modal app for preprocessing SpokenSwag with frame-aligned text token IDs.

Wraps preprocess_spoken_swag.py — runs Whisper alignment on GPU with
checkpoint-based resume on a persistent Modal volume.

Usage:
  # Smoke test (10 train + 10 val examples)
  PREPROCESS_TRAIN_TAKE=10 PREPROCESS_VAL_TAKE=10 \
    modal run --detach infra/modal_preprocess_align.py

  # Full run with resume from existing checkpoint
  modal run --detach infra/modal_preprocess_align.py

  # First-time full run (no resume)
  PREPROCESS_NO_RESUME=1 modal run --detach infra/modal_preprocess_align.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "moshi-dpo-preprocess-align"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"

# Whisper-medium fits on a single GPU comfortably; H200 is overkill but
# the volume is already on H200-friendly infra. Using L40S/A100 saves credits.
DEFAULT_GPU = os.environ.get("MODAL_GPU_TYPE", "A100")
DEFAULT_CPU = 4.0
DEFAULT_MEMORY_MB = 32768
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24  # 24h — adjust if dataset is larger

DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS = 300

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


def _run_subprocess_with_periodic_volume_commits(cmd: list[str]) -> None:
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


# ============ image ============
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .run_commands("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130")
    .run_commands(
        "pip install transformers datasets sentencepiece safetensors "
        "huggingface_hub tqdm whisper-timestamped auditok numpy torchcodec"
    )
)

NETRC_PATH = Path("~/.netrc").expanduser()
if NETRC_PATH.is_file():
    image = image.add_local_file(NETRC_PATH, remote_path="/root/.netrc", copy=True)

image = image.add_local_dir(
    str(_PROJECT_ROOT), remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)

app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    "HF_HOME": f"{VOLUME_PATH}/hf",
    "HF_DATASETS_CACHE": f"{VOLUME_PATH}/hf/datasets",
    "TRANSFORMERS_CACHE": f"{VOLUME_PATH}/hf/transformers",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}

# Forward HF token (needed to push to your private repo)
function_secrets = []
for secret_var in ("HF_TOKEN",):
    if os.environ.get(secret_var):
        function_secrets.append(modal.Secret.from_dict({secret_var: os.environ[secret_var]}))

# Forward optional knobs from local environment
PREPROCESS_FORWARD_VARS = [
    "PREPROCESS_TRAIN_TAKE",
    "PREPROCESS_VAL_TAKE",
    "PREPROCESS_NO_RESUME",
    "PREPROCESS_WHISPER_MODEL",
    "PREPROCESS_HF_REPO",
    "PREPROCESS_PRIVATE",
]
for var in PREPROCESS_FORWARD_VARS:
    if os.environ.get(var):
        env[var] = os.environ[var]


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
def preprocess_align() -> None:
    """Run frame-aligned text preprocessing for SpokenSwag."""
    hf_repo = os.environ.get("PREPROCESS_HF_REPO", "Bmingg/SpokenSwag-aligned")
    whisper_model = os.environ.get("PREPROCESS_WHISPER_MODEL", "medium")

    cmd = [
        "python", "-u", f"{PROJECT_DIR}/preprocess_spoken_swag.py",
        "--hf_push_repo", hf_repo,
        "--checkpoint_dir", f"{VOLUME_PATH}/swag_ckpts",
        "--whisper_model", whisper_model,
    ]

    # Resume by default unless explicitly disabled
    if os.environ.get("PREPROCESS_NO_RESUME") != "1":
        cmd.append("--resume")

    if os.environ.get("PREPROCESS_PRIVATE") == "1":
        cmd.append("--private")

    if os.environ.get("PREPROCESS_TRAIN_TAKE"):
        cmd.extend(["--train_take", os.environ["PREPROCESS_TRAIN_TAKE"]])
    if os.environ.get("PREPROCESS_VAL_TAKE"):
        cmd.extend(["--val_take", os.environ["PREPROCESS_VAL_TAKE"]])

    print(f"[modal] Running: {' '.join(cmd)}")
    print(f"[modal] Checkpoint dir: {VOLUME_PATH}/swag_ckpts")
    print(f"[modal] HF push target: {hf_repo}")
    _run_subprocess_with_periodic_volume_commits(cmd)


@app.local_entrypoint()
def main() -> None:
    print(f"[modal] Launching {APP_NAME}")
    print(f"[modal]   GPU: {DEFAULT_GPU}")
    print(f"[modal]   Volume: moshi-dpo-volume")
    print(f"[modal]   Timeout: {DEFAULT_TIMEOUT_SECONDS // 3600}h")
    preprocess_align.remote()