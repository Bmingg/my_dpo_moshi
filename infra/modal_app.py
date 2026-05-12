"""
Modal app for Moshi DPO training on H100.

Usage:
  # Full run (detached — keeps running after you close terminal)
  modal run --detach infra/modal_app.py

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

# GPU configuration — overridable via env vars at launch time.
#   MODAL_NUM_GPUS=2 modal run --detach infra/modal_app.py
#   MODAL_GPU_TYPE=B200 MODAL_NUM_GPUS=4 modal run --detach infra/modal_app.py
MODAL_GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "H200")
NPROC_PER_NODE = int(os.environ.get("MODAL_NUM_GPUS", "4"))

DEFAULT_GPU = f"{MODAL_GPU_TYPE}:{NPROC_PER_NODE}" if NPROC_PER_NODE > 1 else MODAL_GPU_TYPE
DEFAULT_CPU = 4.0
DEFAULT_MEMORY_MB = 81920
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24 # 24 hours

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
    "peft "
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
    "MAX_PROMPT_SEC", "MAX_COMPLETION_SEC", "WANDB_PROJECT", "WANDB_NAME",
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
    """Run Moshi DPO training with torchrun (DDP if NPROC_PER_NODE > 1)."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={NPROC_PER_NODE}",
        "--standalone",
        f"{PROJECT_DIR}/run_dpo_moshi.py",
    ]
    print(f"[modal] Running: {' '.join(cmd)}")
    print(f"[modal] Checkpoints → {VOLUME_PATH}/moshi-dpo-checkpoints")
    _run_subprocess_with_periodic_volume_commits(cmd)

@app.function(
    image=image,
    cpu=4.0,
    memory=24576,    # 24 GB system RAM, no GPU
    timeout=600,
)
def inspect_modules():
    """One-off: dump checkpointed module names. No GPU, no training."""
    import torch
    from transformers import MoshiForConditionalGeneration
    
    model = MoshiForConditionalGeneration.from_pretrained(
        "kmhf/hf-moshiko",
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    print("=== Checkpointed modules ===")
    ckpt = [(n, type(m).__name__) for n, m in model.named_modules()
            if getattr(m, "gradient_checkpointing", False)]
    print(f"Total: {len(ckpt)}")
    for name, cls in ckpt[:30]:
        print(f"  {name}  ({cls})")
    if len(ckpt) > 30:
        print(f"  ... and {len(ckpt) - 30} more")
    
    print("\n=== Depth-related modules ===")
    for n, m in model.named_modules():
        if "depth" in n.lower() or "depformer" in n.lower():
            flag = getattr(m, "gradient_checkpointing", "N/A")
            print(f"  {n}  ({type(m).__name__})  ckpt={flag}")
    
    print("\n=== Top-level structure ===")
    print(type(model).__name__)
    for name, child in model.named_children():
        print(f"  {name}: {type(child).__name__}")

# ========================== entrypoint ==========================

@app.local_entrypoint()
def main() -> None:
    """Default entrypoint: launch training on Modal."""
    print(f"[modal] Launching {APP_NAME}")
    print(f"[modal]   GPU spec: {DEFAULT_GPU}  ({NPROC_PER_NODE} GPU{'s' if NPROC_PER_NODE > 1 else ''})")
    print(f"[modal]   Volume: moshi-dpo-volume (mounted at {VOLUME_PATH})")
    print(f"[modal]   Timeout: {DEFAULT_TIMEOUT_SECONDS // 3600}h")
    train_remote.remote()