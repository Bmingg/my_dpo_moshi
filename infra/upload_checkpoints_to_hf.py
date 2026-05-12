"""
Upload Moshi DPO checkpoints from Modal volume → HuggingFace Hub.

Usage:
  # Upload one specific checkpoint
  CHECKPOINT_DIR=lr1e-5_b1e-1 CHECKPOINT_NAME=checkpoint-2220 \
    HF_REPO_PREFIX=your-username/moshi-dpo \
    modal run infra/upload_checkpoints_to_hf.py

  # Upload all checkpoints (one HF repo per run dir, named by run config)
  UPLOAD_ALL=1 HF_REPO_PREFIX=your-username/moshi-dpo \
    modal run infra/upload_checkpoints_to_hf.py
"""

from __future__ import annotations
import os
from pathlib import Path
import modal

APP_NAME = "moshi-dpo-upload"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
CHECKPOINTS_ROOT = f"{VOLUME_PATH}/moshi-dpo-checkpoints"

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

volume = modal.Volume.from_name("moshi-dpo-volume", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands("pip install torch --index-url https://download.pytorch.org/whl/cu130")
    .run_commands(
        "pip install transformers safetensors huggingface_hub hf_transfer"
    )
)

# HF token forwarding
function_secrets = []
if os.environ.get("HF_TOKEN"):
    function_secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

env = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",  # 5-10x faster uploads
    "HF_HOME": f"{VOLUME_PATH}/hf",
}
for var in ("CHECKPOINT_DIR", "CHECKPOINT_NAME", "HF_REPO_PREFIX", "UPLOAD_ALL"):
    if os.environ.get(var):
        env[var] = os.environ[var]

app = modal.App(APP_NAME)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 4,  # 4h for big uploads
    env=env,
    image=image,
    secrets=function_secrets,
    cpu=4.0,
    memory=32768,
    # No GPU — pure CPU upload work
)
def upload_checkpoint() -> None:
    import torch
    from transformers import (
        MoshiForConditionalGeneration,
        AutoTokenizer,
    )
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set in environment")
    login(token=token)

    repo_prefix = os.environ.get("HF_REPO_PREFIX")
    if not repo_prefix:
        raise RuntimeError("HF_REPO_PREFIX not set (e.g. 'your-username/moshi-dpo')")

    upload_all = os.environ.get("UPLOAD_ALL") == "1"

    if upload_all:
        run_dirs = sorted([
            d for d in Path(CHECKPOINTS_ROOT).iterdir() if d.is_dir()
        ])
        print(f"[upload] Found {len(run_dirs)} run directories")
    else:
        run_dir_names = os.environ.get("CHECKPOINT_DIR")
        if not run_dir_names:
            raise RuntimeError("CHECKPOINT_DIR not set (e.g. 'lr1e-5_b1e-1' or 'lr1e-5_b1e-1,lr2e-6_b1e-1')")
        # Support comma-separated list
        run_dirs = [Path(CHECKPOINTS_ROOT) / name.strip() for name in run_dir_names.split(",")]
        print(f"[upload] Will process {len(run_dirs)} run directories")

    ckpt_name = os.environ.get("CHECKPOINT_NAME", "checkpoint-2220")

    api = HfApi()

    for run_dir in run_dirs:
        ckpt_path = run_dir / ckpt_name
        if not ckpt_path.exists():
            # Fallback: pick the highest-numbered checkpoint
            available = sorted(
                [d for d in run_dir.iterdir() if d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            if not available:
                print(f"[upload] No checkpoints in {run_dir}, skipping")
                continue
            ckpt_path = available[-1]
            print(f"[upload] {ckpt_name} not found, using {ckpt_path.name}")

        repo_name = f"{repo_prefix}-{run_dir.name}"  # e.g. user/moshi-dpo-lr1e-5_b1e-1
        print(f"[upload] {ckpt_path}  →  {repo_name}")

        # Sanity check the checkpoint loads correctly before pushing
        print(f"[upload] Verifying checkpoint loads...")
        model = MoshiForConditionalGeneration.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.bfloat16,
        )
        print(f"[upload] Loaded successfully, {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

        # Create repo (private by default — flip if you want public)
        api.create_repo(repo_id=repo_name, exist_ok=True, private=True)

        # Push the model
        model.push_to_hub(
            repo_id=repo_name,
            commit_message=f"DPO checkpoint {ckpt_path.name} from {run_dir.name}",
        )

        # Also push the tokenizer (downstream eval will need it)
        tokenizer = AutoTokenizer.from_pretrained("kmhf/hf-moshiko")
        tokenizer.push_to_hub(repo_id=repo_name)

        # Free memory before next iteration
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"Done with {repo_name}")


@app.local_entrypoint()
def main() -> None:
    upload_checkpoint.remote()