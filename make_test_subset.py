"""
Create a tiny SpokenSwag subset and push to HF for pipeline testing.

Usage:
  python make_test_subset.py --hf_repo YOUR_USERNAME/SpokenSwag-test-10
"""

import argparse
from datasets import load_dataset, DatasetDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_val", type=int, default=5)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    print(f"Loading slprl/SpokenSwag...")
    ds = load_dataset("slprl/SpokenSwag")

    subset = DatasetDict({
        "train": ds["train"].select(range(args.n_train)),
        "validation": ds["validation"].select(range(args.n_val)),
    })

    for split, d in subset.items():
        print(f"  {split}: {len(d)} examples")

    print(f"Pushing to {args.hf_repo}...")
    subset.push_to_hub(args.hf_repo, private=args.private)
    print(f"Done! https://huggingface.co/datasets/{args.hf_repo}")

if __name__ == "__main__":
    main()