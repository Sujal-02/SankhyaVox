"""
Release a versioned model checkpoint with DVC + Git.

Usage:
    python scripts/release.py 1.0 "Initial HMM with per-token states" --accuracy 72.5
    python scripts/release.py 1.0 "Initial HMM" --accuracy 72.5 --dry-run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run(cmd: str, *, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY RUN] {cmd}")
        return
    print(f">> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}): {cmd}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Release a versioned model checkpoint.")
    parser.add_argument("version", help="Version in x.y format (e.g. 1.0, 2.3)")
    parser.add_argument("message", help="Short description of what changed")
    parser.add_argument("--accuracy", type=float, required=True, help="Model accuracy %% (e.g. 72.5)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without executing")
    args = parser.parse_args()

    # Validate version format
    if not re.fullmatch(r"\d+\.\d+", args.version):
        print(f"Error: Version must be in x.y format (e.g. 1.0, 2.3). Got: {args.version}", file=sys.stderr)
        sys.exit(1)

    tag = f"v{args.version}"

    # Check tag doesn't already exist
    existing = subprocess.run(
        f"git tag -l {tag}", shell=True, capture_output=True, text=True
    ).stdout.strip()
    if existing:
        print(f"Error: Tag '{tag}' already exists. Bump the version.", file=sys.stderr)
        sys.exit(1)

    # Check required artifacts exist
    x, y = args.version.split(".")
    ckpt = Path("checkpoints") / f"hmm_classifier_{x}_{y}.pkl"
    notebook = Path("results/hmm_evaluations") / f"hmm_{x}_{y}.ipynb"

    missing = []
    if not ckpt.exists():
        missing.append(str(ckpt))
    if not notebook.exists():
        missing.append(str(notebook))
    if missing:
        print(f"Error: Required artifacts missing:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    commit_msg = f"HMM {tag} (acc={args.accuracy:.2f}%): {args.message}"

    # Execute release steps
    steps = [
        "dvc add checkpoints",
        "git add -u",
        "git add checkpoints.dvc",
        "git add results/hmm_evaluations/",
        f'git commit -m "{commit_msg}"',
        f"git tag {tag}",
        "dvc push",
        "git push --tags",
        "git push",
    ]

    for cmd in steps:
        run(cmd, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\nReleased {tag} successfully.")


if __name__ == "__main__":
    main()
