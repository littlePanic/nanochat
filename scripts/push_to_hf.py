#!/usr/bin/env python3
"""
Utility for pushing nanochat checkpoints/reports to the Hugging Face Hub.

You can either point the script at an arbitrary directory via --model-dir, or
use --stage {base,mid,sft,report} to automatically pick the folder inside
$NANOCHAT_BASE_DIR (defaults to $HOME/.cache/nanochat) and optionally provide
--model-tag (e.g. d20). This makes it easy to upload the artifacts produced by
speedrun.sh without juggling git LFS or manual zip uploads.

Examples:
  # Upload the latest base checkpoint under $HOME/.cache/nanochat/base_checkpoints
  python push_to_hf.py --stage base --model-tag d20 --repo-id org/nanochat-base

  # Upload the generated report folder
  python push_to_hf.py --model-dir "$HOME/.cache/nanochat/report" --repo-id org/nanochat --path-in-repo report/latest
"""

from __future__ import annotations
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.utils import HfHubHTTPError, get_token as hf_get_token


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    """
    Resolve a Hugging Face token in this order:
      1) --token argument
      2) HF_TOKEN env var
      3) HUGGINGFACE_HUB_TOKEN env var
      4) cached login ($HOME/.huggingface/token)
    Returns None if nothing found (some endpoints allow anonymous, but model creation will not).
    """
    if cli_token:
        return cli_token.strip()
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token.strip()
    cached = hf_get_token()
    return cached.strip() if cached else None


def bool_from_flags(private_flag: bool, public_flag: bool) -> bool:
    """
    Determine repo visibility from mutually exclusive flags.
    Defaults to private=True if neither flag is provided (safer default).
    """
    if private_flag and public_flag:
        print("ERROR: --private and --public are mutually exclusive.", file=sys.stderr)
        sys.exit(2)
    if public_flag:
        return False
    return True  # default private


def ensure_repo(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    private: bool,
    token: Optional[str],
    exist_ok: bool = True,
) -> None:
    """
    Create the repo if it doesn't exist. If it exists, optionally adjust visibility.
    """
    # Create or ensure existence
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=exist_ok,
        token=token,
    )

    # Double-check and fix visibility if needed (create_repo won't toggle on existing)
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        is_private_now = info.private
        if bool(is_private_now) != bool(private):
            # Update visibility to match requested flag
            api.update_repo_visibility(
                repo_id=repo_id,
                private=private,
                repo_type=repo_type,
                token=token,
            )
            print(f"Adjusted visibility for {repo_id} to {'private' if private else 'public'}.")
    except HfHubHTTPError as e:
        print(f"WARNING: Could not verify/adjust visibility: {e}", file=sys.stderr)


def guess_model_tag(checkpoint_dir: Path) -> str:
    """Heuristic similar to checkpoint_manager: prefer highest dXX, fallback to newest folder."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"{checkpoint_dir} does not exist")
    tags = [p.name for p in checkpoint_dir.iterdir() if p.is_dir()]
    if not tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    candidates = []
    for tag in tags:
        match = re.match(r"d(\d+)", tag)
        if match:
            candidates.append((int(match.group(1)), tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    tags.sort(key=lambda t: (checkpoint_dir / t).stat().st_mtime, reverse=True)
    return tags[0]


def resolve_stage_dir(stage: str, base_dir: Path, model_tag: Optional[str]) -> Path:
    stage_map = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
    }
    if stage not in stage_map:
        raise ValueError(f"Unknown stage '{stage}'")
    stage_root = base_dir / stage_map[stage]
    tag = model_tag or guess_model_tag(stage_root)
    return stage_root / tag


def main():
    parser = argparse.ArgumentParser(description="Push a local model folder to Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repo ID in the form ORG_OR_USER/REPO_NAME (e.g., my-org/my-model).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Folder to upload. Overrides --stage if both are provided (default: ./checkpoints).",
    )
    parser.add_argument(
        "--stage",
        choices=["base", "mid", "sft"],
        help="Shortcut for uploading base/mid/sft checkpoints from $NANOCHAT_BASE_DIR.",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Optional model tag (e.g., d20). If omitted with --stage base/mid/sft we guess the largest tag.",
    )
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("NANOCHAT_BASE_DIR", os.path.join(Path.home(), ".cache", "nanochat")),
        help="Base directory containing nanochat artifacts (default: $NANOCHAT_BASE_DIR or $HOME/.cache/nanochat).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=None,
        help="Set repo to private (default if neither flag is passed).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=None,
        help="Set repo to public (mutually exclusive with --private).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN / HUGGINGFACE_HUB_TOKEN / cached login if omitted.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of repo to create/use. Default: model.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Target Git branch / revision to commit to (e.g., 'main'). Optional.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload via push_to_hf.py",
        help="Commit message for the upload. Default: 'Upload via push_to_hf.py'.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="",
        help="Optional subfolder path in the repo to place the uploaded files (default: root).",
    )
    parser.add_argument(
        "--allow-external-storage",
        action="store_true",
        help="Allow files to be stored on HF's external object storage (useful for very large files).",
    )
    args = parser.parse_args()

    # Resolve token
    token = resolve_token(args.token)

    # Determine which folder we are uploading
    if args.stage:
        base_dir = Path(args.base_dir).expanduser().resolve()
        model_dir = resolve_stage_dir(args.stage, base_dir, args.model_tag)
    else:
        model_dir_arg = args.model_dir or "checkpoints"
        model_dir = Path(model_dir_arg).expanduser().resolve()

    # Basic checks
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"ERROR: --model-dir not found or not a directory: {model_dir}", file=sys.stderr)
        sys.exit(1)

    private = bool_from_flags(args.private, args.public)

    # Informative header
    print("=== Hugging Face Upload ===")
    print(f"Repo ID         : {args.repo_id}")
    print(f"Repo Type       : {args.repo_type}")
    print(f"Model Dir       : {model_dir}")
    print(f"Visibility      : {'private' if private else 'public'}")
    print(f"Branch          : {args.branch or '(default)'}")
    print(f"Path in Repo    : {args.path_in_repo or '(root)'}")
    print(f"Token Source    : {'--token/env/cached' if token else '(none)'}")
    print("===========================")

    # Creating a repo requires auth; uploading to private ALWAYS requires auth.
    if token is None:
        print(
            "ERROR: No Hugging Face token found. "
            "Pass --token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN, or run `huggingface-cli login`.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create or ensure the repo, then upload
    api = HfApi()

    # Ensure you have permissions (e.g., to push to an org, you must be a member with write access).
    try:
        ensure_repo(api, args.repo_id, args.repo_type, private, token, exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{args.repo_id}")
    except HfHubHTTPError as e:
        print(f"ERROR: Could not create or access repo '{args.repo_id}': {e}", file=sys.stderr)
        sys.exit(1)

    # Upload all files from model_dir
    try:
        commit_info = upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(model_dir),
            path_in_repo=args.path_in_repo or None,
            commit_message=args.commit_message,
            token=token,
            revision=args.branch,  # can be None
            # allow_external_storage=args.allow_external_storage,
        )
        # commit_info is a CommitInfo object with .commit_url, .oid, etc.
        print("Upload complete.")
        if getattr(commit_info, "commit_url", None):
            print(f"Commit URL: {commit_info.commit_url}")
    except HfHubHTTPError as e:
        print(f"ERROR: Upload failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
