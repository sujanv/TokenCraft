#!/usr/bin/env python3
"""
TokenCraft Sequential Git Pusher
Pushes git commits to remote repository one at a time sequentially,
with randomized delays (e.g. 30 to 45 minutes) between pushes.

Usage:
  # Loop mode: push next commit, wait 30-45 mins at random, push next...
  python3 scripts/push_sequentially.py

  # Push just one single commit right now
  python3 scripts/push_sequentially.py --step-once

  # Dry-run inspection
  python3 scripts/push_sequentially.py --dry-run
"""

from __future__ import annotations
import subprocess
import argparse
import random
import time
import sys
import json
import os
from datetime import datetime, timedelta

STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".git", "push_state.json")


def run_cmd(cmd: list[str]) -> str:
    """Run a shell command and return stdout."""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {res.stderr.strip()}")
    return res.stdout.strip()


def get_commit_list(branch: str = "main", remote_branch: str = "origin/main") -> list[dict]:
    """
    Get all local commits on branch ahead of remote_branch in chronological order (oldest to newest).
    """
    try:
        # Check if remote branch exists
        run_cmd(["git", "rev-parse", "--verify", remote_branch])
        range_spec = f"{remote_branch}..{branch}"
    except Exception:
        # Remote doesn't have the branch yet
        range_spec = branch

    log_output = run_cmd([
        "git", "log", "--reverse", "--pretty=format:%H%x09%an%x09%ad%x09%s", range_spec
    ])

    commits = []
    if not log_output:
        return commits

    for line in log_output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            commits.append({
                "hash": parts[0],
                "author": parts[1],
                "date": parts[2],
                "subject": parts[3],
            })
    return commits


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"pushed_hashes": [], "last_push_time": None}


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def push_single_commit(commit_hash: str, branch: str = "main", remote: str = "origin", dry_run: bool = False):
    """Pushes a specific commit to the remote tracking branch."""
    refspec = f"{commit_hash}:refs/heads/{branch}"
    if dry_run:
        print(f"[DRY RUN] Would execute: git push {remote} {refspec}")
        return True

    print(f"Executing: git push {remote} {refspec}...")
    res = subprocess.run(["git", "push", remote, refspec], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"Error pushing commit {commit_hash}:\n{res.stderr}", file=sys.stderr)
        return False
    print(f"Successfully pushed commit {commit_hash[:7]} to {remote}/{branch}!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sequential Git commit pusher with random delays")
    parser.add_argument("--interval-min", type=float, default=30.0, help="Minimum delay between pushes in minutes (default: 30)")
    parser.add_argument("--interval-max", type=float, default=45.0, help="Maximum delay between pushes in minutes (default: 45)")
    parser.add_argument("--step-once", action="store_true", help="Push only the next commit and exit immediately")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without actually pushing")
    parser.add_argument("--remote", default="origin", help="Git remote name (default: origin)")
    parser.add_argument("--branch", default="main", help="Git branch name (default: main)")
    parser.add_argument("--push-all", action="store_true", help="Push all remaining commits immediately")
    args = parser.parse_args()

    state = load_state()
    commits = get_commit_list(branch=args.branch, remote_branch=f"{args.remote}/{args.branch}")

    if not commits:
        print(f"Repository is up to date! No unpushed commits found between {args.branch} and {args.remote}/{args.branch}.")
        return

    print(f"Found {len(commits)} unpushed commit(s) on branch '{args.branch}':")
    for i, c in enumerate(commits, 1):
        print(f"  {i}. [{c['hash'][:7]}] {c['subject']} ({c['date']})")

    if args.push_all:
        print(f"\nPushing all {len(commits)} commits immediately to {args.remote}/{args.branch}...")
        if not args.dry_run:
            run_cmd(["git", "push", args.remote, args.branch])
        print("All commits pushed!")
        return

    if args.step_once:
        next_commit = commits[0]
        print(f"\n[Step Once] Pushing next commit: [{next_commit['hash'][:7]}] {next_commit['subject']}")
        success = push_single_commit(next_commit["hash"], branch=args.branch, remote=args.remote, dry_run=args.dry_run)
        if success and not args.dry_run:
            state["pushed_hashes"].append(next_commit["hash"])
            state["last_push_time"] = datetime.now().isoformat()
            save_state(state)
        return

    # Loop mode: push sequentially with randomized intervals
    print(f"\nStarting sequential push loop (Interval: {args.interval_min} - {args.interval_max} mins at random)...")

    for i, commit in enumerate(commits, 1):
        print(f"\n[{i}/{len(commits)}] Pushing commit: [{commit['hash'][:7]}] {commit['subject']}")
        success = push_single_commit(commit["hash"], branch=args.branch, remote=args.remote, dry_run=args.dry_run)
        if not success:
            print("Push failed. Aborting sequential push.", file=sys.stderr)
            sys.exit(1)

        state["pushed_hashes"].append(commit["hash"])
        state["last_push_time"] = datetime.now().isoformat()
        save_state(state)

        # If more commits remain, calculate randomized sleep time
        if i < len(commits):
            delay_minutes = random.uniform(args.interval_min, args.interval_max)
            delay_seconds = int(delay_minutes * 60)
            next_push_time = datetime.now() + timedelta(seconds=delay_seconds)
            print(f"Waiting {delay_minutes:.1f} minutes ({delay_seconds}s) until next push at {next_push_time.strftime('%H:%M:%S')}...")

            if not args.dry_run:
                try:
                    time.sleep(delay_seconds)
                except KeyboardInterrupt:
                    print("\nSequential pusher paused by user. Run again to resume where you left off.")
                    sys.exit(0)

    print("\nAll commits successfully pushed to origin!")


if __name__ == "__main__":
    main()
