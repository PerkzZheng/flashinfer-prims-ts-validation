#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Interleave base and candidate mask-tile runs on one GPU, then compare.

Round ``k`` runs both checkouts back to back, alternating which goes first
(base/candidate, then candidate/base, ...), each in a fresh process. Every
process writes one result JSON; the comparison follows
``summarize_mask_tiles.py``.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from summarize_mask_tiles import summarize

SCRIPTS = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--suite", type=Path)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--case", action="append", help="Select a case ID (repeatable)")
    p.add_argument("--samples", type=int, help="Override timing samples per run")
    p.add_argument("--max-regression", type=float, default=0.01)
    p.add_argument("--allow-output-change", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    out = args.output_dir
    if out.exists() and any(out.glob("*.json")):
        raise ValueError("Choose a new --output-dir; results are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    roots = {"base": args.base_root, "candidate": args.candidate_root}
    files = {"base": [], "candidate": []}
    for k in range(args.rounds):
        order = ("base", "candidate") if k % 2 == 0 else ("candidate", "base")
        for label in order:
            path = out / f"{label}-r{k}.json"
            cmd = [
                sys.executable,
                str(SCRIPTS / "bench_attention_ts_mask_tiles.py"),
                "--source-root",
                str(roots[label]),
                "--output",
                str(path),
                "--workdir",
                str(out),
            ]
            if args.suite:
                cmd += ["--suite", str(args.suite)]
            for case in args.case or ():
                cmd += ["--case", case]
            if args.samples:
                cmd += ["--samples", str(args.samples)]
            print(f"[round {k + 1}/{args.rounds}] {label}", flush=True)
            with (out / f"{label}-r{k}.log").open("w") as log:
                rc = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT
                ).returncode
            if not path.exists():
                raise RuntimeError(f"{label} round {k} wrote no result (exit {rc})")
            files[label].append(path)
    return summarize(
        files["base"],
        files["candidate"],
        out / "comparison",
        max_regression=args.max_regression,
        allow_output_change=args.allow_output_change,
    )


if __name__ == "__main__":
    raise SystemExit(main())
