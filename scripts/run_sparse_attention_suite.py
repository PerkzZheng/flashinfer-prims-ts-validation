#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, PerkzZheng.
"""List, verify, or run the recorded TP2 sparse-attention benchmark suites.

The list and validate actions require only Python's standard library. GPU
execution uses the caller's installed CUDA environment and explicit source
checkouts. This runner never installs packages or allocates cluster jobs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from sparse_attention_suites import _expand_manifest, _select_cases

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "q5": ("q5_tp2.json", (), False),
    "tp2": ("tp2_prefill_decode.json", (), True),
    "prefill-g4": ("tp2_prefill_decode.json", ("prefill-*",), False),
}


def build_command(args: argparse.Namespace, name: str) -> list[str]:
    """Build one complete suite invocation without loading a CUDA module."""

    filename, default_patterns, auto_group = SUITES[name]
    command = [
        sys.executable,
        str(ROOT / "scripts/bench_q_token_kv_block_sparse_attention.py"),
        str(ROOT / "suites/sparse_attention" / filename),
        "--trace-root",
        str(args.trace_root.resolve()),
        "--source-root",
        str(args.source_root.resolve()),
        "--vllm-root",
        str(args.vllm_root.resolve()),
        "--output-json",
        str(args.output_dir.resolve() / f"{name}.json"),
        "--device",
        str(args.device),
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
    ]
    for pattern in default_patterns:
        command.extend(("--case-id", pattern))
    if auto_group:
        command.append("--auto-group-size")
    if args.resume:
        command.append("--resume")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "validate", "run"))
    parser.add_argument(
        "--suite",
        choices=tuple(SUITES),
        action="append",
        help="Repeat to select suites; default: q5 and tp2.",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        help="Directory containing the recorded trace hierarchy.",
    )
    parser.add_argument("--source-root", type=Path, help="FlashInfer Git checkout.")
    parser.add_argument(
        "--vllm-root", type=Path, help="vLLM Git checkout and Triton reference."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/sparse-attention")
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print run commands without CUDA execution.",
    )
    args = parser.parse_args(argv)
    if args.warmup_iterations < 0 or args.iterations <= 0 or args.device < 0:
        parser.error("warmups/device must be nonnegative; iterations must be positive")
    if args.action != "list" and args.trace_root is None:
        parser.error("--trace-root is required for validate/run")
    if args.action == "run" and (args.source_root is None or args.vllm_root is None):
        parser.error("run requires --source-root and --vllm-root")
    names = tuple(dict.fromkeys(args.suite or ("q5", "tp2")))
    for name in names:
        filename, patterns, auto_group = SUITES[name]
        _, cases = _expand_manifest(
            ROOT / "suites/sparse_attention" / filename,
            args.trace_root or Path.cwd(),
            validate_traces=args.action == "validate",
        )
        selected = _select_cases(cases, patterns, None)
        policy = "suggested G (resolved on GPU)" if auto_group else "fixed manifest G"
        print(f"{name}: {len(selected)} cases; {policy}", flush=True)
        if args.action == "list":
            for case in selected:
                print(f"  {case.case_id}")
    if args.action != "run":
        return 0
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(args.source_root.resolve()),
            str(args.vllm_root.resolve()),
            str(ROOT / "scripts"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    for name in names:
        command = build_command(args, name)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            completed = subprocess.run(command, env=environment, cwd=ROOT, check=False)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
