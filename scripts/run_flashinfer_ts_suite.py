#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run the FlashInfer PrimTS FMHA-decode, MLA-decode, and context suites."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

FMHA_DRIVER = Path("benchmarks/bench_attention_ts_decode.py")
MLA_DRIVER = Path("benchmarks/bench_attention_ts_mla_decode.py")
CONTEXT_DRIVER = Path("benchmarks/attention_ts_suite/bench_attention_ts_context.py")
DECODE_SUPPORT = Path("benchmarks/routines/attention_ts_benchmark")
BUNDLE_ROOT = Path(__file__).resolve().parent
CONTEXT_COMPARISON_DRIVER = BUNDLE_ROOT / "bench_attention_ts_context_vs_trtllm_gen.py"
CONTEXT_METRICS_TEST = BUNDLE_ROOT / "context_comparison_metrics_cpu_checks.py"
RUN_COMPARISON_TEST = BUNDLE_ROOT / "compare_runs_cpu_checks.py"
ACCURACY_GATE_TEST = BUNDLE_ROOT / "run_accuracy_cpu_checks.py"
BUNDLED_DECODE_FILES = {
    FMHA_DRIVER: "bench_attention_ts_decode.py",
    MLA_DRIVER: "bench_attention_ts_mla_decode.py",
    Path("benchmarks/__init__.py"): "decode_benchmarks_init.py",
    Path("benchmarks/routines/__init__.py"): "decode_routines_init.py",
    DECODE_SUPPORT / "__init__.py": "decode_attention_ts_benchmark_init.py",
    DECODE_SUPPORT / "artifacts.py": "attention_ts_decode_artifacts.py",
    DECODE_SUPPORT / "fmha_fixtures.py": "attention_ts_decode_fmha_fixtures.py",
    DECODE_SUPPORT / "mla_fixtures.py": "attention_ts_decode_mla_fixtures.py",
    DECODE_SUPPORT / "timing.py": "attention_ts_decode_timing.py",
    DECODE_SUPPORT / "test_timing_cpu.py": ("decode_timing_cpu_checks.py"),
    Path("benchmarks/test_attention_ts_decode_metrics_cpu.py"): (
        "decode_metrics_cpu_checks.py"
    ),
}
PUBLISHED_BUNDLE_FILES = (
    "run_flashinfer_ts_suite.py",
    "run_accuracy.py",
    "compare_runs.py",
    "bench_attention_ts_context.py",
    "bench_attention_ts_context_vs_trtllm_gen.py",
    "context_comparison_metrics_cpu_checks.py",
    "compare_runs_cpu_checks.py",
    "run_accuracy_cpu_checks.py",
    "run_context_isolated.py",
    *BUNDLED_DECODE_FILES.values(),
)


@dataclass(frozen=True)
class Suite:
    name: str
    rows: int
    description: str
    timing: str


SUITES = (
    Suite(
        "fmha-decode",
        150,
        "Public PrimTS vs TRTLLM-gen paged FMHA decode: 120 SQ1 + 30 SQ2/SQ4/SQ8",
        "paired cold-L2 CUDA graphs",
    ),
    Suite(
        "mla-decode",
        200,
        "Public PrimTS vs TRTLLM-gen paged MLA decode: 100 SQ1 + 100 SQ4",
        "paired cold-L2 CUDA graphs",
    ),
    Suite(
        "context",
        128,
        "PrimTS FP8/BF16 causal context: D128/D256, MHA/GQA, B1/B4, square 1K/4K/16K plus SQ256/SKV4K, ragged/paged",
        "individually cold-L2 CUDA graphs",
    ),
    Suite(
        "context-compare",
        64,
        "FP8 causal context through public PrimTS and TRTLLM-gen interfaces",
        "paired cold-L2 CUDA graphs",
    ),
)
SUITE_BY_NAME = {suite.name: suite for suite in SUITES}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("list", "validate", "run"),
        help="list contracts, validate the bundle on CPU, or run GPU suites",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=tuple(SUITE_BY_NAME),
        help="suite to list/run; repeat to select multiple (default: all)",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="FlashInfer source tree under test (default: current directory)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("attention-ts-results"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use short timing loops for a functional smoke run",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without executing them"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue with later suites after a subprocess failure",
    )
    return parser


def _selected(names: Sequence[str] | None) -> tuple[Suite, ...]:
    wanted = set(names or SUITE_BY_NAME)
    return tuple(suite for suite in SUITES if suite.name in wanted)


def _print_suite_inventory(suites: Sequence[Suite]) -> None:
    print("\nFlashInfer PrimTS benchmark suites:")
    for suite in suites:
        print(
            f"  {suite.name:12} {suite.rows:3} rows  {suite.description} "
            f"[{suite.timing}]"
        )
    sys.stdout.flush()


def _require(path: Path, purpose: str) -> Path:
    path = path.resolve()
    if not path.is_file() and not path.is_dir():
        raise SystemExit(f"missing {purpose}: {path}")
    return path


def _tracked_source_files(source_root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "-z",
            "--",
            "flashinfer",
            "csrc",
            "include",
        ],
        check=True,
        capture_output=True,
    )
    paths = tuple(Path(item.decode()) for item in result.stdout.split(b"\0") if item)
    if not paths:
        raise SystemExit("the FlashInfer checkout has no tracked source files")
    return paths


def _stage_decode_drivers(source_root: Path) -> Path:
    """Stage the bundled drivers with tracked source files under test.

    The original decode scripts prepend their repository root to sys.path. A
    temporary staging tree ensures they import and hash ``--source-root``.
    Only Git-tracked files are copied, so local artifacts cannot enter the
    staging tree.
    """

    bundled_sources = {
        destination: _require(BUNDLE_ROOT / filename, "bundled decode support")
        for destination, filename in BUNDLED_DECODE_FILES.items()
    }
    overlay = Path(tempfile.mkdtemp(prefix="flashinfer-prims-ts-"))
    try:
        (overlay / "benchmarks" / "routines").mkdir(parents=True)
        for relative in _tracked_source_files(source_root):
            source = source_root / relative
            destination = overlay / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                destination.symlink_to(os.readlink(source))
            elif source.is_file():
                shutil.copy2(source, destination)
        data_dir = overlay / "flashinfer" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for name in ("csrc", "include"):
            (data_dir / name).symlink_to(Path("../..") / name, target_is_directory=True)
        for name in ("cccl", "cutlass", "spdlog"):
            source = _require(source_root / "3rdparty" / name, f"{name} checkout")
            (data_dir / name).symlink_to(source, target_is_directory=True)
        for destination, source in bundled_sources.items():
            staged = overlay / destination
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
    except BaseException:
        shutil.rmtree(overlay, ignore_errors=True)
        raise
    return overlay


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_provenance() -> dict[str, object]:
    files = []
    for filename in PUBLISHED_BUNDLE_FILES:
        path = _require(BUNDLE_ROOT / filename, "published benchmark bundle file")
        files.append(
            {
                "name": filename,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {"files": files}


def _validate_staged_bundle(overlay: Path) -> int:
    """Compile staged drivers and run their CPU-only metric contract tests."""

    compile_targets = (
        overlay / FMHA_DRIVER,
        overlay / MLA_DRIVER,
        overlay / DECODE_SUPPORT / "timing.py",
        CONTEXT_COMPARISON_DRIVER,
        CONTEXT_METRICS_TEST,
        RUN_COMPARISON_TEST,
        ACCURACY_GATE_TEST,
    )
    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", *(str(path) for path in compile_targets)],
        cwd=overlay,
        check=False,
    )
    if compile_result.returncode:
        return compile_result.returncode
    decode_tests = (
        overlay / DECODE_SUPPORT / "test_timing_cpu.py",
        overlay / "benchmarks/test_attention_ts_decode_metrics_cpu.py",
    )
    decode_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--rootdir",
            str(overlay),
            *(str(path) for path in decode_tests),
        ],
        cwd=overlay,
        check=False,
    )
    if decode_result.returncode:
        return decode_result.returncode
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--rootdir",
            str(BUNDLE_ROOT),
            str(CONTEXT_METRICS_TEST),
            str(RUN_COMPARISON_TEST),
            str(ACCURACY_GATE_TEST),
        ],
        cwd=BUNDLE_ROOT,
        check=False,
    ).returncode


def _validate_context_contracts(source_root: Path, context_driver: Path) -> None:
    """Check the public context-driver CLIs and their complete row counts."""

    commands = (
        (
            "context",
            128,
            [
                sys.executable,
                str(context_driver),
                "--repo-root",
                str(source_root),
                "--list",
            ],
        ),
        (
            "context-compare",
            64,
            [
                sys.executable,
                str(CONTEXT_COMPARISON_DRIVER),
                "--repo-root",
                str(source_root),
                "--matrix-helper",
                str(context_driver),
                "--list",
            ],
        ),
    )
    for name, expected_rows, command in commands:
        result = subprocess.run(
            command,
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        cases = payload.get("cases")
        if not isinstance(cases, list) or len(cases) != expected_rows:
            actual = len(cases) if isinstance(cases, list) else "missing"
            raise SystemExit(
                f"{name} list contract: expected {expected_rows} rows, got {actual}"
            )


def _append_many(command: list[str], flag: str, values: Sequence[object]) -> None:
    for value in values:
        command.extend((flag, str(value)))


def _fmha_commands(
    python: str, driver: Path, output_dir: Path, device: int, quick: bool
) -> list[tuple[str, list[str], list[Path]]]:
    warmup, iterations = (2, 10) if quick else (10, 200)
    sq1_output = output_dir / "fmha-decode" / "sq1-full120.json"
    sq1_command = [
        python,
        str(driver),
        "--matrix-sweep",
        "--num-kv-heads",
        "4",
        "--expect-cases",
        "120",
        "--cold-l2-cache",
        "--warmup-iters",
        str(warmup),
        "--iters",
        str(iterations),
        "--gap-threshold",
        "5",
        "--fail-on-regression",
        "--continue-on-error",
        "--device",
        str(device),
        "--json-output",
        str(sq1_output),
    ]
    _append_many(sq1_command, "--batch-size", (1, 4, 16, 64, 128))
    _append_many(sq1_command, "--num-qo-heads", (32, 64))
    _append_many(sq1_command, "--kv-len", (2048, 8192))
    _append_many(sq1_command, "--head-dim", (64, 128, 256))
    _append_many(sq1_command, "--dtype", ("fp8", "bf16"))
    grouped_output = output_dir / "fmha-decode" / "pr2265-sq2-sq4-sq8.json"
    return [
        ("fmha-decode/sq1-full120", sq1_command, [sq1_output]),
        (
            "fmha-decode/pr2265-sq2-sq4-sq8",
            [
                python,
                str(driver),
                "--suite",
                "pr2265",
                "--expect-cases",
                "30",
                "--cold-l2-cache",
                "--warmup-iters",
                str(warmup),
                "--iters",
                str(iterations),
                "--gap-threshold",
                "5",
                "--fail-on-regression",
                "--continue-on-error",
                "--device",
                str(device),
                "--json-output",
                str(grouped_output),
            ],
            [grouped_output],
        ),
    ]


def _mla_commands(
    python: str, driver: Path, output_dir: Path, device: int, quick: bool
) -> list[tuple[str, list[str], list[Path]]]:
    warmup, iterations = (2, 10) if quick else (10, 200)
    commands = []
    for q_len in (1, 4):
        base = output_dir / "mla-decode" / f"sq{q_len}-signoff"
        outputs = [base.with_suffix(suffix) for suffix in (".json", ".csv", ".md")]
        command = [
            python,
            str(driver),
            "--q-len",
            str(q_len),
            "--expect-cases",
            "100",
            "--warmup-iters",
            str(warmup),
            "--iters",
            str(iterations),
            "--gap-threshold",
            "5",
            "--fail-on-regression",
            "--continue-on-error",
            "--device",
            str(device),
            "--json-output",
            str(outputs[0]),
            "--csv-output",
            str(outputs[1]),
            "--markdown-output",
            str(outputs[2]),
        ]
        commands.append((f"mla-decode/sq{q_len}", command, outputs))
    return commands


def _context_command(
    python: str,
    driver: Path,
    source_root: Path,
    output_dir: Path,
    device: int,
    quick: bool,
) -> tuple[str, list[str], list[Path]]:
    output = output_dir / "context" / "causal128.json"
    rounds, iterations, warmup = (2, 2, 2) if quick else (6, 20, 50)
    command = [
        python,
        str(driver),
        "--repo-root",
        str(source_root),
        "--output",
        str(output),
        "--device",
        str(device),
        "--rounds",
        str(rounds),
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--fail-fast",
    ]
    return "context", command, [output]


def _context_compare_command(
    python: str,
    source_root: Path,
    output_dir: Path,
    device: int,
    quick: bool,
) -> tuple[str, list[str], list[Path]]:
    output = output_dir / "context" / "fp8-causal-vs-trtllm-gen.json"
    rounds, iterations, warmup = (2, 2, 2) if quick else (6, 20, 50)
    command = [
        python,
        str(CONTEXT_COMPARISON_DRIVER),
        "--repo-root",
        str(source_root),
        "--matrix-helper",
        str(BUNDLE_ROOT / "bench_attention_ts_context.py"),
        "--output",
        str(output),
        "--device",
        str(device),
        "--rounds",
        str(rounds),
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--max-gap-percent",
        "5",
        "--resume",
        "--fail-fast",
    ]
    return "context-compare", command, [output]


def _git_source_identity(source_root: Path) -> dict[str, object]:
    def value(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "git_commit": value("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(
            value("status", "--porcelain", "--untracked-files=no")
        ),
    }


def _redact_command(
    command: Sequence[str],
    *,
    source_root: Path,
    output_dir: Path,
    overlay: Path | None,
) -> list[str]:
    replacements = {
        str(source_root): "$FLASHINFER_ROOT",
        str(output_dir): "$OUTPUT_DIR",
        str(BUNDLE_ROOT): "$SUITE_ROOT/scripts",
        sys.executable: "$PYTHON",
    }
    if overlay is not None:
        replacements[str(overlay)] = "$STAGING_ROOT"
    return [
        next(
            (
                value + item[len(prefix) :]
                for prefix, value in sorted(
                    replacements.items(), key=lambda pair: len(pair[0]), reverse=True
                )
                if item.startswith(prefix)
            ),
            item,
        )
        for item in command
    ]


def _run(command: Sequence[str], *, cwd: Path, env: dict[str, str], dry: bool) -> int:
    print(f"\n$ {shlex.join(command)}", flush=True)
    if dry:
        return 0
    return subprocess.run(command, cwd=cwd, env=env, check=False).returncode


def _artifact_metadata(outputs: Sequence[Path], *, dry: bool) -> dict[str, object]:
    """Load stable metadata without making the runner own artifact schemas."""

    if dry:
        return {}
    for output in outputs:
        if output.suffix != ".json" or not output.is_file():
            continue
        try:
            payload = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError):
            return {"artifact_read_error": True}
        if not isinstance(payload, dict):
            return {"artifact_read_error": True}
        summary = payload.get("summary") if isinstance(payload, dict) else None
        provenance = payload.get("provenance")
        timing = provenance.get("timing") if isinstance(provenance, dict) else None
        if not isinstance(timing, dict) and isinstance(provenance, dict):
            timing = provenance.get("timing_contract")
        return {
            "artifact_schema_version": payload.get("schema_version"),
            "artifact_status": payload.get("status"),
            "timing_method": timing.get("method") if isinstance(timing, dict) else None,
            "summary": summary if isinstance(summary, dict) else None,
        }
    return {}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suites = _selected(args.suite)
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    bundled_context_driver = (
        Path(__file__).resolve().with_name("bench_attention_ts_context.py")
    )
    context_driver = (
        bundled_context_driver
        if bundled_context_driver.is_file()
        else source_root / CONTEXT_DRIVER
    )

    if args.action == "list":
        _print_suite_inventory(suites)
        if any(suite.name in {"context", "context-compare"} for suite in suites):
            _require(context_driver, "context benchmark driver")
            subprocess.run(
                [
                    sys.executable,
                    str(context_driver),
                    "--repo-root",
                    str(source_root),
                    "--list",
                ],
                cwd=source_root,
                check=True,
            )
        return 0

    if args.action == "validate":
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay = _stage_decode_drivers(source_root)
        try:
            _bundle_provenance()
            returncode = _validate_staged_bundle(overlay)
            if returncode == 0:
                _validate_context_contracts(source_root, context_driver)
            print(f"\nValidation exit: {returncode}")
            return returncode
        finally:
            shutil.rmtree(overlay, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_names = {suite.name for suite in suites}
    commands: list[tuple[str, list[str], list[Path]]] = []
    overlay = None
    if selected_names & {"fmha-decode", "mla-decode"}:
        overlay = _stage_decode_drivers(source_root)
        atexit.register(shutil.rmtree, overlay, True)
        if "fmha-decode" in selected_names:
            commands.extend(
                _fmha_commands(
                    sys.executable,
                    overlay / FMHA_DRIVER,
                    output_dir,
                    args.device,
                    args.quick,
                )
            )
        if "mla-decode" in selected_names:
            commands.extend(
                _mla_commands(
                    sys.executable,
                    overlay / MLA_DRIVER,
                    output_dir,
                    args.device,
                    args.quick,
                )
            )
    if "context" in selected_names:
        commands.append(
            _context_command(
                sys.executable,
                _require(context_driver, "context benchmark driver"),
                source_root,
                output_dir,
                args.device,
                args.quick,
            )
        )
    if "context-compare" in selected_names:
        commands.append(
            _context_compare_command(
                sys.executable,
                source_root,
                output_dir,
                args.device,
                args.quick,
            )
        )

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(source_root) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    git_dir = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "--absolute-git-dir"],
        check=False,
        capture_output=True,
        text=True,
    )
    if git_dir.returncode == 0:
        env["GIT_DIR"] = git_dir.stdout.strip()
        env["GIT_WORK_TREE"] = str(source_root)
    records = []
    status = 0
    for name, command, outputs in commands:
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
        returncode = _run(command, cwd=source_root, env=env, dry=args.dry_run)
        record = {
            "name": name,
            "returncode": returncode,
            "command": _redact_command(
                command,
                source_root=source_root,
                output_dir=output_dir,
                overlay=overlay,
            ),
            "outputs": [str(path.relative_to(output_dir)) for path in outputs],
        }
        record.update(_artifact_metadata(outputs, dry=args.dry_run))
        records.append(record)
        if returncode:
            status = returncode
            if not args.continue_on_error:
                break

    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": _git_source_identity(source_root),
        "decode_driver_source": "self-contained-bundle",
        "benchmark_bundle": _bundle_provenance(),
        "dry_run": args.dry_run,
        "quick": args.quick,
        "suites": [suite.name for suite in suites],
        "runs": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if overlay is not None:
        shutil.rmtree(overlay, ignore_errors=True)

    _print_suite_inventory(suites)
    print("\nOutput artifacts:")
    for record in records:
        result = "planned" if args.dry_run else f"exit={record['returncode']}"
        print(f"  {record['name']} ({result})")
        summary = record.get("summary")
        if summary and "gate_regression_row_count" in summary:
            print(
                "    decode gate: "
                f"{summary['gate_regression_row_count']} gate failures; "
                f"{summary.get('raw_median_exception_row_count', 0)} "
                "raw-median exceptions; "
                f"{summary.get('error_row_count', 0)} row errors; "
                f"{summary.get('paired_metric_error_row_count', 0)} metric errors"
            )
        for output in record["outputs"]:
            print(f"    {output}")
    print(f"  manifest\n    {manifest_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
