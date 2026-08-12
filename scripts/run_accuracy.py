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
"""Run the numerical PrimTS accuracy gate from a FlashInfer checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ACCURACY_TESTS = (
    "tests/attention/test_attention_ts_context.py",
    "tests/attention/test_attention_ts_decode.py",
    "tests/attention/test_attention_ts_mask.py",
    "tests/attention/test_attention_ts_mla_decode.py",
)
TRACE_TEST = "tests/trace/test_fi_trace_template_consistency.py"
EXPECTED_CUTLASS_DSL_VERSION = "4.7.0"
MINIMUM_ACCURACY_TEST_COUNT = 360


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="FlashInfer Git checkout to test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for accuracy.log, accuracy.xml, and manifest.json",
    )
    parser.add_argument(
        "--device",
        type=int,
        help="optional physical GPU index to expose as CUDA device 0",
    )
    parser.add_argument(
        "--include-trace",
        action="store_true",
        help="also run the structural trace-template suite",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="additional pytest argument; repeat as needed",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="diagnostic only: allow fewer than 360 executed tests or skips",
    )
    parser.add_argument("--list", action="store_true", help="print tests and exit")
    return parser


def _git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _junit_summary(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


def _runtime_versions(env: dict[str, str], source_root: Path) -> dict[str, object]:
    script = """
import json
from pathlib import Path
import torch
import flashinfer
payload = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "flashinfer": getattr(flashinfer, "__version__", "unknown"),
    "flashinfer_file": str(Path(flashinfer.__file__).resolve()),
}
print(json.dumps(payload))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    imported = Path(payload.pop("flashinfer_file"))
    if not imported.is_relative_to((source_root / "flashinfer").resolve()):
        raise SystemExit("flashinfer import did not resolve under --source-root")
    return payload


def _redact_argument(value: str, *, source_root: Path, output_dir: Path) -> str:
    return value.replace(str(source_root), "$FLASHINFER_ROOT").replace(
        str(output_dir), "$OUTPUT_DIR"
    )


def _accuracy_gate_returncode(
    pytest_returncode: int,
    junit_summary: dict[str, int] | None,
    *,
    allow_partial: bool,
) -> int:
    if pytest_returncode:
        return pytest_returncode
    if allow_partial:
        return 0
    if junit_summary is None:
        return 3
    if (
        junit_summary["tests"] < MINIMUM_ACCURACY_TEST_COUNT
        or junit_summary["skipped"] != 0
    ):
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_root = args.source_root.resolve()
    tests = list(ACCURACY_TESTS)
    if args.include_trace:
        tests.append(TRACE_TEST)
    if args.list:
        print("\n".join(tests))
        return 0
    if not (source_root / ".git").exists():
        raise SystemExit(f"not a FlashInfer Git checkout: {source_root}")
    missing = [test for test in tests if not (source_root / test).is_file()]
    if missing:
        raise SystemExit(f"missing test files: {missing}")

    wheel_version = importlib.metadata.version("nvidia-cutlass-dsl")
    if wheel_version.partition("+")[0] != EXPECTED_CUTLASS_DSL_VERSION:
        raise SystemExit(
            f"expected nvidia-cutlass-dsl {EXPECTED_CUTLASS_DSL_VERSION}, "
            f"got {wheel_version}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "accuracy.log"
    junit_path = output_dir / "accuracy.xml"
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(source_root) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    versions = _runtime_versions(env, source_root)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *tests,
        f"--junitxml={junit_path}",
        *args.pytest_arg,
    ]

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = process.wait()

    pytest_returncode = returncode
    junit_summary = _junit_summary(junit_path)
    returncode = _accuracy_gate_returncode(
        pytest_returncode, junit_summary, allow_partial=args.allow_partial
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_commit": _git_value(source_root, "rev-parse", "HEAD"),
            "tracked_worktree_dirty": bool(
                _git_value(source_root, "status", "--porcelain", "--untracked-files=no")
            ),
        },
        "versions": {
            **versions,
            "nvidia_cutlass_dsl": wheel_version,
            "python": sys.version,
        },
        "tests": tests,
        "includes_structural_trace_test": args.include_trace,
        "controls": {
            "physical_device_exposed_as_cuda_zero": args.device,
            "minimum_executed_test_count": MINIMUM_ACCURACY_TEST_COUNT,
            "zero_skips_required": True,
            "allow_partial_diagnostic": args.allow_partial,
            "additional_pytest_arguments": [
                _redact_argument(value, source_root=source_root, output_dir=output_dir)
                for value in args.pytest_arg
            ],
        },
        "command": [
            "$PYTHON",
            "-m",
            "pytest",
            "-q",
            *tests,
            "--junitxml=$OUTPUT_DIR/accuracy.xml",
            *[
                _redact_argument(value, source_root=source_root, output_dir=output_dir)
                for value in args.pytest_arg
            ],
        ],
        "pytest_returncode": pytest_returncode,
        "returncode": returncode,
        "junit": {
            "file": junit_path.name,
            "sha256": _sha256(junit_path),
            "summary": junit_summary,
        },
        "log": {"file": log_path.name, "sha256": _sha256(log_path)},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
