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
"""Run every selected context-comparison case in a fresh Python process.

The CUTLASS DSL 4.7 frontend can retain compiler preprocessing state across
unrelated specializations in one process. This coordinator first
asks the ordinary comparator for its filtered case list, launches each case in
an independent process, and merges the ordinary one-case JSON reports.  A case
is retried only when it fails to produce a successful result; a measured
performance gap is preserved without cherry-picking another sample.

Invoke this script from the same CUTLASS DSL environment used for the regular
context comparator and pass its arguments after ``--``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DRIVER = Path(__file__).with_name("bench_attention_ts_context_vs_trtllm_gen.py")
MAX_ERROR_ATTEMPTS = 3


def _output_path(arguments: list[str]) -> Path:
    try:
        index = arguments.index("--output")
        return Path(arguments[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("pass the comparator's required --output PATH") from error


def _without_flag(arguments: list[str], flag: str, *, takes_value: bool) -> list[str]:
    filtered: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == flag:
            index += 2 if takes_value else 1
            continue
        if takes_value and argument.startswith(f"{flag}="):
            index += 1
            continue
        filtered.append(argument)
        index += 1
    return filtered


def _selection(arguments: list[str]) -> dict[str, Any]:
    list_arguments = _without_flag(arguments, "--list", takes_value=False)
    list_arguments = _without_flag(list_arguments, "--dry-run", takes_value=False)
    command = [
        sys.executable,
        "-P",
        "-s",
        str(DRIVER),
        *list_arguments,
        "--list",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return json.loads(result.stdout)


def _case_arguments(arguments: list[str], case_id: str, output: Path) -> list[str]:
    filtered = list(arguments)
    for flag, takes_value in (
        ("--output", True),
        ("--case-glob", True),
        ("--resume", False),
        ("--fail-fast", False),
        ("--list", False),
        ("--dry-run", False),
    ):
        filtered = _without_flag(filtered, flag, takes_value=takes_value)
    return [*filtered, "--case-glob", case_id, "--output", str(output), "--fail-fast"]


def _successful_row(report: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    rows = [row for row in report.get("results", []) if row.get("case_id") == case_id]
    if len(rows) == 1 and rows[0].get("status") == "ok":
        return rows[0]
    return None


def _summary(
    *,
    rows: list[dict[str, Any]],
    matrix_count: int,
    selected_count: int,
    started_at: str,
    finished: bool,
) -> dict[str, Any]:
    completed = sum(row.get("status") == "ok" for row in rows)
    failed = sum(row.get("status") == "error" for row in rows)
    over_limit_ids = [
        row["case_id"]
        for row in rows
        if row.get("status") == "ok"
        and not row.get("performance", {}).get("within_gap_limit", False)
    ]
    if not finished:
        status = "running"
    elif failed or completed != selected_count:
        status = "failed"
    elif over_limit_ids:
        status = "performance_gap"
    else:
        status = "ok"
    summary: dict[str, Any] = {
        "status": status,
        "matrix_count": matrix_count,
        "selected": selected_count,
        "completed": completed,
        "failed": failed,
        "over_gap_limit": len(over_limit_ids),
        "over_gap_limit_case_ids": over_limit_ids,
        "all_selected_within_gap_limit": (
            finished
            and not over_limit_ids
            and not failed
            and completed == selected_count
        ),
        "all_64_completed": (
            finished and selected_count == 64 and completed == 64 and failed == 0
        ),
        "all_64_within_gap_limit": (
            finished
            and selected_count == 64
            and completed == 64
            and failed == 0
            and not over_limit_ids
        ),
        "started_at": started_at,
    }
    if finished:
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return summary


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    arguments = list(sys.argv[1:])
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    output = _output_path(arguments)
    selection = _selection(arguments)
    if "--list" in arguments or "--dry-run" in arguments:
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 0
    selected_ids = selection["selected_case_ids"]
    cases_by_id = {case["case_id"]: case for case in selection["cases"]}
    shard_dir = output.parent / f".{output.stem}-isolated-shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] | None = None
    for case_index, case_id in enumerate(selected_ids):
        shard = shard_dir / f"{case_index:03d}.json"
        case_report: dict[str, Any] | None = None
        row: dict[str, Any] | None = None
        for attempt in range(1, MAX_ERROR_ATTEMPTS + 1):
            shard.unlink(missing_ok=True)
            print(
                f"[isolated {case_index + 1}/{len(selected_ids)}] "
                f"{case_id} attempt {attempt}",
                flush=True,
            )
            command = [
                sys.executable,
                "-P",
                "-s",
                str(DRIVER),
                *_case_arguments(arguments, case_id, shard),
            ]
            result = subprocess.run(command, check=False)
            if shard.exists():
                case_report = json.loads(shard.read_text())
                row = _successful_row(case_report, case_id)
            if row is not None:
                break
            print(
                f"[isolated] execution error (exit {result.returncode}); "
                "retrying in a clean process",
                flush=True,
            )

        if case_report is not None and aggregate is None:
            aggregate = {
                "schema_version": case_report["schema_version"],
                "environment": case_report["environment"],
                "protocol": case_report["protocol"],
                "matrix": case_report["matrix"],
                "results": [],
                "summary": {},
            }
            aggregate["environment"]["process_isolation"] = {
                "mode": "fresh_python_process_per_selected_case",
                "coordinator": Path(__file__).name,
                "coordinator_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "maximum_error_attempts_per_case": MAX_ERROR_ATTEMPTS,
            }
            aggregate["protocol"]["selected_case_ids"] = selected_ids

        if row is None:
            if case_report is not None:
                error_rows = [
                    item
                    for item in case_report.get("results", [])
                    if item.get("case_id") == case_id
                ]
                if error_rows:
                    row = error_rows[-1]
            if row is None:
                row = {
                    "case_id": case_id,
                    "case": cases_by_id[case_id],
                    "status": "error",
                    "error": (
                        "isolated comparator produced no successful row across "
                        f"{MAX_ERROR_ATTEMPTS} clean attempts"
                    ),
                }
        rows.append(row)
        rows.sort(key=lambda item: item["case"]["matrix_index"])

        if aggregate is None:
            print("[isolated] no shard report was produced", flush=True)
            return 1
        aggregate["results"] = rows
        aggregate["summary"] = _summary(
            rows=rows,
            matrix_count=int(selection["matrix_count"]),
            selected_count=len(selected_ids),
            started_at=started_at,
            finished=False,
        )
        _write_report(output, aggregate)
        if row.get("status") != "ok":
            break

    assert aggregate is not None
    aggregate["summary"] = _summary(
        rows=rows,
        matrix_count=int(selection["matrix_count"]),
        selected_count=len(selected_ids),
        started_at=started_at,
        finished=True,
    )
    _write_report(output, aggregate)
    status = aggregate["summary"]["status"]
    if status == "ok":
        return 0
    if status == "performance_gap":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
