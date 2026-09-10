# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only manifest expansion, case selection, and recorded-route checks."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SuiteCase:
    case_id: str
    phase: str
    tp: int
    dtype_name: str
    batch_size: int
    seq_len_q: int
    kv_length: int
    group_size: int
    query_layout: str
    trace_paths: tuple[Path, ...]
    q5_start_position: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_declared_trace_hashes(topk: dict[str, Any], workspace_root: Path) -> None:
    """Fail closed when a recorded route artifact differs from its manifest."""

    declared: dict[str, str]
    if "files" in topk:
        files = list(topk["files"])
        hashes = list(topk.get("sha256", []))
        if len(files) != len(hashes):
            raise ValueError("top-k files and sha256 lists must have equal length")
        declared = dict(zip(files, hashes, strict=True))
    else:
        files = [
            value
            for trace_set in topk.get("trace_sets", {}).values()
            for value in trace_set
        ]
        declared = dict(topk.get("trace_sha256", {}))
        if set(files) != set(declared):
            raise ValueError(
                "top-k trace sets and trace_sha256 keys must match exactly"
            )

    for relative_path, expected in declared.items():
        path = workspace_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing top-k trace: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"top-k trace checksum mismatch for {path}: "
                f"expected {expected}, got {actual}"
            )


def _expand_manifest(
    manifest_path: Path, trace_root: Path, *, validate_traces: bool = True
) -> tuple[dict[str, Any], list[SuiteCase]]:
    manifest_path = manifest_path.resolve()
    workspace_root = trace_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported QSA suite schema")
    if not isinstance(manifest.get("model_len"), int) or manifest["model_len"] <= 0:
        raise ValueError("QSA suite model_len must be a positive integer")
    topk = manifest["topk_source"]
    if validate_traces:
        _validate_declared_trace_hashes(topk, workspace_root)
    cases: list[SuiteCase] = []
    if "cases" in manifest:
        geometry = manifest["geometry"]
        paths = tuple(workspace_root / value for value in topk["files"])
        for value in manifest["cases"]:
            cases.append(
                SuiteCase(
                    case_id=value["case_id"],
                    phase=value["phase"],
                    tp=int(value["tp"]),
                    dtype_name=value["dtype"],
                    batch_size=int(value["batch_size"]),
                    seq_len_q=int(geometry["query_group_size"]),
                    kv_length=int(geometry["context_length"]),
                    group_size=int(geometry["query_group_size"]),
                    query_layout=geometry["query_layout"],
                    trace_paths=paths,
                    q5_start_position=int(geometry["first_logical_position"]),
                )
            )
    else:
        geometry = manifest["geometry"]
        workloads = manifest["workloads"]
        trace_sets = topk["trace_sets"]
        tp = int(geometry["tp"])
        for dtype_name in geometry["qkv_dtypes"]:
            for length in workloads["prefill"]["query_lengths"]:
                cases.append(
                    SuiteCase(
                        case_id=f"prefill-tp{tp}-{dtype_name}-q{length}-kv{length}-bs1-g4",
                        phase="prefill",
                        tp=tp,
                        dtype_name=dtype_name,
                        batch_size=1,
                        seq_len_q=int(length),
                        kv_length=int(length),
                        group_size=int(workloads["prefill"]["query_group_size"]),
                        query_layout=workloads["prefill"]["query_layout"],
                        trace_paths=tuple(
                            workspace_root / value for value in trace_sets[str(length)]
                        ),
                    )
                )
        for dtype_name in geometry["qkv_dtypes"]:
            for length in workloads["decode"]["kv_lengths"]:
                for mtp, group_size in workloads["decode"][
                    "mtp_to_query_group_size"
                ].items():
                    for batch_size in workloads["decode"]["batch_sizes"]:
                        cases.append(
                            SuiteCase(
                                case_id=(
                                    f"decode-tp{tp}-{dtype_name}-kv{length}-"
                                    f"bs{batch_size}-mtp{mtp}-g{group_size}"
                                ),
                                phase="decode",
                                tp=tp,
                                dtype_name=dtype_name,
                                batch_size=int(batch_size),
                                seq_len_q=int(group_size),
                                kv_length=int(length),
                                group_size=int(group_size),
                                query_layout=workloads["decode"]["query_layout"],
                                trace_paths=tuple(
                                    workspace_root / value
                                    for value in trace_sets[str(length)]
                                ),
                            )
                        )
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("suite contains duplicate case IDs")
    expected = manifest.get("matrix", {}).get("total_cases")
    if expected is not None and len(cases) != int(expected):
        raise ValueError(
            f"manifest declares {expected} cases but expands to {len(cases)}"
        )
    for case in cases:
        if case.phase == "prefill":
            expected_layout = "packed"
        elif case.phase == "decode":
            expected_layout = "fixed_unpacked_[B,num_groups,G,Hq,D]"
        elif case.phase == "grouped_decode_proxy":
            expected_layout = "fixed_grouped"
        else:
            raise ValueError(f"unsupported suite phase: {case.phase}")
        if case.query_layout != expected_layout:
            raise ValueError(
                f"{case.case_id}: query_layout={case.query_layout!r}, "
                f"expected {expected_layout!r} for {case.phase}"
            )
        missing = [str(path) for path in case.trace_paths if not path.is_file()]
        if validate_traces and missing:
            raise FileNotFoundError(f"{case.case_id}: missing trace files: {missing}")
    return manifest, cases


def _select_cases(
    cases: Sequence[SuiteCase],
    patterns: Sequence[str],
    max_cases: int | None,
) -> list[SuiteCase]:
    selected = [
        case
        for case in cases
        if not patterns
        or any(fnmatch.fnmatchcase(case.case_id, pattern) for pattern in patterns)
    ]
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise ValueError("case filters selected no benchmark cases")
    return selected
