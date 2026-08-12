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
"""Compare matched PrimTS rows from baseline and candidate result JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument(
        "--allow-environment-mismatch",
        action="store_true",
        help="compare despite differing non-source environment signatures",
    )
    return parser


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"{path} is not a supported benchmark result")
    return payload


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        dependencies = provenance.get("execution_dependencies")
        if isinstance(dependencies, Mapping):
            source_hashes = dependencies.get("source_hashes")
            return {
                "schema": "mla-decode",
                "versions": dependencies.get("versions"),
                "gpu": dependencies.get("gpu"),
                "trtllm_artifact": dependencies.get("trtllm_artifact"),
                "semantic_contract": dependencies.get("semantic_contract"),
                "oracle_precision": dependencies.get("oracle_precision"),
                "timing_contract": dependencies.get("timing_contract"),
                "benchmark_sha256": (
                    source_hashes.get("benchmark")
                    if isinstance(source_hashes, Mapping)
                    else None
                ),
                "timing_helper_sha256": (
                    source_hashes.get("timing_helper")
                    if isinstance(source_hashes, Mapping)
                    else None
                ),
            }
        if "flashinfer_version" in provenance or "torch_version" in provenance:
            attention_ts = provenance.get("attention_ts")
            trtllm_gen = provenance.get("trtllm_gen")
            benchmark = provenance.get("benchmark")
            timing_helper = provenance.get("timing_helper")
            return {
                "schema": "fmha-decode",
                "versions": {
                    "python": provenance.get("python"),
                    "flashinfer": provenance.get("flashinfer_version"),
                    "torch": provenance.get("torch_version"),
                    "torch_cuda": provenance.get("torch_cuda_version"),
                    "torch_git": provenance.get("torch_git_version"),
                    "cutlass_dsl": provenance.get("cutlass_dsl"),
                },
                "gpu": provenance.get("gpu"),
                "trtllm_artifact": trtllm_gen,
                "semantic_contract": {
                    "interface": (
                        attention_ts.get("interface")
                        if isinstance(attention_ts, Mapping)
                        else None
                    ),
                    "public_api": (
                        attention_ts.get("public_api")
                        if isinstance(attention_ts, Mapping)
                        else None
                    ),
                },
                "timing_contract": provenance.get("timing"),
                "benchmark_sha256": (
                    benchmark.get("sha256") if isinstance(benchmark, Mapping) else None
                ),
                "timing_helper_sha256": (
                    timing_helper.get("sha256")
                    if isinstance(timing_helper, Mapping)
                    else None
                ),
            }
        raise ValueError("unsupported decode provenance schema")

    environment = payload.get("environment", {})
    if not isinstance(environment, Mapping):
        raise ValueError("benchmark result has no environment provenance")
    cutlass = environment.get("cutlass_dsl", {})
    source = environment.get("source", {})
    comparison_script = environment.get("comparison_script")
    external_distributions = environment.get("external_distributions")
    return {
        "schema": (
            "context-compare" if isinstance(comparison_script, Mapping) else "context"
        ),
        "versions": {
            "python": environment.get("python"),
            "torch": environment.get("torch"),
            "torch_cuda": environment.get("torch_cuda"),
            "flashinfer": environment.get("flashinfer"),
            "cutlass_dsl": (
                cutlass.get("distribution_version")
                if isinstance(cutlass, Mapping)
                else None
            ),
        },
        "gpu": environment.get("gpu"),
        "benchmark_sha256": (
            source.get("benchmark_sha256") if isinstance(source, Mapping) else None
        ),
        "timing_helper_sha256": (
            source.get("timing_helper_sha256") if isinstance(source, Mapping) else None
        ),
        "comparison_script": comparison_script,
        "matrix_helper": environment.get("matrix_helper"),
        "cutlass_dsl_package": environment.get("cutlass_dsl_package"),
        "trtllm_artifact": external_distributions,
        "protocol": payload.get("protocol"),
    }


def _validate_environment_contract(contract: Mapping[str, Any]) -> None:
    schema = contract.get("schema")
    if schema not in {"fmha-decode", "mla-decode", "context", "context-compare"}:
        raise ValueError(f"unsupported environment contract: {schema}")
    versions = contract.get("versions")
    if not isinstance(versions, Mapping) or any(
        versions.get(name) in (None, "")
        for name in ("python", "torch", "torch_cuda", "flashinfer", "cutlass_dsl")
    ):
        raise ValueError("benchmark result lacks complete software-version provenance")
    gpu = contract.get("gpu")
    if not isinstance(gpu, Mapping) or not gpu.get("name"):
        raise ValueError("benchmark result lacks GPU provenance")
    benchmark_sha256 = contract.get("benchmark_sha256")
    if (
        not isinstance(benchmark_sha256, str)
        or len(benchmark_sha256) != 64
        or any(character not in "0123456789abcdef" for character in benchmark_sha256)
    ):
        raise ValueError("benchmark result lacks a valid benchmark source hash")
    timing_helper_sha256 = contract.get("timing_helper_sha256")
    if (
        not isinstance(timing_helper_sha256, str)
        or len(timing_helper_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in timing_helper_sha256
        )
    ):
        raise ValueError("benchmark result lacks a valid timing-helper source hash")
    if schema in {"fmha-decode", "mla-decode", "context-compare"}:
        artifact = contract.get("trtllm_artifact")
        if not isinstance(artifact, Mapping) or not artifact:
            raise ValueError("paired result lacks TRTLLM-Gen artifact provenance")
        if schema == "fmha-decode":
            if not artifact.get("resolved_manifest_sha256") or not artifact.get(
                "flashinfer_cubin_version"
            ):
                raise ValueError("FMHA result lacks verified reference identity")
        elif schema == "mla-decode":
            if not artifact.get("verified_manifest_sha256") or not artifact.get(
                "flashinfer_cubin_version"
            ):
                raise ValueError("MLA result lacks verified reference identity")
        else:
            cubin = artifact.get("flashinfer-cubin")
            if not isinstance(cubin, Mapping) or not cubin.get("version"):
                raise ValueError("context result lacks reference package identity")


def _row_latency_ms(row: Mapping[str, Any]) -> tuple[float, int]:
    attention_ts = row.get("attention_ts")
    if isinstance(attention_ts, Mapping):
        total = attention_ts.get("total_duration_ms")
        count = attention_ts.get("sample_count")
        if isinstance(total, (int, float)) and isinstance(count, int) and count > 0:
            return float(total) / count, count
        mean_us = attention_ts.get("arithmetic_mean_us")
        if isinstance(mean_us, (int, float)):
            return float(mean_us) / 1000.0, int(count or 0)

    performance = row.get("performance")
    if isinstance(performance, Mapping):
        ts = performance.get("ts")
        if isinstance(ts, Mapping):
            total = ts.get("total_duration_ms")
            samples = ts.get("sample_ms_by_round")
            count = (
                sum(len(round_) for round_ in samples)
                if isinstance(samples, list)
                else 0
            )
            if isinstance(total, (int, float)) and count > 0:
                return float(total) / count, count
            mean = ts.get("arithmetic_mean_ms")
            if isinstance(mean, (int, float)):
                return float(mean), count
        samples = performance.get("samples_ms_by_round")
        if isinstance(samples, list):
            flat = [float(value) for round_ in samples for value in round_]
            if flat:
                return sum(flat) / len(flat), len(flat)
        mean = performance.get("mean_ms")
        if isinstance(mean, (int, float)):
            return float(mean), 0
    raise ValueError(f"row {row.get('case_id')} has no PrimTS timing")


def _row_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    performance = row.get("performance", {})
    attention_ts = row.get("attention_ts", {})
    return {
        "shape": row.get("shape", row.get("case")),
        "timing": {
            "mode": (
                attention_ts.get("timing_mode")
                if isinstance(attention_ts, Mapping)
                else None
            ),
            "cold_l2": (
                attention_ts.get("cold_l2_cache")
                if isinstance(attention_ts, Mapping)
                else performance.get("cache_state")
                if isinstance(performance, Mapping)
                else None
            ),
            "rounds": (
                performance.get("rounds") if isinstance(performance, Mapping) else None
            ),
            "iterations_per_round": (
                performance.get("iterations_per_round")
                if isinstance(performance, Mapping)
                else None
            ),
        },
    }


def _rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in payload["results"]:
        if not isinstance(row, Mapping):
            raise ValueError("result rows must be JSON objects")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in rows:
            raise ValueError(f"missing or duplicate case ID: {case_id}")
        if row.get("status") != "ok":
            raise ValueError(f"row {case_id} is not successful")
        rows[case_id] = row
    if not rows:
        raise ValueError("benchmark result contains no rows")
    return rows


def _render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# PrimTS baseline comparison",
        "",
        f"Threshold: `{summary['threshold_percent']:.3f}%`.",
        "",
        "| Case | Baseline (us) | Candidate (us) | Drift | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["results"]:
        lines.append(
            f"| `{row['case_id']}` | {row['baseline_ms'] * 1000:.3f} | "
            f"{row['candidate_ms'] * 1000:.3f} | {row['regression_percent']:+.3f}% | "
            f"{row['status']} |"
        )
    lines.extend(
        [
            "",
            f"Worst drift: `{summary['worst_regression_percent']:+.3f}%` "
            f"at `{summary['worst_case_id']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not math.isfinite(args.threshold) or args.threshold < 0:
        raise SystemExit("--threshold must be finite and nonnegative")
    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve()
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    baseline_environment = _environment_contract(baseline)
    candidate_environment = _environment_contract(candidate)
    _validate_environment_contract(baseline_environment)
    _validate_environment_contract(candidate_environment)
    environment_match = baseline_environment == candidate_environment
    if not environment_match and not args.allow_environment_mismatch:
        raise SystemExit(
            "baseline and candidate environment/protocol signatures differ; "
            "rerun on one matched system or pass --allow-environment-mismatch"
        )

    baseline_rows = _rows(baseline)
    candidate_rows = _rows(candidate)
    if baseline_rows.keys() != candidate_rows.keys():
        missing = sorted(baseline_rows.keys() - candidate_rows.keys())
        extra = sorted(candidate_rows.keys() - baseline_rows.keys())
        raise SystemExit(f"case sets differ: missing={missing}, extra={extra}")

    results = []
    for case_id in baseline_rows:
        baseline_row = baseline_rows[case_id]
        candidate_row = candidate_rows[case_id]
        if _row_contract(baseline_row) != _row_contract(candidate_row):
            raise SystemExit(f"row contract differs for {case_id}")
        baseline_ms, baseline_samples = _row_latency_ms(baseline_row)
        candidate_ms, candidate_samples = _row_latency_ms(candidate_row)
        if (
            not math.isfinite(baseline_ms)
            or not math.isfinite(candidate_ms)
            or baseline_ms <= 0
            or candidate_ms <= 0
        ):
            raise SystemExit(f"nonfinite or nonpositive latency for {case_id}")
        if baseline_samples <= 0 or candidate_samples <= 0:
            raise SystemExit(f"missing timing sample count for {case_id}")
        if baseline_samples != candidate_samples:
            raise SystemExit(f"sample count differs for {case_id}")
        regression = (candidate_ms / baseline_ms - 1.0) * 100.0
        results.append(
            {
                "case_id": case_id,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "sample_count": candidate_samples or baseline_samples,
                "regression_percent": regression,
                "status": "regression" if regression > args.threshold else "pass",
            }
        )
    results.sort(key=lambda row: row["regression_percent"], reverse=True)
    regressions = [row for row in results if row["status"] == "regression"]
    worst = results[0] if results else None
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_sha256": _file_sha256(baseline_path),
        "candidate_sha256": _file_sha256(candidate_path),
        "environment_signature_match": environment_match,
        "environment_signature_sha256": {
            "baseline": _canonical_sha256(baseline_environment),
            "candidate": _canonical_sha256(candidate_environment),
        },
        "summary": {
            "status": "regression" if regressions else "ok",
            "threshold_percent": args.threshold,
            "case_count": len(results),
            "regression_count": len(regressions),
            "worst_case_id": worst["case_id"] if worst else None,
            "worst_regression_percent": (
                worst["regression_percent"] if worst else None
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_render_markdown(report))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 2 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
