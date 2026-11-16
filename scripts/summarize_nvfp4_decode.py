#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Print absolute decode timings and optional matched-revision latency ratios."""

import argparse
import json
from pathlib import Path

from attention_ts_decode_artifacts import atomic_write_text
from nvfp4_decode_suite import result_summary, timing_summary


def validate_results(data):
    """Reject missing/duplicated rows and preserve numerical failures in reports."""
    cases = [c for c in data["suite"]["cases"] if c["id"] in data["selected_cases"]]
    selected = data["selected_cases"]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or len(cases) != len(selected)
    ):
        raise ValueError(
            "Selected cases must be nonempty, unique, and present in the suite"
        )
    summary = result_summary(data["rows"], cases, data["backends"])
    if not summary["complete"]:
        raise ValueError("Incomplete or duplicate result rows")
    for row in data["rows"]:
        if row["status"] == "ok":
            measured = timing_summary(row["timing"]["samples_us"])
            if any(
                row["timing"][key] != measured[key] for key in ("best_us", "median_us")
            ):
                raise ValueError("Stored summary differs from raw timing samples")
            if any(
                not row[phase]["all_outputs_finite"] for phase in ("before", "after")
            ):
                raise ValueError("Successful row has nonfinite outputs")
            if row["backend"].startswith("prims_") and any(
                row[phase]["oracle_pass"] is not True for phase in ("before", "after")
            ):
                raise ValueError("Prims timing lacks numerical qualification")
    if data["summary"].get("source_changed_during_run"):
        summary["passed"] = False
    return summary


def paired_rows(before, after):
    """Reject comparisons across workload, timing, software, or input differences."""
    for key in ("suite", "controls", "selected_cases", "backends"):
        if before[key] != after[key]:
            raise ValueError(f"Incompatible comparison contract: {key}")
    for key in ("gpu", "capability", "sms", "torch", "cuda", "dsl", "harness_sha256"):
        if before["metadata"][key] != after["metadata"][key]:
            raise ValueError(f"Incompatible comparison environment: {key}")
    baseline = {(r["case"]["id"], r["backend"]): r for r in before["rows"]}
    for row in after["rows"]:
        old = baseline[(row["case"]["id"], row["backend"])]
        if old["status"] == row["status"] == "ok":
            for key in ("input_sample_sha256", "out_dtype", "mask"):
                if old[key] != row[key]:
                    raise ValueError(f"Incompatible row: {key}")
            if len(old["timing"]["samples_us"]) != len(row["timing"]["samples_us"]):
                raise ValueError("Incompatible sample counts")
    return baseline


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    data = json.loads(args.result.read_text())
    valid = validate_results(data)["passed"]
    baseline = None
    if args.baseline:
        old = json.loads(args.baseline.read_text())
        valid = validate_results(old)["passed"] and valid
        baseline = paired_rows(old, data)
    stat = "best_us" if data["controls"]["protocol"] == "hot-graph" else "median_us"
    lines = [
        f"Source `{data['metadata']['source_commit']}`; {data['metadata']['gpu']}; "
        f"{data['controls']['protocol']}, {stat}. Times are microseconds.",
        "",
        "| Case | Backend | Output | Status | Latency µs | Q CTAs | KV splits | Base µs | Base/current |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in data["rows"]:
        ok = row["status"] == "ok"
        us = row["timing"][stat] if ok else None
        old = baseline[(row["case"]["id"], row["backend"])] if baseline else None
        base_us = old["timing"][stat] if old and old["status"] == "ok" else None
        policy = row.get("policy") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case"]["id"],
                    row["backend"],
                    row.get("out_dtype", "—"),
                    row["status"],
                    f"{us:.3f}" if us else "—",
                    str(policy.get("q_ctas_per_kv_head", "—")),
                    str(policy.get("requested_splits", "—")),
                    f"{base_us:.3f}" if base_us else "—",
                    f"{base_us / us:.3f}" if base_us and us else "—",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "TRT rows are finiteness-only diagnostics; Prims rows require the sampled numerical oracle.",
        "A passing run means requested executions passed their stated checks, not a performance threshold.",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.output:
        atomic_write_text(args.output, text)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
