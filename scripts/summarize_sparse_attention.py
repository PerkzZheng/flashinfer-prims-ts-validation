#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, PerkzZheng.
"""Print a compact absolute-latency table from complete sparse suite results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def render(document: dict[str, Any]) -> str:
    """Require complete, accurate rows; compute speedup from pipeline means."""

    if document.get("status") != "complete" or document.get("failures"):
        raise ValueError("cannot summarize an incomplete or failed run as passing")
    cases = document["cases"]
    ids = [row["case_id"] for row in cases]
    if len(ids) != len(set(ids)) or set(ids) != set(document["selected_case_ids"]):
        raise ValueError("result rows do not match the selected case inventory")
    if not cases:
        raise ValueError("run has no result rows")
    lines = [
        "Times are µs. Speedup = Triton / PrimTS; >1 favors PrimTS.",
        "",
        "| Case | G | Splits | Metadata | PrimTS core | PrimTS total | Triton expand | Triton core | Triton total | Speedup |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    protocol = document.get("timing_contract", {}).get("l2_eviction_pattern")
    if protocol != "seeded-random-uint8-add-one-v1":
        lines[0:0] = [
            "Legacy or unrecognized L2 scrub: these timings are not signoff for the corrected non-compressible cold-L2 protocol.",
            "",
        ]
    for row in cases:
        if not row["correctness"].get("post_cuda_graph_replay"):
            raise ValueError("row lacks a post-graph correctness check")
        names = (
            "metadata",
            "prims_ts_attention",
            "prims_ts_combined",
            "triton_expand_indices",
            "triton_attention",
            "triton_combined",
        )
        values = [float(row["timings"][name]["mean_us"]) for name in names]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("latency means must be positive and finite")
        times = " | ".join(f"{value:.3f}" for value in values)
        speedup = values[-1] / values[2]
        lines.append(
            f"| {row['case_id']} | {row['query_group_size']} | "
            f"{row['resolved_splits_kv']} | {times} | {speedup:.3f}x |"
        )
    lines.extend(
        [
            "",
            "Core timings include the backend's split-KV reduction when used. Separately measured components are diagnostics; do not add them to estimate the combined graph or interpret total-minus-core as metadata kernel duration.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.results:
        document = json.loads(path.read_text(encoding="utf-8"))
        print(f"## {path.name}\n")
        print(render(document))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
