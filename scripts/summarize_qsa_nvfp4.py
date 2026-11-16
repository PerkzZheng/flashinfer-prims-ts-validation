#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Render QSA NVFP4 latencies, throughput and speedups from reports."""

import argparse
import json
from pathlib import Path

BACKENDS = ("nvfp4-fp8mma", "nvfp4-bf16mma", "fp8")
# Per comparison: header lines, (backend, column label) latency columns,
# speedup columns as (speedup key, label) and throughput columns.
LAYOUTS = {
    "fp8": dict(
        note="speedup = FP8-KV / NVFP4 FP8-MMA (>1 favors NVFP4).",
        latency=(
            ("nvfp4-fp8mma", "NVFP4 FP8-MMA µs"),
            ("nvfp4-bf16mma", "NVFP4 BF16-MMA µs"),
            ("fp8", "FP8 µs"),
        ),
        speedups=(("fp8", "vs FP8"),),
        throughput=(
            ("nvfp4-fp8mma", "achieved_tbps", "NVFP4 TB/s"),
            ("nvfp4-fp8mma", "achieved_tflops", "NVFP4 TFLOP/s"),
            ("fp8", "achieved_tbps", "FP8 TB/s"),
        ),
    ),
    "bf16": dict(
        note=(
            "speedup = BF16-KV / NVFP4 BF16-MMA rows; separate = separate-scales "
            "median / rows median (>1 favors the rows layout)."
        ),
        latency=(
            ("nvfp4-bf16mma", "NVFP4 rows µs"),
            ("nvfp4-bf16mma-separate", "NVFP4 separate µs"),
            ("bf16", "BF16 µs"),
        ),
        speedups=(("bf16", "vs BF16"), ("nvfp4-bf16mma-separate", "separate")),
        throughput=(
            ("nvfp4-bf16mma", "achieved_tbps", "NVFP4 TB/s"),
            ("nvfp4-bf16mma", "achieved_tflops", "NVFP4 TFLOP/s"),
            ("bf16", "achieved_tbps", "BF16 TB/s"),
        ),
    ),
}


def render(reports):
    """Merge one or more reports; failed or missing rows stay visible as N/A."""
    rows, requested, sources = [], [], set()
    comparisons = {report.get("comparison", "fp8") for report in reports}
    if len(comparisons) != 1:
        raise ValueError("Summarize reports of one comparison at a time")
    layout = LAYOUTS[comparisons.pop()]
    for report in reports:
        if report.get("schema") != "qsa-nvfp4-v1":
            raise ValueError("Expected qsa-nvfp4-v1 reports")
        rows.extend(report["cases"])
        requested.extend(report["requested"])
        provenance = report.get("provenance", {})
        if provenance:
            sources.add(
                (
                    provenance["gpu"],
                    provenance["cuda"],
                    provenance["dsl"],
                    provenance["source_revision"],
                    provenance["prims_source_sha256"],
                )
            )
    seen = {r["id"] for r in rows}
    headers = (
        ["Case"]
        + [label for _, label in layout["latency"]]
        + [label for _, label in layout["speedups"]]
        + [label for _, _, label in layout["throughput"]]
    )
    lines = [
        f"Completed {len(seen)}/{len(set(requested))} requested cases.",
        "",
        "Median microseconds; " + layout["note"],
        "TB/s counts unique needed K/V once (NVFP4: payload + scales) plus Q and O.",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|" + "---:|" * (len(headers) - 1),
    ]
    if any(report.get("quick") for report in reports):
        lines.insert(1, "Includes `--quick` functional runs; not performance signoff.")
    for row in rows:
        backends = row.get("backends", {})
        cells = [row["id"]]
        for name, _ in layout["latency"]:
            result = backends.get(name, {})
            cells.append(
                f"{result['median_us']:.2f}"
                if result.get("status") == "ok"
                else f"N/A ({result.get('status', row.get('status', 'missing'))})"
            )
        for key, _ in layout["speedups"]:
            speedup = row.get("speedups", {}).get(key)
            cells.append(f"{speedup:.3f}×" if speedup is not None else "N/A")
        for name, key, _ in layout["throughput"]:
            value = backends.get(name, {}).get(key)
            cells.append(f"{value:.2f}" if value is not None else "N/A")
        lines.append("| " + " | ".join(cells) + " |")
    for missing in sorted(set(requested) - seen):
        lines.append(f"| {missing} | N/A (missing) |" + " |" * (len(headers) - 2))
    for gpu, cuda, dsl, revision, digest in sorted(sources):
        lines.append(
            f"\nDevice: {gpu}; CUDA {cuda}; CuTe DSL {dsl}; FlashInfer {revision} "
            f"(Prims-TS sources {digest[:12]})."
        )
    return "\n".join(lines) + "\n"


def main():
    """Print Markdown, or save it to the explicitly requested output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = render([json.loads(path.read_text()) for path in args.reports])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
