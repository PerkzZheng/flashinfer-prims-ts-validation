#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Render absolute MLA latencies and reference/NVFP4 speedups from one report."""

import argparse
import json
from pathlib import Path


def render(report):
    """Keep failed/missing comparisons visible as N/A, including partial campaigns."""
    if report.get("schema") != "scaled-mla-v1":
        raise ValueError("Expected a scaled-mla-v1 report")
    lines = [
        f"Status: {report['status']}. Completed {len(report['cases'])}/{len(report['requested'])} cases.",
        "",
        "Latency is median microseconds. Speedup = reference / packed NVFP4; >1 favors packed Prims-TS.",
        "",
        "| Case | NVFP4 µs | FP8 µs | FlashMLA µs | vs FP8 | vs FlashMLA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if report.get("quick"):
        lines.insert(1, "Functional smoke only (`--quick`); not performance signoff.")
    for row in report["cases"]:
        timings = []
        for name in ("prims-nvfp4", "prims-fp8", "flashmla"):
            result = row.get("backends", {}).get(name, {})
            timings.append(
                f"{result['median_us']:.3f}"
                if result.get("status") == "ok"
                else f"N/A ({result.get('status', row.get('status', 'missing'))})"
            )
        ratios = []
        for name in ("prims-fp8", "flashmla"):
            value = row.get("speedups", {}).get(name)
            ratios.append(f"{value:.3f}×" if value is not None else "N/A")
        lines.append("| " + " | ".join([row["id"], *timings, *ratios]) + " |")
    lines.extend(
        [
            "",
            "Both Prims paths use FP8 MMA; FlashMLA uses BF16 MMA on exact widenings of FP8 Q.",
            "FlashMLA H<64 runs physical H64 with padding outside timing. Its D512 FP8 primary uses the labeled 528B constant-scale adapter, also prepared outside timing.",
        ]
    )
    provenance = report.get("provenance", {})
    if provenance:
        lines.append(
            f"Device: {provenance['gpu']}; CUDA {provenance['cuda']}; CuTe DSL {provenance['dsl']}; FlashInfer {provenance['source_revision']}."
        )
    return "\n".join(lines) + "\n"


def main():
    """Print Markdown, or save it to the explicitly requested output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = render(json.loads(args.report.read_text()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
