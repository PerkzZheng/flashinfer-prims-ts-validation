#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Render QSA NVFP4/FP8 latencies, throughput and speedups from reports."""

import argparse
import json
from pathlib import Path

BACKENDS = ("nvfp4-fp8mma", "nvfp4-bf16mma", "fp8")


def render(reports):
    """Merge one or more reports; failed or missing rows stay visible as N/A."""
    rows, requested, sources = [], [], set()
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
    lines = [
        f"Completed {len(seen)}/{len(set(requested))} requested cases.",
        "",
        "Median microseconds; speedup = FP8-KV / NVFP4 FP8-MMA (>1 favors NVFP4).",
        "TB/s counts unique needed K/V once (NVFP4: payload + scales) plus Q and O.",
        "",
        "| Case | NVFP4 FP8-MMA µs | NVFP4 BF16-MMA µs | FP8 µs | vs FP8 | "
        "NVFP4 TB/s | NVFP4 TFLOP/s | FP8 TB/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if any(report.get("quick") for report in reports):
        lines.insert(1, "Includes `--quick` functional runs; not performance signoff.")
    for row in rows:
        backends = row.get("backends", {})
        cells = [row["id"]]
        for name in BACKENDS:
            result = backends.get(name, {})
            cells.append(
                f"{result['median_us']:.2f}"
                if result.get("status") == "ok"
                else f"N/A ({result.get('status', row.get('status', 'missing'))})"
            )
        speedup = row.get("speedups", {}).get("fp8")
        cells.append(f"{speedup:.3f}×" if speedup is not None else "N/A")
        for name, key in (
            ("nvfp4-fp8mma", "achieved_tbps"),
            ("nvfp4-fp8mma", "achieved_tflops"),
            ("fp8", "achieved_tbps"),
        ):
            value = backends.get(name, {}).get(key)
            cells.append(f"{value:.2f}" if value is not None else "N/A")
        lines.append("| " + " | ".join(cells) + " |")
    for missing in sorted(set(requested) - seen):
        lines.append(f"| {missing} | N/A (missing) | | | | | | |")
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
