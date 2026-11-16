#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Per-call GPU spans of the QSA NVFP4 suites from Nsight Systems traces.

CUDA event timestamps quantize small latencies (about 2 us on GB300). For
precise comparisons, run the bench with ``--nvtx`` under

    nsys profile -t cuda,nvtx --cuda-graph-trace=node -o REPORT python \\
        bench_q_token_kv_block_sparse_nvfp4.py --nvtx ...

and pass the ``.nsys-rep`` files here. Each NVTX range ``QSA|<case>|<backend>``
covers one backend's sampled replays. A call starts at the QSA metadata
kernel and ends with the last kernel before the next cache-eviction kernel or
metadata kernel, so it includes attention and any split-KV reduction while
the L2 eviction stays outside, as in the event timing. Several reports,
for example independent processes, are pooled per case and backend.
"""

import argparse
import csv
import io
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from bench_q_token_kv_block_sparse_nvfp4 import COMPARISONS, NVTX_PREFIX

# Kernel-name fragments: the QSA metadata kernel starts a call; PyTorch
# eviction/reduction kernels and memsets belong to the cold-L2 harness.
METADATA_KERNEL = "QTokenKvBlockSparse"
HARNESS_KERNELS = ("at::native", "[CUDA memset]", "[CUDA memcpy")


def is_metadata(name):
    """True for the QSA metadata kernel that starts every timed call."""
    return METADATA_KERNEL in name and "Metadata" in name


def extract_spans(kernels, ranges):
    """Group kernels into calls inside each QSA NVTX range.

    ``kernels`` are (start_ns, duration_ns, name) tuples; ``ranges`` are
    (name, start_ns, end_ns) tuples of projected GPU NVTX ranges. Returns
    {(case, backend): [dict(span_us, metadata_us), ...]}.
    """
    kernels = sorted(kernels)
    calls = defaultdict(list)
    marker = NVTX_PREFIX + "|"
    for name, start, end in ranges:
        # nsys prefixes the NVTX domain (":" for the default domain).
        if marker not in name:
            continue
        parts = name[name.index(marker) :].split("|")
        if len(parts) != 3 or parts[0] != NVTX_PREFIX:
            continue
        key = (parts[1], parts[2])
        inside = [k for k in kernels if start <= k[0] and k[0] + k[1] <= end]
        current = None
        for kernel in inside:
            if is_metadata(kernel[2]):
                if current:
                    calls[key].append(current)
                current = [kernel]
            elif current is not None:
                if any(tag in kernel[2] for tag in HARNESS_KERNELS):
                    calls[key].append(current)
                    current = None
                else:
                    current.append(kernel)
        if current:
            calls[key].append(current)
    spans = defaultdict(list)
    for key, groups in calls.items():
        for group in groups:
            first = group[0]
            last_end = max(k[0] + k[1] for k in group)
            spans[key].append(
                dict(span_us=(last_end - first[0]) / 1e3, metadata_us=first[1] / 1e3)
            )
    return spans


def read_csv(text):
    """Rows of an ``nsys stats --format csv`` report written to stdout."""
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(("Start", '"Start'))), 0
    )
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def load_report(path, nsys="nsys"):
    """Export kernels and projected NVTX ranges of one ``.nsys-rep`` file."""

    def stats(report):
        return subprocess.check_output(
            [nsys, "stats", "-q", "-r", report, "-f", "csv", str(path)], text=True
        )

    kernels = [
        (int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"])
        for r in read_csv(stats("cuda_gpu_trace"))
    ]
    ranges = [
        (
            r["Name"],
            int(r["Projected Start (ns)"]),
            int(r["Projected Start (ns)"]) + int(r["Projected Duration (ns)"]),
        )
        for r in read_csv(stats("nvtx_gpu_proj_trace"))
        if NVTX_PREFIX + "|" in r.get("Name", "")
    ]
    return kernels, ranges


def summarize_spans(spans):
    """Per case: mean/median span and metadata time per backend, and speedups.

    The comparison is inferred from the traced backends. Speedup = backend
    mean span / primary mean span; means are used because small-batch spans
    can be bimodal, which makes a median jump between modes.
    """
    traced = {backend for _, backend in spans}
    matches = [(n, s) for n, s in COMPARISONS.items() if traced <= set(s["backends"])]
    if not matches:
        raise ValueError(
            f"Traced backends {sorted(traced)} mix comparisons; pass traces of one"
        )
    name, spec = matches[0]
    cases = defaultdict(dict)
    for (case, backend), values in spans.items():
        cases[case][backend] = dict(
            mean_us=statistics.mean(v["span_us"] for v in values),
            median_us=statistics.median(v["span_us"] for v in values),
            metadata_mean_us=statistics.mean(v["metadata_us"] for v in values),
            calls=len(values),
        )
    rows = []
    for case in sorted(cases):
        row = dict(id=case, backends=cases[case], speedups={})
        primary = cases[case].get(spec["primary"])
        for backend, result in cases[case].items():
            if primary and backend != spec["primary"]:
                row["speedups"][backend] = result["mean_us"] / primary["mean_us"]
        rows.append(row)
    geomean = {}
    for backend in spec["backends"]:
        values = [r["speedups"][backend] for r in rows if backend in r["speedups"]]
        if values:
            geomean[backend] = math.exp(sum(map(math.log, values)) / len(values))
    return dict(comparison=name, primary=spec["primary"], cases=rows, geomean=geomean)


def render(summary):
    """Markdown table of mean spans and speedups versus the primary backend."""
    backends = COMPARISONS[summary["comparison"]]["backends"]
    others = [b for b in backends if b != summary["primary"]]
    lines = [
        "Mean GPU span per call, metadata start through the last call kernel, in us; "
        f"speedup = backend / {summary['primary']} (>1 favors it).",
        "",
        "| Case | "
        + " | ".join(f"{b} µs" for b in backends)
        + " | "
        + " | ".join(f"vs {b}" for b in others)
        + " | metadata µs | calls |",
        "|---|" + "---:|" * (len(backends) + len(others) + 2),
    ]
    for row in summary["cases"]:
        res = row["backends"]
        cells = [row["id"]]
        cells += [f"{res[b]['mean_us']:.2f}" if b in res else "N/A" for b in backends]
        cells += [
            f"{row['speedups'][b]:.3f}×" if b in row["speedups"] else "N/A"
            for b in others
        ]
        primary = res.get(summary["primary"], {})
        cells.append(f"{primary['metadata_mean_us']:.2f}" if primary else "N/A")
        cells.append(str(min(r["calls"] for r in res.values())))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(
        "\nGeomean speedup: "
        + ", ".join(f"vs {b} {v:.3f}×" for b, v in summary["geomean"].items())
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Pool one or more traces and print or save the span summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help=".nsys-rep files")
    parser.add_argument("--nsys", default="nsys", help="nsys executable")
    parser.add_argument("--output", type=Path, help="Write the JSON summary here")
    args = parser.parse_args(argv)
    spans = defaultdict(list)
    for path in args.reports:
        for key, values in extract_spans(*load_report(path, args.nsys)).items():
            spans[key].extend(values)
    if not spans:
        parser.error("No QSA NVTX ranges found; run the bench with --nvtx")
    summary = summarize_spans(spans)
    summary["reports"] = [p.name for p in args.reports]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(render(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
