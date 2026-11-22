# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Workload contract, statistics and A/B comparison for the mask-tile suite.

This module is CPU-only: it validates the declarative manifest, expands
deterministic request lengths, and compares interleaved base/candidate runs.
"""

import json
import math
import random
import statistics
from pathlib import Path

FAMILIES = ("context", "decode", "mla")
DTYPES = ("bf16", "fp8")
MASKS = ("causal", "dense")
TOLERANCE_KEYS = ("rel_l2", "max_abs")


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def expand_lengths(spec, count, name):
    """Return ``count`` lengths from an integer, a list or a seeded range.

    Ranges use Python's ``random.Random`` so every host and PyTorch version
    expands the same lengths. ``align`` rounds each length down to a multiple.
    """
    if type(spec) is int:
        lengths = [spec] * count
    elif isinstance(spec, list):
        if len(spec) != count:
            raise ValueError(f"{name} must list exactly {count} lengths")
        lengths = list(spec)
    elif isinstance(spec, dict):
        if set(spec) - {"lo", "hi", "seed", "align"}:
            raise ValueError(f"{name} accepts lo, hi, seed and align")
        lo, hi, seed = spec["lo"], spec["hi"], spec["seed"]
        align = spec.get("align", 1)
        for key, value in (("lo", lo), ("hi", hi), ("align", align)):
            _positive_int(value, f"{name}.{key}")
        if type(seed) is not int or lo >= hi:
            raise ValueError(f"{name} needs an integer seed and lo < hi")
        rng = random.Random(seed)
        lengths = [rng.randrange(lo, hi) for _ in range(count)]
        lengths = [max(x - x % align, align) for x in lengths]
    else:
        raise ValueError(
            f"{name} must be an integer, a list or a {{lo, hi, seed}} range"
        )
    for x in lengths:
        _positive_int(x, name)
    return lengths


def _check_case(case):
    family = case["family"]
    if family not in FAMILIES:
        raise ValueError(f"{case['id']}: unknown family {family!r}")
    if case["dtype"] not in DTYPES or case["mask"] not in MASKS:
        raise ValueError(f"{case['id']}: unsupported dtype or mask")
    if not isinstance(case.get("covers"), str) or not case["covers"]:
        raise ValueError(f"{case['id']}: describe the mask path in 'covers'")
    _positive_int(case["batch"], "batch")
    window = case.get("window_left", -1)
    if type(window) is not int or window < -1:
        raise ValueError(f"{case['id']}: window_left must be -1 or nonnegative")
    if window >= 0 and case["mask"] != "causal":
        raise ValueError(f"{case['id']}: sliding windows require a causal mask")
    b = case["batch"]
    if family == "context":
        if case["layout"] not in ("fixed", "ragged", "paged"):
            raise ValueError(f"{case['id']}: layout must be fixed, ragged or paged")
        if case["layout"] == "paged":
            _positive_int(case["page_size"], "page_size")
        for key in ("heads_q", "heads_kv", "head_dim"):
            _positive_int(case[key], key)
        if case["heads_q"] % case["heads_kv"]:
            raise ValueError(f"{case['id']}: heads_q must be a multiple of heads_kv")
        q_lens = expand_lengths(case["q_lens"], b, "q_lens")
        if case["layout"] == "fixed" and len(set(q_lens)) != 1:
            raise ValueError(f"{case['id']}: fixed layout needs one query length")
        if type(case.get("kv_extra", 0)) is not int or case.get("kv_extra", 0) < 0:
            raise ValueError(f"{case['id']}: kv_extra must be nonnegative")
        width = case.get("variable_window")
        if width is not None:
            _positive_int(width, "variable_window")
            if case["layout"] != "fixed" or case.get("kv_extra", 0):
                raise ValueError(f"{case['id']}: variable windows use fixed S_q=S_kv")
        if case["dtype"] != "bf16":
            raise ValueError(f"{case['id']}: context cases use BF16")
    elif family == "decode":
        for key in ("heads_q", "heads_kv", "head_dim", "queries", "page_size"):
            _positive_int(case[key], key)
        if case["heads_q"] % case["heads_kv"]:
            raise ValueError(f"{case['id']}: heads_q must be a multiple of heads_kv")
        kv = expand_lengths(case["kv_lens"], b, "kv_lens")
        if min(kv) < case["queries"]:
            raise ValueError(f"{case['id']}: every KV length must cover the queries")
    else:
        for key in ("heads_q", "page_size"):
            _positive_int(case[key], key)
        if not isinstance(case.get("packed", False), bool):
            raise ValueError(f"{case['id']}: packed must be boolean")
        if case["dtype"] != "bf16" or case.get("window_left", -1) != -1:
            raise ValueError(f"{case['id']}: MLA cases use BF16 without windows")
        q_lens = expand_lengths(case["q_lens"], b, "q_lens")
        kv = expand_lengths(case["kv_lens"], b, "kv_lens")
        if any(k < q for k, q in zip(kv, q_lens)):
            raise ValueError(f"{case['id']}: every KV length must cover the queries")
        cp = case.get("cp")
        if cp is not None:
            if not (isinstance(cp, list) and len(cp) == 3):
                raise ValueError(f"{case['id']}: cp is [world, interleave, rank]")
            world, interleave, rank = cp
            _positive_int(world, "cp world")
            _positive_int(interleave, "cp interleave")
            if type(rank) is not int or not 0 <= rank < world:
                raise ValueError(f"{case['id']}: cp rank must lie in [0, world)")
            if any(local_kv_length(k, cp) == 0 for k in kv):
                raise ValueError(
                    f"{case['id']}: every rank-local cache must be nonempty"
                )
    tol = case.get("tolerance")
    if tol is not None and (
        set(tol) != set(TOLERANCE_KEYS)
        or any(not isinstance(tol[k], (int, float)) or tol[k] <= 0 for k in tol)
    ):
        raise ValueError(f"{case['id']}: tolerance needs positive rel_l2 and max_abs")


def load_suite(path):
    """Validate a declarative workload before importing CUDA dependencies."""
    spec = json.loads(Path(path).read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported schema version")
    for key in ("name", "seed", "timing", "tolerance", "cases"):
        if key not in spec:
            raise ValueError(f"Suite is missing {key!r}")
    timing = spec["timing"]
    if set(timing) != {"warmups", "samples", "calls_per_sample"}:
        raise ValueError("timing needs warmups, samples and calls_per_sample")
    for key, value in timing.items():
        _positive_int(value, f"timing.{key}")
    tol = spec["tolerance"]
    if set(tol) != set(DTYPES):
        raise ValueError("tolerance needs bf16 and fp8 entries")
    for dtype in DTYPES:
        if set(tol[dtype]) != set(TOLERANCE_KEYS):
            raise ValueError("Each tolerance needs rel_l2 and max_abs")
    cases = spec["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Cases must be nonempty with unique IDs")
    for case in cases:
        _check_case(case)
    return spec


def case_tolerance(spec, case):
    """Return the oracle tolerance for one case."""
    return case.get("tolerance") or spec["tolerance"][case["dtype"]]


def case_seed(spec, index):
    """Return the deterministic input seed for the case at ``index``."""
    return spec["seed"] + 7919 * index


def local_kv_length(length, cp):
    """Count the keys of a global prefix owned by one block-cyclic CP rank."""
    world, interleave, rank = cp
    period = world * interleave
    remainder = length % period - rank * interleave
    return (length // period) * interleave + min(max(remainder, 0), interleave)


def owned_positions(local_length, cp):
    """Return the global positions of a rank's first ``local_length`` keys."""
    world, interleave, rank = cp
    return [
        (j // interleave) * world * interleave + rank * interleave + j % interleave
        for j in range(local_length)
    ]


def timing_summary(samples):
    """Summarize per-call device times in microseconds."""
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": list(samples),
    }


def result_summary(rows, cases):
    """Return the per-run pass status; every selected case must pass."""
    by_id = {r["case_id"]: r for r in rows}
    missing = [c["id"] for c in cases if c["id"] not in by_id]
    failed = [r["case_id"] for r in rows if r["status"] != "passed"]
    return {
        "passed": not missing and not failed,
        "cases": len(cases),
        "completed": len(rows),
        "failed": failed,
        "missing": missing,
    }


# ------------------------------------------------------------- A/B comparison
COMPARABLE_METADATA = ("gpu", "capability", "sms", "torch", "cuda", "dsl")


def _check_run(run, label):
    if run.get("schema_version") != 1 or "rows" not in run:
        raise ValueError(f"{label}: not a mask-tile result")


def compare_runs(base_runs, candidate_runs, *, max_regression, allow_output_change):
    """Compare interleaved base/candidate rounds case by case.

    ``base_runs[k]`` and ``candidate_runs[k]`` are one interleaved round on
    the same GPU. The statistic is the median over rounds of the paired
    candidate/base ratio of per-run median latency, which cancels slow drift
    between rounds. A case regresses when that median exceeds
    ``1 + max_regression``. Outputs must be bitwise identical across all
    runs of each revision and, unless ``allow_output_change``, between them.
    """
    if not base_runs or len(base_runs) != len(candidate_runs):
        raise ValueError("Need the same nonzero number of base and candidate rounds")
    runs = [("base", r) for r in base_runs] + [("candidate", r) for r in candidate_runs]
    for i, (label, run) in enumerate(runs):
        _check_run(run, f"{label} run {i}")
    first = base_runs[0]
    for label, run in runs:
        if (
            run["suite"] != first["suite"]
            or run["selected_cases"] != first["selected_cases"]
        ):
            raise ValueError(f"{label}: suite or case selection differs")
        for key in COMPARABLE_METADATA:
            if run["metadata"].get(key) != first["metadata"].get(key):
                raise ValueError(f"{label}: metadata {key!r} differs")
        if run["metadata"]["harness_sha256"] != first["metadata"]["harness_sha256"]:
            raise ValueError(f"{label}: harness differs")
    for label, group in (("base", base_runs), ("candidate", candidate_runs)):
        if len({r["metadata"]["source_commit"] for r in group}) != 1:
            raise ValueError(f"{label}: rounds come from different source commits")

    rows = []
    for case_id in first["selected_cases"]:
        per = {}
        for label, group in (("base", base_runs), ("candidate", candidate_runs)):
            per[label] = [
                next((r for r in run["rows"] if r["case_id"] == case_id), None)
                for run in group
            ]
        row = {"case_id": case_id, "issues": []}
        everything = per["base"] + per["candidate"]
        if any(r is None or r["status"] != "passed" for r in everything):
            row["issues"].append("failed")
            rows.append(row)
            continue
        if len({r["input_sha256"] for r in everything}) != 1:
            row["issues"].append("inputs differ")
        for label in ("base", "candidate"):
            if len({r["output_sha256"] for r in per[label]}) != 1:
                row["issues"].append(f"{label} output nondeterministic")
        row["outputs_identical"] = (
            per["base"][0]["output_sha256"] == per["candidate"][0]["output_sha256"]
        )
        if not row["outputs_identical"] and not allow_output_change:
            row["issues"].append("outputs differ")
        base_us = [r["timing"]["median_us"] for r in per["base"]]
        cand_us = [r["timing"]["median_us"] for r in per["candidate"]]
        ratios = [c / b for b, c in zip(base_us, cand_us)]
        row.update(
            family=per["base"][0]["family"],
            base_us=statistics.median(base_us),
            candidate_us=statistics.median(cand_us),
            ratio=statistics.median(ratios),
            ratio_min=min(ratios),
            ratio_max=max(ratios),
        )
        if row["ratio"] > 1 + max_regression:
            row["issues"].append("regression")
        rows.append(row)
    timed = [r for r in rows if "ratio" in r]
    geomean = (
        math.exp(sum(math.log(r["ratio"]) for r in timed) / len(timed))
        if timed
        else None
    )
    return {
        "passed": all(not r["issues"] for r in rows),
        "rounds": len(base_runs),
        "max_regression": max_regression,
        "allow_output_change": allow_output_change,
        "base_commit": base_runs[0]["metadata"]["source_commit"],
        "candidate_commit": candidate_runs[0]["metadata"]["source_commit"],
        "gpu": first["metadata"]["gpu"],
        "geomean_ratio": geomean,
        "rows": rows,
    }


def comparison_markdown(result):
    """Render a comparison as a Markdown report."""
    lines = [
        "# Mask-tile A/B comparison",
        "",
        f"- GPU: {result['gpu']}",
        f"- Base: `{result['base_commit']}`",
        f"- Candidate: `{result['candidate_commit']}`",
        f"- Interleaved rounds: {result['rounds']}",
        f"- Regression limit: +{result['max_regression'] * 100:.2f}% "
        "(median paired ratio)",
    ]
    if result["geomean_ratio"] is not None:
        lines.append(
            f"- Geomean candidate/base: {(result['geomean_ratio'] - 1) * 100:+.2f}%"
        )
    lines += [
        f"- Verdict: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        "| Case | Family | Base µs | Candidate µs | Δ (median) | Δ range | Outputs | Issues |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for r in result["rows"]:
        if "ratio" not in r:
            lines.append(f"| {r['case_id']} | | | | | | | {', '.join(r['issues'])} |")
            continue
        lines.append(
            f"| {r['case_id']} | {r['family']} | {r['base_us']:.2f} | "
            f"{r['candidate_us']:.2f} | {(r['ratio'] - 1) * 100:+.2f}% | "
            f"{(r['ratio_min'] - 1) * 100:+.2f}% … {(r['ratio_max'] - 1) * 100:+.2f}% | "
            f"{'identical' if r['outputs_identical'] else 'differ'} | "
            f"{', '.join(r['issues']) or '—'} |"
        )
    return "\n".join(lines) + "\n"
