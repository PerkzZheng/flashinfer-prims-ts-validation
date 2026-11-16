# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Workload contracts and report validation for dense NVFP4 decode campaigns."""

import itertools
import json
import math
import statistics
from pathlib import Path

BACKENDS = ("prims_nvfp4", "prims_fp8", "trt_fp8", "trt_nvfp4")
POLICY_FIELDS = (
    "use_keeps_mma_ab",
    "tile_size_q",
    "tile_size_kv",
    "num_insts_kv",
    "groups_tokens_heads_q",
    "q_tokens_per_cta",
    "use_split_kv",
    "splits_kv",
    "max_splits_kv",
    "use_separate_reduction_kernel",
    "use_persistent_scheduler",
    "threads_per_cta",
    "q_stages",
    "kv_stages",
    "transformed_kv_stages",
    "supports_auto_grouped_keeps",
    "use_fp8_qkv",
    "use_nvfp4_kv",
)


def load_suite(path):
    """Validate a declarative workload before importing CUDA dependencies."""
    spec = json.loads(Path(path).read_text())
    if spec["schema_version"] != 1:
        raise ValueError("Unsupported schema version")
    if spec["fixture"] not in ("independent-issue", "matched-nvfp4", "normal-fp8"):
        raise ValueError("Unknown input fixture")
    if spec["timing"] not in ("hot-graph", "cold-cupti"):
        raise ValueError("Unknown timing protocol")
    if spec["mask"] not in ("causal", "dense-q1-causal-multi"):
        raise ValueError("Unknown masking contract")
    if not isinstance(spec["planned_lengths"], bool):
        raise ValueError("planned_lengths must be boolean")
    if not spec["backends"] or len(set(spec["backends"])) != len(spec["backends"]):
        raise ValueError("Backend list must be nonempty and unique")
    if any(b not in BACKENDS for b in spec["backends"]):
        raise ValueError("Unknown backend")
    if spec["fixture"] == "normal-fp8" and any(
        b.endswith("nvfp4") for b in spec["backends"]
    ):
        raise ValueError("Normal FP8 fixtures cannot feed NVFP4 backends")
    cases = spec["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Cases must be nonempty with unique IDs")
    for c in cases:
        for k in (
            "batch",
            "queries",
            "head_dim",
            "heads_q",
            "heads_kv",
            "kv_len",
            "page_size",
        ):
            if type(c[k]) is not int or c[k] <= 0:
                raise ValueError(f"{k} must be a positive integer")
        if c["heads_q"] % c["heads_kv"] or c["head_dim"] % 64:
            raise ValueError("Require integral GQA and head dimension divisible by 64")
        if (
            c["kv_len"] % c["page_size"]
            or c["page_size"] % 4
            or c["queries"] > c["kv_len"]
        ):
            raise ValueError(
                "Require complete pages, four-row V-scale groups, and Q <= KV"
            )
        if c["out_dtype"] not in ("fp16", "bf16", "fp8"):
            raise ValueError("Unsupported output dtype")
    return spec


def timing_summary(samples):
    """Reject absent/nonfinite measurements instead of treating failures as wins."""
    if not samples or any(not math.isfinite(t) or t <= 0 for t in samples):
        raise ValueError("Timing samples must be finite positive microseconds")
    return {
        "best_us": min(samples),
        "median_us": statistics.median(samples),
        "samples_us": samples,
    }


def result_summary(rows, cases, backends):
    """Require exactly one successful execution per requested case/backend."""
    expected = {(c["id"], b) for c in cases for b in backends}
    actual = [(r["case"]["id"], r["backend"]) for r in rows]
    failed = [list(k) for k, r in zip(actual, rows) if r["status"] != "ok"]
    complete = (
        bool(expected) and len(actual) == len(expected) and set(actual) == expected
    )
    return {
        "expected": len(expected),
        "completed": len(rows),
        "failed": failed,
        "complete": complete,
        "passed": complete and not failed,
    }


def policy_cases():
    """Generate the review's FP8 scope probes and unchanged NVFP4 controls."""
    cases = []
    for out, b, q, ratio, kv, page, scope in itertools.product(
        ["fp16", "bf16"],
        [1, 32, 128],
        [1, 4, 8],
        [8, 16, 32, 64],
        [8192, 65536],
        [32, 64],
        ["fixed", "sinks", "packed", "packed_sinks"],
    ):
        cases.append(("fp8", 128, out, b, q, 1, ratio, kv, page, scope, False))
    for dim, out, b, q, hkv, ratio, kv in itertools.product(
        [64, 128, 256],
        ["fp16", "bf16"],
        [16, 64, 256],
        [2, 4, 8],
        [1, 4],
        [8, 16],
        [16384, 32768, 65536],
    ):
        cases.append(("fp8", dim, out, b, q, hkv, ratio, kv, 64, "fixed", False))
    for out, q, scope in itertools.product(
        ["fp16", "bf16"],
        [1, 4, 8],
        ["fixed", "sinks", "packed", "packed_sinks"],
    ):
        cases.append(("fp8", 128, out, 1, q, 1, 8, 8192, 64, scope, True))
    for dim, out, b, q, kv, page in itertools.product(
        [128, 256, 512],
        ["fp16", "bf16"],
        [8, 128],
        [1, 6, 8, 9],
        [8192, 65536],
        [64, 128],
    ):
        cases.append(("nvfp4", dim, out, b, q, 2, 8, kv, page, "fixed", False))
    return cases
