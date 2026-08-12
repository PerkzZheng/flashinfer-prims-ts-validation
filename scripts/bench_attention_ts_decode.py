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

"""Benchmark public Attention-TS and TRTLLM-gen paged decode.

Each row is constructed once by the benchmark fixture module. The public
selected TS interface and public TRTLLM-gen function receive the same Q/K/V
storage, page IDs, derived sequence lengths, scales, output shape/dtype, and
FP32 reference. ``--prims-ts-interface standalone`` exercises the public
caller-workspace API; the default ``wrapper`` mode preserves historical runs.
Default hot timing captures the same number of calls in one graph per backend,
alternates graph replay order across samples, and reports per-call latency.
Paired modes gate on the ratio of the order-balanced total TS and reference
durations. Per-cycle percentiles remain diagnostics; unpaired CUPTI timing is
explicitly raw-only.
PrimTS setup and both first calls remain separate diagnostics.
Curated cases retain their original geometry. ``--matrix-sweep`` constructs a
deterministic Cartesian product for broader public-interface comparisons.

Examples
--------
Run the smallest shared row with short timing loops::

    python benchmarks/bench_attention_ts_decode.py \
        --case smoke --warmup-iters 3 --iters 20

Run every supported shared row and save machine-readable results::

    python benchmarks/bench_attention_ts_decode.py \
        --json-output /tmp/attention_ts_decode.json

Reproduce the high-resolution long-KV comparison::

    python benchmarks/bench_attention_ts_decode.py \
        --case kv2048 --case kv8192 \
        --batch-size 2 --batch-size 8 --batch-size 16 \
        --batch-size 32 --batch-size 128 \
        --calls-per-graph 100 --iters 200 \
        --json-output /tmp/attention_ts_decode_long_kv.json

List the pinned PR2265 speculative-decode catalog::

    python benchmarks/bench_attention_ts_decode.py \
        --suite pr2265 --list-cases
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.routines.attention_ts_benchmark import (  # noqa: E402, I001
    atomic_write_json as _write_json_atomic,
    cache_info_record as _cache_info_dict,
    effective_command,
    first_call as _first_call,
    git_value,
    paired_backend_order as _paired_backend_order,
    prepare_cold_l2_scrubber as _prepare_cold_l2_scrubber,
    sha256_tree as _shared_sha256_tree,
    summarize_times as _summarize_times,
    time_backend as _time_backend,
    time_paired_cold_l2_cuda_graphs as _time_paired_cold_l2_cuda_graphs,
    time_paired_cuda_graphs as _time_paired_cuda_graphs,
)

_CPU_TEST_EXPORTS = (_paired_backend_order, _summarize_times)

_TRTLLM_WORKSPACE_BYTES = 256 * 1024 * 1024
_PRIMS_TS_INTERFACES = ("wrapper", "standalone")
_SCHEMA_VERSION = 5
_DEFAULT_GAP_THRESHOLD_PERCENT = 5.0
_PAIRED_GATE_METRIC = "paired_two_order_cycles.gap_percent_total_duration_ratio"


class _MatrixCaseSpec(NamedTuple):
    """Benchmark-local case spec with explicit head and query dimensions."""

    case_id: str
    kv_lens: tuple[int, ...]
    num_qo_heads: int
    num_kv_heads: int
    page_size: int
    mask_type: str
    qkv_dtype: Any
    output_dtype: Any
    cache_form: str
    provide_out: bool
    expected_bucket: int
    seed: int
    head_dim: int
    seq_len_q: int = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the public Attention-TS paged-decode API against "
            "TRTLLM-gen on curated shared correctness rows."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        metavar="ID",
        help="Benchmark case ID to run; repeat the option. Default: all rows.",
    )
    parser.add_argument(
        "--suite",
        choices=("curated", "pr2265", "all"),
        default="curated",
        help=(
            "Case catalog to list or select: legacy curated rows, the pinned "
            "30-row PR2265 speculative catalog, or both (default: curated)."
        ),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List benchmark case IDs and exit.",
    )
    parser.add_argument(
        "--batch-size",
        action="append",
        dest="batch_sizes",
        type=int,
        metavar="B",
        help=(
            "Override a single-length case with uniform batch size B; repeat "
            "the option to sweep sizes. Ragged base cases are rejected."
        ),
    )
    parser.add_argument(
        "--matrix-sweep",
        action="store_true",
        help=(
            "Build a Cartesian matrix from --batch-size, --num-qo-heads, "
            "--num-kv-heads, --kv-len, --head-dim, and --dtype."
        ),
    )
    parser.add_argument(
        "--num-qo-heads",
        action="append",
        type=int,
        metavar="H",
        help="Query-head count for --matrix-sweep; repeat to sweep.",
    )
    parser.add_argument(
        "--num-kv-heads",
        action="append",
        type=int,
        metavar="H",
        help="KV-head count for --matrix-sweep; repeat to sweep.",
    )
    parser.add_argument(
        "--kv-len",
        action="append",
        dest="kv_lens",
        type=int,
        metavar="K",
        help="Uniform KV length for --matrix-sweep; repeat to sweep.",
    )
    parser.add_argument(
        "--head-dim",
        action="append",
        dest="head_dims",
        type=int,
        metavar="D",
        help="Head dimension for --matrix-sweep; repeat to sweep.",
    )
    parser.add_argument(
        "--dtype",
        action="append",
        dest="dtypes",
        choices=("bf16", "fp8"),
        help=(
            "Q/K/V and output dtype for --matrix-sweep; repeat to sweep. "
            "Supported values: bf16, fp8."
        ),
    )
    parser.add_argument(
        "--expect-cases",
        type=int,
        help="Fail before GPU work unless the expanded matrix has this many rows.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed row and continue the matrix sweep.",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=_DEFAULT_GAP_THRESHOLD_PERCENT,
        help=(
            "Regression threshold relative to TRTLLM-gen "
            f"(default: {_DEFAULT_GAP_THRESHOLD_PERCENT:g}%%)."
        ),
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help=(
            "Return status 2 after writing results if the selected gate metric "
            "exceeds --gap-threshold. Paired timing gates on the order-balanced "
            "total-duration ratio; CUPTI gates on the raw backend medians."
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--prims-ts-interface",
        choices=_PRIMS_TS_INTERFACES,
        default="wrapper",
        help=(
            "PrimTS public interface to benchmark: planned wrapper or direct "
            "caller-workspace standalone call (default: wrapper)."
        ),
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help=(
            "Untimed graph replays per backend in default graph mode, or "
            "untimed calls in CUPTI/cold-L2 mode (default: 10)."
        ),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help=(
            "Timed graph-replay samples per backend in default graph mode, or "
            "timed calls in CUPTI/cold-L2 mode (default: 200)."
        ),
    )
    parser.add_argument(
        "--calls-per-graph",
        type=int,
        default=100,
        help=(
            "Identical public-interface calls captured per backend graph "
            "for hot-cache timing (default: 100). Cold-L2 graph timing "
            "captures one public call and flushes before every replay."
        ),
    )
    parser.add_argument(
        "--enable-cupti",
        action="store_true",
        help=(
            "Use CUPTI timing when cupti-python is installed; default hot-cache "
            "timing uses CUDA graphs."
        ),
    )
    parser.add_argument(
        "--cold-l2-cache",
        action="store_true",
        help=(
            "Use paired one-call CUDA graphs and flush 2x L2 immediately before "
            "every replay; flush latency is excluded from the event interval."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for provenance and result JSON.",
    )
    return parser


def _load_benchmark_specs(suite: str = "curated"):
    import torch
    from benchmarks.routines.attention_ts_benchmark.fmha_fixtures import (
        ATTENTION_TS_PR2265_CASE_SPECS,
        ATTENTION_TS_SHARED_CASE_SPECS,
        AttentionTSDecodeCaseSpec,
    )

    # Keep long-KV rows benchmark-local so routine unit tests do not inherit
    # additional multi-second JIT specializations.
    long_kv_specs = (
        AttentionTSDecodeCaseSpec(
            "kv2048",
            (2048,),
            8,
            1,
            16,
            "dense",
            torch.float16,
            torch.float16,
            "combined",
            True,
            2048,
            105,
        ),
        AttentionTSDecodeCaseSpec(
            "kv8192",
            (8192,),
            8,
            1,
            16,
            "dense",
            torch.float16,
            torch.float16,
            "combined",
            True,
            8192,
            151,
        ),
        AttentionTSDecodeCaseSpec(
            "fp8-kv8192-f16",
            (8192,),
            8,
            1,
            16,
            "dense",
            torch.float8_e4m3fn,
            torch.float16,
            "combined",
            True,
            8192,
            243,
        ),
        AttentionTSDecodeCaseSpec(
            "fp8-kv8192-fp8",
            (8192,),
            8,
            1,
            16,
            "dense",
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            "combined",
            True,
            8192,
            247,
        ),
        AttentionTSDecodeCaseSpec(
            "fp8-kv8192-near-ragged-f16",
            (8192, 8191, 8190, 8189, 8188, 8187, 8186, 8185),
            8,
            1,
            16,
            "dense",
            torch.float8_e4m3fn,
            torch.float16,
            "combined",
            True,
            8192,
            251,
        ),
    )
    curated_specs = (*ATTENTION_TS_SHARED_CASE_SPECS, *long_kv_specs)
    if suite == "curated":
        return curated_specs
    if suite == "pr2265":
        return ATTENTION_TS_PR2265_CASE_SPECS
    if suite == "all":
        return (*curated_specs, *ATTENTION_TS_PR2265_CASE_SPECS)
    raise ValueError(f"unknown benchmark suite {suite!r}")


def _select_specs(
    parser: argparse.ArgumentParser,
    requested: Sequence[str] | None,
    suite: str = "curated",
):
    specs = _load_benchmark_specs(suite)
    by_id = {spec.case_id: spec for spec in specs}
    if not requested:
        return list(specs)

    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        parser.error(
            f"unknown --case value(s): {', '.join(unknown)}; "
            f"available: {', '.join(by_id)}"
        )
    # Preserve user order while rejecting accidental duplicate measurements.
    selected = []
    seen = set()
    for case_id in requested:
        if case_id not in seen:
            selected.append(by_id[case_id])
            seen.add(case_id)
    return selected


def _dedupe_positive_axis(
    parser: argparse.ArgumentParser,
    values: Sequence[int] | None,
    option: str,
) -> list[int]:
    if not values:
        parser.error(f"{option} is required with --matrix-sweep")
    invalid = sorted({value for value in values if value <= 0})
    if invalid:
        parser.error(
            f"{option} values must be positive; got "
            + ", ".join(str(value) for value in invalid)
        )
    return list(dict.fromkeys(values))


def _matrix_seed(case_id: str) -> int:
    """Derive a stable positive seed without Python's randomized hash()."""

    return (
        int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:4], "little")
        & 0x7FFFFFFF
    )


def _build_matrix_specs(parser: argparse.ArgumentParser, args, torch):
    if args.case_ids:
        parser.error("--case cannot be combined with --matrix-sweep")

    batch_sizes = _dedupe_positive_axis(parser, args.batch_sizes, "--batch-size")
    num_qo_heads = _dedupe_positive_axis(parser, args.num_qo_heads, "--num-qo-heads")
    num_kv_heads = _dedupe_positive_axis(parser, args.num_kv_heads, "--num-kv-heads")
    kv_lens = _dedupe_positive_axis(parser, args.kv_lens, "--kv-len")
    head_dims = _dedupe_positive_axis(parser, args.head_dims, "--head-dim")
    if not args.dtypes:
        parser.error("--dtype is required with --matrix-sweep")
    dtype_names = list(dict.fromkeys(args.dtypes))
    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
    }

    invalid_geometry = [
        (hq, hkv)
        for hq in num_qo_heads
        for hkv in num_kv_heads
        if hq % hkv != 0 or hq // hkv > 32
    ]
    if invalid_geometry:
        parser.error(
            "matrix head geometry requires Hq divisible by Hkv with ratio <= 32; "
            f"got {invalid_geometry}"
        )

    specs = []
    for dtype_name in dtype_names:
        dtype = dtype_by_name[dtype_name]
        for head_dim in head_dims:
            for hq in num_qo_heads:
                for hkv in num_kv_heads:
                    for kv_len in kv_lens:
                        for batch_size in batch_sizes:
                            case_id = (
                                f"matrix-{dtype_name}-b{batch_size}-hq{hq}-hkv{hkv}"
                                f"-kv{kv_len}-d{head_dim}"
                            )
                            specs.append(
                                _MatrixCaseSpec(
                                    case_id=case_id,
                                    kv_lens=(kv_len,) * batch_size,
                                    num_qo_heads=hq,
                                    num_kv_heads=hkv,
                                    page_size=16,
                                    mask_type="dense",
                                    qkv_dtype=dtype,
                                    output_dtype=dtype,
                                    cache_form="combined",
                                    provide_out=True,
                                    expected_bucket=kv_len,
                                    seed=_matrix_seed(case_id),
                                    head_dim=head_dim,
                                )
                            )

    if args.expect_cases is not None and len(specs) != args.expect_cases:
        parser.error(
            f"expanded matrix has {len(specs)} rows, expected {args.expect_cases}"
        )
    return specs


def _expand_specs_for_batch_sizes(
    parser: argparse.ArgumentParser,
    specs: Sequence[Any],
    requested_batch_sizes: Sequence[int] | None,
):
    """Expand uniform single-request rows without mutating shared test specs."""

    if not requested_batch_sizes:
        return list(specs)

    invalid = sorted({size for size in requested_batch_sizes if size <= 0})
    if invalid:
        parser.error(
            "--batch-size must be positive; got "
            + ", ".join(str(size) for size in invalid)
        )

    non_single = [spec.case_id for spec in specs if len(spec.kv_lens) != 1]
    if non_single:
        parser.error(
            "--batch-size only applies to single-length base cases; got "
            + ", ".join(non_single)
        )

    batch_sizes = []
    seen = set()
    for batch_size in requested_batch_sizes:
        if batch_size not in seen:
            batch_sizes.append(batch_size)
            seen.add(batch_size)

    expanded = []
    for spec in specs:
        kv_len = spec.kv_lens[0]
        for batch_size in batch_sizes:
            expanded.append(
                replace(
                    spec,
                    case_id=f"{spec.case_id}-b{batch_size}",
                    kv_lens=(kv_len,) * batch_size,
                )
            )
    return expanded


def _format_kv_lens(kv_lens: Sequence[int]) -> str:
    if len(kv_lens) > 4 and len(set(kv_lens)) == 1:
        return f"[{kv_lens[0]}] x {len(kv_lens)}"
    return str(list(kv_lens))


def _performance_disposition(
    raw_gap_percent: float,
    paired_two_order_cycles: dict[str, Any] | None,
    threshold_percent: float,
) -> dict[str, Any]:
    """Gate paired rows on their order-balanced total-duration ratio."""

    raw_exceeds = raw_gap_percent > threshold_percent
    if paired_two_order_cycles is None:
        return {
            "threshold_percent": threshold_percent,
            "raw_gap_percent": raw_gap_percent,
            "balanced_gap_percent": None,
            "raw_exceeds_threshold": raw_exceeds,
            "balanced_exceeds_threshold": None,
            "classification": (
                "raw-only-regression" if raw_exceeds else "raw-only-pass"
            ),
            "gate_metric": "attention_ts_gap_percent_vs_trtllm_gen",
        }

    balanced_gap_percent = float(
        paired_two_order_cycles["gap_percent_total_duration_ratio"]
    )
    balanced_exceeds = balanced_gap_percent > threshold_percent
    if balanced_exceeds:
        classification = "regression"
    elif raw_exceeds:
        classification = "raw-median-exception-balanced-pass"
    else:
        classification = "pass"
    return {
        "threshold_percent": threshold_percent,
        "raw_gap_percent": raw_gap_percent,
        "balanced_gap_percent": balanced_gap_percent,
        "raw_exceeds_threshold": raw_exceeds,
        "balanced_exceeds_threshold": balanced_exceeds,
        "classification": classification,
        "gate_metric": _PAIRED_GATE_METRIC,
    }


def _summarize_results(
    results: Sequence[dict[str, Any]], threshold_percent: float
) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "ok"]
    errors = [result for result in results if result.get("status") != "ok"]
    dispositions = [result["performance_disposition"] for result in successful]
    balanced_results = [
        result
        for result in successful
        if result["performance_disposition"]["balanced_gap_percent"] is not None
    ]
    worst_balanced = (
        max(
            balanced_results,
            key=lambda result: result["performance_disposition"][
                "balanced_gap_percent"
            ],
        )
        if balanced_results
        else None
    )
    return {
        "row_count": len(results),
        "successful_row_count": len(successful),
        "error_row_count": len(errors),
        "error_case_ids": [result["case_id"] for result in errors],
        "gap_threshold_percent": threshold_percent,
        "raw_independent_median_regression_row_count": sum(
            disposition["raw_exceeds_threshold"] for disposition in dispositions
        ),
        "order_balanced_total_duration_evaluable_row_count": len(balanced_results),
        "order_balanced_total_duration_regression_row_count": sum(
            disposition["balanced_exceeds_threshold"] is True
            for disposition in dispositions
        ),
        "raw_median_exception_row_count": sum(
            disposition["classification"] == "raw-median-exception-balanced-pass"
            for disposition in dispositions
        ),
        "gate_regression_row_count": sum(
            disposition["classification"] in ("regression", "raw-only-regression")
            for disposition in dispositions
        ),
        "worst_order_balanced_case_id": (
            None if worst_balanced is None else worst_balanced["case_id"]
        ),
        "worst_order_balanced_gap_percent": (
            None
            if worst_balanced is None
            else worst_balanced["performance_disposition"]["balanced_gap_percent"]
        ),
    }


def _reference_tolerances(spec) -> tuple[float, float]:
    import torch

    if spec.qkv_dtype == torch.float8_e4m3fn:
        if spec.output_dtype == torch.float16:
            return 4e-2, 7e-2
        return 5e-2, 7e-2
    return 1e-2, 1e-2


def _normalize_result_layout(result, case, spec, *, backend: str):
    """Return canonical [B,SQ,H,D] output without entering a timed callable."""

    seq_len_q = getattr(spec, "seq_len_q", 1)
    if seq_len_q == 1:
        if result.shape != case.reference_real.shape:
            raise ValueError(
                f"{backend} SQ1 output shape {tuple(result.shape)} does not match "
                f"{tuple(case.reference_real.shape)}"
            )
        return result

    batch_size = len(spec.kv_lens)
    canonical_shape = (
        batch_size,
        seq_len_q,
        spec.num_qo_heads,
        case.q.shape[-1],
    )
    if backend == "attention_ts":
        native_shape = canonical_shape
        if tuple(result.shape) != native_shape:
            raise ValueError(
                f"Attention-TS output shape {tuple(result.shape)} does not match "
                f"native grouped-Q shape {native_shape}"
            )
        return result
    if backend == "trtllm_gen":
        native_shape = (
            batch_size * seq_len_q,
            spec.num_qo_heads,
            case.q.shape[-1],
        )
        if tuple(result.shape) != native_shape:
            raise ValueError(
                f"TRTLLM-gen output shape {tuple(result.shape)} does not match "
                f"flattened-Q shape {native_shape}"
            )
        return result.reshape(canonical_shape)
    raise ValueError(f"unknown backend layout {backend!r}")


def _check_result(
    name: str, result, case, spec, torch, *, backend: str
) -> dict[str, float]:
    from benchmarks.routines.attention_ts_benchmark.fmha_fixtures import (
        dequantize_attention_ts_output,
    )

    normalized = _normalize_result_layout(result, case, spec, backend=backend)
    result_real = dequantize_attention_ts_output(normalized, o_scale=case.o_scale)
    if not bool(torch.isfinite(result_real).all()):
        raise AssertionError(f"{name} produced a non-finite output for {spec.case_id}")
    rtol, atol = _reference_tolerances(spec)
    torch.testing.assert_close(
        result_real,
        case.reference_real,
        rtol=rtol,
        atol=atol,
        msg=lambda message: f"{name} failed {spec.case_id}: {message}",
    )
    error = (result_real - case.reference_real).abs()
    return {
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
    }


def _validate_comparison_metadata(case, spec, block_tables, seq_lens, torch) -> None:
    """Require both backend ABIs to describe the exact same paged problem."""

    for name, tensor in (
        ("paged_kv_indptr", case.paged_kv_indptr),
        ("paged_kv_indices", case.paged_kv_indices),
        ("paged_kv_last_page_len", case.paged_kv_last_page_len),
        ("TRT block_tables", block_tables),
        ("TRT seq_lens", seq_lens),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be CUDA-resident")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    seq_len_q = getattr(spec, "seq_len_q", 1)
    expected_q_shape = (
        (len(spec.kv_lens), spec.num_qo_heads, case.q.shape[-1])
        if seq_len_q == 1
        else (
            len(spec.kv_lens),
            seq_len_q,
            spec.num_qo_heads,
            case.q.shape[-1],
        )
    )
    if tuple(case.q.shape) != expected_q_shape:
        raise ValueError(
            f"fixture Q shape {tuple(case.q.shape)} does not match {expected_q_shape}"
        )
    if seq_len_q > 1 and spec.mask_type != "causal":
        raise ValueError(
            "the TRTLLM-gen comparison supports grouped-Q rows only with "
            "bottom-right causal masking"
        )
    if case.q.shape[-1] not in (64, 128, 256):
        raise ValueError(
            "the comparison supports head dimensions 64, 128, and 256; "
            f"got {case.q.shape[-1]}"
        )

    expected_seq_lens = torch.tensor(
        spec.kv_lens, dtype=torch.int32, device=seq_lens.device
    )
    torch.testing.assert_close(seq_lens, expected_seq_lens, rtol=0, atol=0)
    page_counts = case.paged_kv_indptr[1:] - case.paged_kv_indptr[:-1]
    expected_page_counts = torch.div(
        seq_lens + spec.page_size - 1,
        spec.page_size,
        rounding_mode="floor",
    )
    torch.testing.assert_close(page_counts, expected_page_counts, rtol=0, atol=0)
    if not bool(
        (
            (case.paged_kv_last_page_len >= 1)
            & (case.paged_kv_last_page_len <= spec.page_size)
        )
        .all()
        .item()
    ):
        raise ValueError("last-page lengths must be in [1, page_size]")

    for batch_idx in range(case.q.shape[0]):
        row_begin = int(case.paged_kv_indptr[batch_idx].item())
        row_end = int(case.paged_kv_indptr[batch_idx + 1].item())
        torch.testing.assert_close(
            block_tables[batch_idx, : row_end - row_begin],
            case.paged_kv_indices[row_begin:row_end],
            rtol=0,
            atol=0,
        )
    if case.paged_kv_indices.numel() > 0:
        min_page = int(case.paged_kv_indices.min().item())
        max_page = int(case.paged_kv_indices.max().item())
        if min_page < 0 or max_page >= case.k_cache.shape[0]:
            raise ValueError("page IDs must index the shared physical K/V cache")


def _prepare_ts_call(
    case,
    spec,
    args,
    runtime,
    *,
    seq_lens,
    exact_max_kv_len: int,
    head_dim: int,
    out,
) -> dict[str, Any]:
    """Own the selected PrimTS interface, setup resources, and hot callable."""

    torch = runtime["torch"]
    cache_before_setup = runtime["cache_info"]()
    torch.cuda.synchronize(case.q.device)
    setup_start = time.perf_counter()

    if args.prims_ts_interface == "wrapper":
        wrapper = runtime["ts_wrapper_type"](kv_layout="HND")
        wrapper.plan(
            case.paged_kv_indptr,
            case.paged_kv_indices,
            case.paged_kv_last_page_len,
            spec.num_qo_heads,
            spec.num_kv_heads,
            head_dim,
            spec.page_size,
            q_data_type=spec.qkv_dtype,
            kv_data_type=spec.qkv_dtype,
            o_data_type=spec.output_dtype,
            mask_type=spec.mask_type,
            max_kv_len=exact_max_kv_len,
            seq_len_q=getattr(spec, "seq_len_q", 1),
        )
        policy = dict(wrapper._policy)

        def run_ts():
            return wrapper.run(
                case.q,
                case.paged_kv_cache,
                bmm1_scale=case.bmm1_scale,
                bmm2_scale=case.bmm2_scale,
                out=out,
            )

        setup_kind = "plan"
        workspace_bytes = None
        public_api = "flashinfer.attention.prims_ts.BatchDecodePagedTSWrapper"
    else:
        workspace_bytes = runtime["get_ts_workspace_size"](
            len(spec.kv_lens),
            spec.num_qo_heads,
            spec.num_kv_heads,
            head_dim,
            spec.page_size,
            exact_max_kv_len,
            q_dtype=spec.qkv_dtype,
            kv_dtype=spec.qkv_dtype,
            out_dtype=spec.output_dtype,
            mask_type=spec.mask_type,
            kv_layout="HND",
            device=args.device,
            seq_len_q=getattr(spec, "seq_len_q", 1),
        )
        # FMHA split counters require zero before the workspace's first use.
        ts_workspace = torch.zeros(
            workspace_bytes, dtype=torch.int8, device=case.q.device
        )
        policy = {
            "source": "auto",
            "diagnostic_available": False,
            "diagnostic_reason": (
                "the standalone public API does not expose its resolved recipe"
            ),
        }

        def run_ts():
            return runtime["standalone_ts_decode"](
                case.q,
                case.paged_kv_cache,
                ts_workspace,
                case.paged_kv_indptr,
                case.paged_kv_indices,
                seq_lens,
                exact_max_kv_len,
                bmm1_scale=case.bmm1_scale,
                bmm2_scale=case.bmm2_scale,
                out=out,
                out_dtype=spec.output_dtype,
                mask_type=spec.mask_type,
                kv_layout="HND",
                seq_len_q=getattr(spec, "seq_len_q", 1),
            )

        setup_kind = "workspace-query-and-allocation"
        public_api = "flashinfer.decode.prims_ts_batch_decode_with_kv_cache"

    torch.cuda.synchronize(case.q.device)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    cache_after_setup = runtime["cache_info"]()
    return {
        "call": run_ts,
        "interface": args.prims_ts_interface,
        "public_api": public_api,
        "setup_kind": setup_kind,
        "setup_ms": setup_ms,
        "workspace_bytes": workspace_bytes,
        "policy": policy,
        "cache_before_setup": cache_before_setup,
        "cache_after_setup": cache_after_setup,
    }


def _run_case(spec, args, runtime, cold_l2_scrubber=None) -> dict[str, Any]:
    from benchmarks.routines.attention_ts_benchmark.fmha_fixtures import (
        attention_ts_rectangular_block_tables,
        attention_ts_seq_lens_from_csr,
    )

    torch = runtime["torch"]
    make_case = runtime["make_case"]
    batch_decode_trtllm = runtime["batch_decode_trtllm"]
    bench_gpu_time = runtime["bench_gpu_time"]
    cache_info = runtime["cache_info"]

    case = make_case(
        kv_lens=spec.kv_lens,
        num_qo_heads=spec.num_qo_heads,
        num_kv_heads=spec.num_kv_heads,
        head_dim=getattr(spec, "head_dim", 128),
        seq_len_q=getattr(spec, "seq_len_q", 1),
        page_size=spec.page_size,
        qkv_dtype=spec.qkv_dtype,
        output_dtype=spec.output_dtype,
        cache_form=spec.cache_form,
        mask_type=spec.mask_type,
        device=f"cuda:{args.device}",
        seed=spec.seed,
    )
    block_tables = attention_ts_rectangular_block_tables(
        case.paged_kv_indptr, case.paged_kv_indices
    )
    seq_lens = attention_ts_seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        spec.page_size,
    )
    _validate_comparison_metadata(case, spec, block_tables, seq_lens, torch)
    exact_max_kv_len = int(seq_lens.max().item())
    head_dim = int(case.q.shape[-1])
    seq_len_q = getattr(spec, "seq_len_q", 1)
    trtllm_query = (
        case.q
        if seq_len_q == 1
        else case.q.reshape(
            len(spec.kv_lens) * seq_len_q,
            spec.num_qo_heads,
            head_dim,
        )
    )
    if trtllm_query.data_ptr() != case.q.data_ptr():
        raise ValueError("TRTLLM-gen query layout must share TS query storage")
    trtllm_workspace = torch.zeros(
        _TRTLLM_WORKSPACE_BYTES, dtype=torch.int8, device=case.q.device
    )
    trtllm_counter_bytes = runtime["get_trtllm_counter_bytes"](
        len(spec.kv_lens),
        spec.num_qo_heads,
        runtime["get_device_sm_count"](case.q.device),
    )
    trtllm_counter_buffer = torch.zeros(
        trtllm_counter_bytes, dtype=torch.uint8, device=case.q.device
    )
    ts_out_shape = tuple(case.q.shape)
    ts_out = torch.empty(ts_out_shape, dtype=spec.output_dtype, device=case.q.device)
    trtllm_out = torch.empty(
        trtllm_query.shape, dtype=spec.output_dtype, device=case.q.device
    )

    prepared_ts = _prepare_ts_call(
        case,
        spec,
        args,
        runtime,
        seq_lens=seq_lens,
        exact_max_kv_len=exact_max_kv_len,
        head_dim=head_dim,
        out=ts_out,
    )
    run_ts = prepared_ts["call"]

    def run_trtllm():
        return batch_decode_trtllm(
            trtllm_query,
            case.paged_kv_cache,
            trtllm_workspace,
            block_tables,
            seq_lens,
            exact_max_kv_len,
            case.bmm1_scale,
            case.bmm2_scale,
            -1,
            out=trtllm_out,
            out_dtype=case.output_dtype,
            sinks=None,
            kv_layout="HND",
            enable_pdl=False,
            backend="trtllm-gen",
            q_len_per_req=seq_len_q,
            o_scale=case.o_scale,
            # TRTLLM-gen applies bottom-right causal decode semantics. They
            # coincide with dense attention only for the legacy SQ1 rows.
            mask=None,
            skip_softmax_threshold_scale_factor=None,
            uses_shared_paged_kv_idx=True,
            lse=None,
            return_lse=False,
            multi_ctas_kv_counter_buffer=trtllm_counter_buffer,
        )

    ts_result, ts_first_call_ms = _first_call(run_ts, torch)
    cache_after_first_call = cache_info()
    trtllm_result, trtllm_first_call_ms = _first_call(run_trtllm, torch)
    counter_nonzero_after_first = int(torch.count_nonzero(trtllm_counter_buffer).item())
    if counter_nonzero_after_first:
        raise AssertionError(
            "TRTLLM-gen did not reset its counter buffer after first call"
        )

    ts_error = _check_result(
        "Attention-TS", ts_result, case, spec, torch, backend="attention_ts"
    )
    trtllm_error = _check_result(
        "TRTLLM-gen", trtllm_result, case, spec, torch, backend="trtllm_gen"
    )

    paired_two_order_cycles = None
    if not args.enable_cupti:
        if args.cold_l2_cache:
            ts_timing, trtllm_timing, paired_two_order_cycles = (
                _time_paired_cold_l2_cuda_graphs(
                    run_ts,
                    run_trtllm,
                    torch=torch,
                    device=args.device,
                    batch_size=case.q.shape[0],
                    warmup_replays=args.warmup_iters,
                    sample_count=args.iters,
                    scrubber=cold_l2_scrubber,
                )
            )
        else:
            ts_timing, trtllm_timing, paired_two_order_cycles = (
                _time_paired_cuda_graphs(
                    run_ts,
                    run_trtllm,
                    torch=torch,
                    batch_size=case.q.shape[0],
                    warmup_replays=args.warmup_iters,
                    sample_count=args.iters,
                    calls_per_graph=args.calls_per_graph,
                )
            )
    else:
        ts_timing = _time_backend(
            run_ts,
            bench_gpu_time=bench_gpu_time,
            batch_size=case.q.shape[0],
            warmup_iters=args.warmup_iters,
            repeat_iters=args.iters,
            enable_cupti=args.enable_cupti,
            cold_l2_cache=args.cold_l2_cache,
        )
        trtllm_timing = _time_backend(
            run_trtllm,
            bench_gpu_time=bench_gpu_time,
            batch_size=case.q.shape[0],
            warmup_iters=args.warmup_iters,
            repeat_iters=args.iters,
            enable_cupti=args.enable_cupti,
            cold_l2_cache=args.cold_l2_cache,
        )

    # Timed graph replays can expose workspace/counter reuse failures that a
    # one-shot preflight misses. Validate both timed outputs again.
    ts_post_timing_error = _check_result(
        "Attention-TS after timing",
        ts_out,
        case,
        spec,
        torch,
        backend="attention_ts",
    )
    trtllm_post_timing_error = _check_result(
        "TRTLLM-gen after timing",
        trtllm_out,
        case,
        spec,
        torch,
        backend="trtllm_gen",
    )
    counter_nonzero_after_timing = int(
        torch.count_nonzero(trtllm_counter_buffer).item()
    )
    if counter_nonzero_after_timing:
        raise AssertionError(
            "TRTLLM-gen did not reset its counter buffer after graph replay"
        )
    timed_ts_real = (
        _normalize_result_layout(ts_out, case, spec, backend="attention_ts").float()
        * case.o_scale
    )
    timed_trtllm_real = (
        _normalize_result_layout(trtllm_out, case, spec, backend="trtllm_gen").float()
        * case.o_scale
    )
    backend_difference = (timed_ts_real - timed_trtllm_real).abs()
    speedup = trtllm_timing["median_us"] / ts_timing["median_us"]
    gap_us = ts_timing["median_us"] - trtllm_timing["median_us"]
    gap_percent = gap_us / trtllm_timing["median_us"] * 100.0
    performance_disposition = _performance_disposition(
        gap_percent,
        paired_two_order_cycles,
        args.gap_threshold,
    )

    return {
        "status": "ok",
        "case_id": spec.case_id,
        "shape": {
            "batch_size": len(spec.kv_lens),
            "num_qo_heads": spec.num_qo_heads,
            "num_kv_heads": spec.num_kv_heads,
            "seq_len_q": seq_len_q,
            "head_dim": head_dim,
            "page_size": spec.page_size,
            "kv_lens": list(spec.kv_lens),
            "exact_max_kv_len": exact_max_kv_len,
            # Retain the shared-matrix field for existing JSON consumers. TS
            # compilation itself is keyed by ``exact_max_kv_len``.
            "max_kv_len_bucket": spec.expected_bucket,
            "mask_type": spec.mask_type,
            "qkv_dtype": str(spec.qkv_dtype),
            "output_dtype": str(spec.output_dtype),
            "cache_form": spec.cache_form,
            "physical_page_capacity": int(case.k_cache.shape[0]),
            "fixture_seed": spec.seed,
        },
        "attention_ts": {
            "interface": prepared_ts["interface"],
            "public_api": prepared_ts["public_api"],
            "setup_kind": prepared_ts["setup_kind"],
            "setup_ms": prepared_ts["setup_ms"],
            # Compatibility field: wrapper setup is plan(); standalone setup
            # is the public size query plus caller-workspace allocation.
            "plan_ms": prepared_ts["setup_ms"],
            "workspace_bytes": prepared_ts["workspace_bytes"],
            "first_call_ms": ts_first_call_ms,
            "compiled_during_plan": (
                prepared_ts["interface"] == "wrapper"
                and prepared_ts["cache_after_setup"].misses
                > prepared_ts["cache_before_setup"].misses
            ),
            "compiled_during_setup": (
                prepared_ts["cache_after_setup"].misses
                > prepared_ts["cache_before_setup"].misses
            ),
            "compiled_on_first_call": (
                cache_after_first_call.misses > prepared_ts["cache_after_setup"].misses
            ),
            "policy": prepared_ts["policy"],
            "cache_before_plan": _cache_info_dict(prepared_ts["cache_before_setup"]),
            "cache_after_plan": _cache_info_dict(prepared_ts["cache_after_setup"]),
            "cache_after_first_call": _cache_info_dict(cache_after_first_call),
            **ts_timing,
            **ts_error,
            "post_timing_max_abs_error": ts_post_timing_error["max_abs_error"],
            "post_timing_mean_abs_error": ts_post_timing_error["mean_abs_error"],
        },
        "trtllm_gen": {
            "first_call_ms": trtllm_first_call_ms,
            "workspace_bytes": _TRTLLM_WORKSPACE_BYTES,
            "counter_buffer_bytes": trtllm_counter_bytes,
            "counter_nonzero_after_first": counter_nonzero_after_first,
            "counter_nonzero_after_timing": counter_nonzero_after_timing,
            **trtllm_timing,
            **trtllm_error,
            "post_timing_max_abs_error": trtllm_post_timing_error["max_abs_error"],
            "post_timing_mean_abs_error": trtllm_post_timing_error["mean_abs_error"],
        },
        "comparison_contract": {
            "execution_apis_public": True,
            "implementation_state_used_only_for_diagnostics": True,
            "selected_prims_ts_interface": prepared_ts["interface"],
            "standalone_uses_explicit_seq_lens": (
                prepared_ts["interface"] == "standalone"
            ),
            "standalone_uses_queried_caller_workspace": (
                prepared_ts["interface"] == "standalone"
            ),
            "single_case_instance": True,
            "q_storage_shared_between_backends": True,
            "trt_query_is_zero_copy_view": seq_len_q > 1,
            "ts_query_layout": "B,SQ,H,D" if seq_len_q > 1 else "B,H,D",
            "trtllm_query_layout": "B*SQ,H,D" if seq_len_q > 1 else "B,H,D",
            "kv_storage_shared_between_backends": True,
            "trt_block_tables_derived_from_native_csr_ids": True,
            "trt_seq_lens_derived_from_native_csr": True,
            "same_bmm1_scale": True,
            "same_bmm2_scale": True,
            "same_output_shape_dtype": seq_len_q == 1,
            "same_semantic_output_shape_dtype": True,
            "backend_layout_normalization_outside_timing": True,
            "ts_grouped_q_output_layout": ("B,SQ,H,D" if seq_len_q > 1 else "B,H,D"),
            "trtllm_grouped_q_input_output_layout": (
                "B*SQ,H,D" if seq_len_q > 1 else "B,H,D"
            ),
            "same_fp32_reference": True,
            "bottom_right_causal_reference": spec.mask_type == "causal",
            "sq1_causal_dense_equivalence": seq_len_q == 1,
        },
        "attention_ts_speedup": speedup,
        "attention_ts_gap_us": gap_us,
        "attention_ts_gap_percent_vs_trtllm_gen": gap_percent,
        "paired_two_order_cycles": paired_two_order_cycles,
        "performance_disposition": performance_disposition,
        "backend_max_abs_difference": float(backend_difference.max().item()),
        "backend_mean_abs_difference": float(backend_difference.mean().item()),
    }


def _non_result_row(spec, *, status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "case_id": spec.case_id,
        "shape": {
            "batch_size": len(spec.kv_lens),
            "num_qo_heads": spec.num_qo_heads,
            "num_kv_heads": spec.num_kv_heads,
            "seq_len_q": getattr(spec, "seq_len_q", 1),
            "head_dim": getattr(spec, "head_dim", 128),
            "page_size": spec.page_size,
            "kv_lens": list(spec.kv_lens),
            "exact_max_kv_len": max(spec.kv_lens),
            "max_kv_len_bucket": spec.expected_bucket,
            "mask_type": spec.mask_type,
            "qkv_dtype": str(spec.qkv_dtype),
            "output_dtype": str(spec.output_dtype),
            "cache_form": spec.cache_form,
            "fixture_seed": spec.seed,
        },
        "reason": reason,
    }


def _sha256_tree(root: Path) -> str:
    return _shared_sha256_tree(root)


def _git_value(*args: str) -> str | None:
    return git_value(REPO_ROOT, *args)


def _collect_provenance(args, runtime) -> dict[str, Any]:
    torch = runtime["torch"]
    import cutlass
    import flashinfer
    from flashinfer.artifacts import ArtifactPath, CheckSumHash
    from flashinfer.jit.env import FLASHINFER_CUBIN_DIR

    properties = torch.cuda.get_device_properties(args.device)
    git_status = _git_value("status", "--porcelain", "--untracked-files=no")
    use_cuda_graph = not args.enable_cupti
    try:
        flashinfer_cubin_version = importlib.metadata.version("flashinfer-cubin")
    except importlib.metadata.PackageNotFoundError:
        flashinfer_cubin_version = None
    flashinfer_version = getattr(flashinfer, "__version__", "unknown")
    trtllm_artifact_dir = (
        Path(FLASHINFER_CUBIN_DIR) / ArtifactPath.TRTLLM_GEN_FMHA
    ).resolve()
    trtllm_manifest = trtllm_artifact_dir / "checksums.txt"
    trtllm_manifest_sha256 = (
        hashlib.sha256(trtllm_manifest.read_bytes()).hexdigest()
        if trtllm_manifest.is_file()
        else None
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "name": Path(__file__).name,
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "timing_helper": {
            "name": "benchmarks/routines/attention_ts_benchmark/timing.py",
            "sha256": hashlib.sha256(
                (
                    REPO_ROOT / "benchmarks/routines/attention_ts_benchmark/timing.py"
                ).read_bytes()
            ).hexdigest(),
        },
        "python": sys.version,
        "flashinfer_version": flashinfer_version,
        "flashinfer_git_head": _git_value("rev-parse", "HEAD"),
        "flashinfer_tracked_worktree_dirty": bool(git_status),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_git_version": torch.version.git_version,
        "gpu": {
            "device_index": args.device,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(args.device)),
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "cutlass_dsl": {
            "cutlass_version": getattr(cutlass, "__version__", "unknown"),
            "distribution_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        },
        "attention_ts": {
            "interface": args.prims_ts_interface,
            "public_api": (
                "flashinfer.decode.prims_ts_batch_decode_with_kv_cache"
                if args.prims_ts_interface == "standalone"
                else "flashinfer.attention.prims_ts.BatchDecodePagedTSWrapper"
            ),
            "integration_source_sha256": _sha256_tree(
                REPO_ROOT / "flashinfer" / "attention" / "prims_ts"
            ),
            "compile_cache": _cache_info_dict(runtime["cache_info"]()),
        },
        "trtllm_gen": {
            "public_api": "flashinfer.decode.trtllm_batch_decode_with_kv_cache",
            "backend": "trtllm-gen",
            "artifact_path": ArtifactPath.TRTLLM_GEN_FMHA,
            "artifact_manifest_sha256": CheckSumHash.TRTLLM_GEN_FMHA,
            "resolved_artifact_exists": trtllm_artifact_dir.is_dir(),
            "resolved_manifest_sha256": trtllm_manifest_sha256,
            "resolved_manifest_matches_expected": (
                trtllm_manifest_sha256 == CheckSumHash.TRTLLM_GEN_FMHA
            ),
            "flashinfer_cubin_version": flashinfer_cubin_version,
            "flashinfer_cubin_version_matches_flashinfer": (
                flashinfer_cubin_version == flashinfer_version
            ),
        },
        "timing": {
            "method": (
                "cupti"
                if args.enable_cupti
                else "paired-alternating-cold-l2-cuda-graphs"
                if args.cold_l2_cache
                else "paired-alternating-cuda-graphs"
                if use_cuda_graph
                else "cuda-event"
            ),
            "use_cuda_graph": use_cuda_graph,
            "paired_timing": use_cuda_graph,
            "alternating_replay_order": use_cuda_graph,
            "warmup_replays_or_calls": args.warmup_iters,
            "sample_count": args.iters,
            "gap_threshold_percent": args.gap_threshold,
            "regression_gate_metric": (
                "attention_ts_gap_percent_vs_trtllm_gen"
                if args.enable_cupti
                else _PAIRED_GATE_METRIC
            ),
            "regression_gate_formula": (
                None
                if args.enable_cupti
                else "(sum(attention_ts_ms) / sum(trtllm_gen_ms) - 1) * 100"
            ),
            "regression_gate_is_raw_only": args.enable_cupti,
            "cycle_ratio_percentiles_are_diagnostic_only": not args.enable_cupti,
            "raw_samples_recorded": not args.enable_cupti,
            "backend_arithmetic_means_recorded": True,
            "first_second_position_summaries_recorded": not args.enable_cupti,
            "calls_per_graph": (
                1
                if args.cold_l2_cache
                else args.calls_per_graph
                if use_cuda_graph
                else 1
            ),
            "requested_batch_sizes": args.batch_sizes,
            "requested_num_qo_heads": args.num_qo_heads,
            "requested_num_kv_heads": args.num_kv_heads,
            "requested_kv_lens": args.kv_lens,
            "requested_head_dims": args.head_dims,
            "requested_dtypes": args.dtypes,
            "suite": args.suite,
            "prims_ts_interface": args.prims_ts_interface,
            "cold_l2_cache": args.cold_l2_cache,
            "cold_l2_strategy": (
                "external-run-scoped-2x-l2-seeded-random-int8-add-before-event-v1"
                if args.cold_l2_cache and use_cuda_graph
                else None
            ),
            "l2_flush_timed": False if args.cold_l2_cache else None,
            "cold_l2_scrub": args.cold_l2_scrub_contract,
            "output_preallocated": True,
            "inputs": (
                "one fixture instance; shared Q/K/V storage, CSR page IDs, "
                "CSR-derived sequence lengths, scales, output shape/dtype, "
                "and FP32 reference"
            ),
            "metric_scope": (
                "per-call device latency from equally sized public-interface "
                "backend graphs; cold-L2 flush is outside the event interval; "
                "TS setup/compile and both first calls are reported separately"
            ),
        },
    }


def _print_result(result: dict[str, Any]) -> None:
    shape = result["shape"]
    if result.get("status", "ok") != "ok":
        print(
            f"\n[{result['case_id']}] status={result['status']} "
            f"B={shape['batch_size']} Hq/Hkv={shape['num_qo_heads']}/"
            f"{shape['num_kv_heads']} SQ={shape['seq_len_q']} "
            f"D={shape['head_dim']} "
            f"KV={shape['exact_max_kv_len']} "
            f"{shape['qkv_dtype']}->{shape['output_dtype']}"
        )
        print(f"  reason: {result['reason']}")
        return
    ts = result["attention_ts"]
    trt = result["trtllm_gen"]
    print(
        f"\n[{result['case_id']}] B={shape['batch_size']} "
        f"Hq/Hkv={shape['num_qo_heads']}/{shape['num_kv_heads']} "
        f"SQ={shape['seq_len_q']} page={shape['page_size']} "
        f"lengths={_format_kv_lens(shape['kv_lens'])} "
        f"{shape['qkv_dtype']}->{shape['output_dtype']} "
        f"cache={shape['cache_form']} exact_max={shape['exact_max_kv_len']} "
        f"shared_bucket={shape['max_kv_len_bucket']}"
    )
    print(
        f"  Attention-TS ({ts['interface']}): {ts['median_us']:9.3f} us median, "
        f"{ts['p95_us']:9.3f} us p95, setup={ts['setup_ms']:9.3f} ms, "
        f"first-run={ts['first_call_ms']:9.3f} ms "
        f"({'compile' if ts['compiled_during_setup'] or ts['compiled_on_first_call'] else 'cache hit'})"
    )
    print(f"  TS policy:     {ts['policy']}")
    print(
        f"  TRTLLM-gen:   {trt['median_us']:9.3f} us median, "
        f"{trt['p95_us']:9.3f} us p95, first={trt['first_call_ms']:9.3f} ms"
    )
    print(
        f"  timing:        {ts['timing_mode']}, samples={ts['sample_count']}, "
        f"calls/sample={ts['calls_per_sample']}, "
        f"alternating={ts['alternating_replay_order']}"
    )
    print(
        f"  TS raw gap:    {result['attention_ts_gap_us']:+.3f} us / "
        f"{result['attention_ts_gap_percent_vs_trtllm_gen']:+.1f}%  "
        f"TRT/TS={result['attention_ts_speedup']:.3f}x  "
        f"max-error(ts/trt)={ts['max_abs_error']:.6g}/{trt['max_abs_error']:.6g} "
        f"backend-max-diff={result['backend_max_abs_difference']:.6g}"
    )
    paired_cycles = result["paired_two_order_cycles"]
    if paired_cycles is None:
        print("  balanced gap:  n/a (unpaired CUPTI timing; raw-only disposition)")
    else:
        print(
            "  balanced gap:  "
            f"{paired_cycles['gap_percent_total_duration_ratio']:+.1f}% "
            "total-duration gate; cycle diagnostics "
            f"median={paired_cycles['cycle_gap_percent_median']:+.1f}%, "
            f"p95={paired_cycles['cycle_gap_percent_p95']:+.1f}% across "
            f"{paired_cycles['cycle_count']} two-order cycles"
        )
        print(
            "  paired means:  "
            f"TS={ts['arithmetic_mean_us']:.3f} us, "
            f"TRT={trt['arithmetic_mean_us']:.3f} us"
        )
    disposition = result["performance_disposition"]
    print(
        f"  disposition:   {disposition['classification']} "
        f"(gate={disposition['gate_metric']}, "
        f"threshold={disposition['threshold_percent']:g}%)"
    )


def _load_runtime() -> dict[str, Any]:
    import torch
    from benchmarks.routines.attention_ts_benchmark.fmha_fixtures import (
        make_attention_ts_decode_case,
    )
    from flashinfer.attention.prims_ts import BatchDecodePagedTSWrapper
    from flashinfer.attention.prims_ts.decode import _get_compiled_decode
    from flashinfer.decode import (
        get_prims_ts_batch_decode_workspace_size,
        prims_ts_batch_decode_with_kv_cache,
        trtllm_batch_decode_with_kv_cache,
    )
    from flashinfer.testing import bench_gpu_time
    from flashinfer.utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )

    return {
        "torch": torch,
        "ts_wrapper_type": BatchDecodePagedTSWrapper,
        "standalone_ts_decode": prims_ts_batch_decode_with_kv_cache,
        "get_ts_workspace_size": get_prims_ts_batch_decode_workspace_size,
        "batch_decode_trtllm": trtllm_batch_decode_with_kv_cache,
        "get_device_sm_count": get_device_sm_count,
        "get_trtllm_counter_bytes": get_trtllm_gen_multi_ctas_kv_counter_bytes,
        "bench_gpu_time": bench_gpu_time,
        "make_case": make_attention_ts_decode_case,
        "cache_info": _get_compiled_decode.cache_info,
        "clear_cache": _get_compiled_decode.cache_clear,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.effective_command = effective_command(Path(__file__), argv)
    if args.list_cases:
        for spec in _load_benchmark_specs(args.suite):
            print(spec.case_id)
        return 0
    if args.matrix_sweep:
        if args.suite != "curated":
            parser.error("--matrix-sweep cannot be combined with a non-curated --suite")
        import torch as parser_torch

        specs = _build_matrix_specs(parser, args, parser_torch)
    else:
        specs = _select_specs(parser, args.case_ids, args.suite)
        specs = _expand_specs_for_batch_sizes(parser, specs, args.batch_sizes)
        if args.expect_cases is not None and len(specs) != args.expect_cases:
            parser.error(f"selected {len(specs)} rows, expected {args.expect_cases}")
    if args.warmup_iters <= 0:
        parser.error("--warmup-iters must be positive")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if not args.enable_cupti and args.iters % 2:
        parser.error("--iters must be even for balanced paired timing")
    if args.calls_per_graph <= 0:
        parser.error("--calls-per-graph must be positive")
    if not math.isfinite(args.gap_threshold) or args.gap_threshold < 0:
        parser.error("--gap-threshold must be finite and non-negative")

    runtime = _load_runtime()
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error(
            f"--device {args.device} is outside [0, {torch.cuda.device_count()})"
        )
    torch.cuda.set_device(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    device_name = torch.cuda.get_device_name(args.device)
    supported_device = (capability == (10, 0) and "B200" in device_name.upper()) or (
        capability == (10, 3) and "B300" in device_name.upper()
    )
    if not supported_device:
        parser.error(
            "Attention-TS decode benchmarking requires NVIDIA B200 / SM100a "
            "or NVIDIA B300 / SM103a; "
            f"device {args.device} is {device_name} with capability {capability}"
        )

    cold_l2_scrubber = None
    if args.cold_l2_cache and not args.enable_cupti:
        cold_l2_scrubber = _prepare_cold_l2_scrubber(torch, args.device)
    args.cold_l2_scrub_contract = (
        cold_l2_scrubber.metadata() if cold_l2_scrubber is not None else None
    )
    runtime["clear_cache"]()
    provenance = _collect_provenance(args, runtime)
    print(json.dumps(provenance, indent=2))
    print(
        "\nCorrectness preflight precedes timing. "
        "Equal-size backend graphs alternate replay order; paired regressions "
        "use the total-duration ratio across complete order cycles. Median/p95 "
        "remain diagnostics, timed outputs are checked again afterward, and TS "
        "setup plus both first calls are separate. "
        f"PrimTS interface: {args.prims_ts_interface}."
    )
    results = []
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "status": "running",
        "prims_ts_interface": args.prims_ts_interface,
        "provenance": provenance,
        "run_plan": [spec.case_id for spec in specs],
        "results": results,
        "summary": _summarize_results(results, args.gap_threshold),
        "final_attention_ts_cache": None,
    }
    for spec in specs:
        try:
            result = _run_case(spec, args, runtime, cold_l2_scrubber)
        except Exception as error:
            if not args.continue_on_error:
                raise
            result = _non_result_row(
                spec,
                status="error",
                reason=f"{type(error).__name__}: {error}",
            )
        results.append(result)
        _print_result(result)
        if args.matrix_sweep:
            torch.cuda.empty_cache()
        payload["summary"] = _summarize_results(results, args.gap_threshold)
        if args.json_output is not None:
            payload["final_attention_ts_cache"] = _cache_info_dict(
                runtime["cache_info"]()
            )
            _write_json_atomic(args.json_output, payload)

    payload["status"] = "complete"
    payload["summary"] = _summarize_results(results, args.gap_threshold)
    payload["final_attention_ts_cache"] = _cache_info_dict(runtime["cache_info"]())
    if args.json_output is not None:
        _write_json_atomic(args.json_output, payload)
        print(f"\nWrote {args.json_output}")
    summary = payload["summary"]
    print(
        "\nPerformance summary: "
        f"raw regressions={summary['raw_independent_median_regression_row_count']}, "
        "order-balanced total-duration regressions="
        f"{summary['order_balanced_total_duration_regression_row_count']}, "
        f"raw-median exceptions={summary['raw_median_exception_row_count']}, "
        f"gate regressions={summary['gate_regression_row_count']}, "
        f"errors={summary['error_row_count']}."
    )
    if summary["error_row_count"]:
        return 1
    if args.fail_on_regression and summary["gate_regression_row_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
