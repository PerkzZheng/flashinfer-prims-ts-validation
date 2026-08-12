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
"""Compare the 64 FP8 causal-context rows against TRTLLM-Gen.

This standalone signoff driver reuses the matrix and input construction from
``bench_attention_ts_context.py``.  Both kernels are reached through public
FlashInfer Python interfaces, consume identical FP8 E4M3 Q/K/V data, and
produce BF16 output.  Every row uses bottom-right causal masking.  Paged rows
use HND page-size-32 caches with shuffled, nonidentity physical page IDs and
poisoned guard pages.

Performance is measured with alternating CUDA-graph replays.  A stream-ordered
write to a buffer twice the size of L2 precedes every individually timed replay
and is excluded from the CUDA-event interval.  The first backend alternates at
sample granularity and round boundaries, and the primary reported gap is the
ratio of total TS duration to total TRTLLM-Gen duration across all samples.
The median paired ratio and order-conditioned distributions remain diagnostics.

The full matrix is::

    D          = 128, 256
    Hq         = 32
    Hkv        = 32, 4
    batch      = 1, 4
    (Sq, Skv)  = (1024, 1024), (4096, 4096), (16384, 16384),
                 (256, 4096)
    layout     = separate_qkv, paged_kv (page size 32)
    Q/K/V      = FP8 E4M3
    output     = BF16
    mask       = bottom-right causal

List or validate selections without touching CUDA::

    python scripts/bench_attention_ts_context_vs_trtllm_gen.py \
        --repo-root /path/to/flashinfer --list

    python scripts/bench_attention_ts_context_vs_trtllm_gen.py \
        --repo-root /path/to/flashinfer --dry-run \
        --head-dim 256 --batch-size 4 --seq-len-q 256 --seq-len-kv 4096

Run a resumable report after installing CUTLASS DSL 4.7.0::

    python scripts/bench_attention_ts_context_vs_trtllm_gen.py \
        --repo-root /path/to/flashinfer \
        --output /tmp/context-fp8-causal-vs-trtllm-gen.json --resume
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from attention_ts_decode_timing import (
    ColdL2Scrubber,
    prepare_cold_l2_scrubber,
)

OUTPUT_DTYPE = torch.bfloat16
FP8_DTYPE_NAME = "float8_e4m3fn"
PAGE_SIZE = 32
WORKSPACE_BYTES = 256 * 1024 * 1024
FULL_COMPARE_CHUNK_ELEMENTS = 8 * 1024 * 1024
FULL_COMPARE_RELATIVE_L2_LIMIT = 0.1
SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
SCHEMA_VERSION = 2
PRIMARY_GATE_METRIC = "performance.ts_gap_percent_order_balanced_total_duration"
PRIMARY_GATE_FORMULA = "(sum(ts_sample_ms) / sum(trtllm_gen_sample_ms) - 1) * 100"

EXPECTED_WHEEL_VERSION = os.environ.get(
    "FLASHINFER_EXPECTED_CUTLASS_DSL_VERSION", "4.7.0"
)
EXPECTED_WHEEL_SHA256 = os.environ.get("FLASHINFER_EXPECTED_CUTLASS_DSL_WHEEL_SHA256")
EXPECTED_WHEEL_PIP_REPORT = os.environ.get("FLASHINFER_CUTLASS_DSL_PIP_REPORT")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Runner:
    """Own one planned public wrapper and its allocation-free launch."""

    def __init__(
        self,
        owner: object,
        launch: Callable[[], torch.Tensor],
        output: torch.Tensor,
        retained: Sequence[object] = (),
        counter_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        self.owner = owner
        self.launch = launch
        self.output = output
        self.retained = tuple(retained)
        self.counter_buffer = counter_buffer


def _counter_nonzero(runner: _Runner) -> Optional[int]:
    if runner.counter_buffer is None:
        return None
    return int(torch.count_nonzero(runner.counter_buffer).item())


def _json_compatible(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if isinstance(value, (Path, torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    raise TypeError(
        "TS policy contains a non-JSON-serializable value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _selected_ts_policy(owner: object) -> dict[str, object]:
    """Capture implementation policy metadata only as a best-effort diagnostic."""

    policy = getattr(owner, "_policy", None)
    if policy is None:
        return {"available": False, "reason": "wrapper exposes no _policy"}
    try:
        serialized = _json_compatible(dict(policy))
        if not isinstance(serialized, dict):
            raise TypeError("planned TS wrapper _policy is not a mapping")
        json.dumps(serialized, allow_nan=False)
    except Exception as error:  # noqa: BLE001 - diagnostic must never fail a run
        return {
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
        }
    return {"available": True, "value": serialized}


def _plan_ts(inputs) -> _Runner:
    from flashinfer.attention.prims_ts import (
        BatchPrefillPagedTSWrapper,
        BatchPrefillTSWrapper,
    )

    output = torch.empty_like(inputs.q, dtype=OUTPUT_DTYPE)
    case = inputs.case
    if case.qkv_layout == "separate_qkv":
        if inputs.kv_indptr is None:
            raise AssertionError("separate_qkv input has no KV indptr")
        wrapper = BatchPrefillTSWrapper()
        wrapper.plan(
            inputs.q,
            inputs.k,
            inputs.v,
            qo_indptr=inputs.qo_indptr,
            kv_indptr=inputs.kv_indptr,
            mask_type="causal",
            out_dtype=OUTPUT_DTYPE,
        )
    else:
        if (
            inputs.paged_kv_indptr is None
            or inputs.paged_kv_indices is None
            or inputs.paged_kv_last_page_len is None
        ):
            raise AssertionError("paged_kv input metadata is incomplete")
        wrapper = BatchPrefillPagedTSWrapper("HND")
        wrapper.plan(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.qo_indptr,
            inputs.paged_kv_indptr,
            inputs.paged_kv_indices,
            inputs.paged_kv_last_page_len,
            page_size=PAGE_SIZE,
            mask_type="causal",
            out_dtype=OUTPUT_DTYPE,
        )

    def launch() -> torch.Tensor:
        return wrapper.run(inputs.q, inputs.k, inputs.v, out=output)

    return _Runner(wrapper, launch, output)


def _padded_block_tables(inputs) -> torch.Tensor:
    if inputs.paged_kv_indptr_host is None or inputs.paged_kv_indices is None:
        raise AssertionError("paged_kv input metadata is incomplete")
    page_counts = tuple(
        inputs.paged_kv_indptr_host[index + 1] - inputs.paged_kv_indptr_host[index]
        for index in range(inputs.case.batch_size)
    )
    result = torch.zeros(
        (inputs.case.batch_size, max(page_counts)),
        dtype=torch.int32,
        device=inputs.q.device,
    )
    for batch_index, page_count in enumerate(page_counts):
        begin = inputs.paged_kv_indptr_host[batch_index]
        end = begin + page_count
        result[batch_index, :page_count].copy_(inputs.paged_kv_indices[begin:end])
    return result


def _plan_trtllm_gen(inputs) -> _Runner:
    import flashinfer

    case = inputs.case
    output = torch.empty_like(inputs.q, dtype=OUTPUT_DTYPE)
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=inputs.q.device)
    sm_scale = 1.0 / math.sqrt(case.head_dim)
    if case.qkv_layout == "separate_qkv":
        if inputs.kv_indptr is None:
            raise AssertionError("separate_qkv input has no KV indptr")
        seq_lens = torch.tensor(
            inputs.kv_lengths, dtype=torch.int32, device=inputs.q.device
        )

        def launch() -> torch.Tensor:
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=inputs.q,
                key=inputs.k,
                value=inputs.v,
                workspace_buffer=workspace,
                seq_lens=seq_lens,
                max_q_len=case.max_seq_len_q,
                max_kv_len=case.max_seq_len_kv,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                o_sf_scale=-1.0,
                batch_size=case.batch_size,
                window_left=-1,
                cum_seq_lens_q=inputs.qo_indptr,
                cum_seq_lens_kv=inputs.kv_indptr,
                enable_pdl=True,
                is_causal=True,
                return_lse=False,
                out=output,
                backend="trtllm-gen",
            )

        return _Runner(
            workspace,
            launch,
            output,
            retained=(workspace, seq_lens),
        )

    if (
        inputs.paged_kv_indptr is None
        or inputs.paged_kv_indices is None
        or inputs.paged_kv_last_page_len is None
    ):
        raise AssertionError("paged_kv input metadata is incomplete")
    qo_buf = torch.empty_like(inputs.qo_indptr)
    kv_indptr_buf = torch.empty_like(inputs.paged_kv_indptr)
    kv_indices_buf = torch.empty_like(inputs.paged_kv_indices)
    last_page_len_buf = torch.empty_like(inputs.paged_kv_last_page_len)
    block_tables = _padded_block_tables(inputs)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "HND",
        use_cuda_graph=True,
        qo_indptr_buf=qo_buf,
        paged_kv_indptr_buf=kv_indptr_buf,
        paged_kv_indices_buf=kv_indices_buf,
        paged_kv_last_page_len_buf=last_page_len_buf,
        backend="trtllm-gen",
    )
    wrapper.plan(
        inputs.qo_indptr,
        inputs.paged_kv_indptr,
        inputs.paged_kv_indices,
        inputs.paged_kv_last_page_len,
        case.num_heads_q,
        case.num_heads_kv,
        case.head_dim,
        PAGE_SIZE,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=inputs.q.dtype,
        kv_data_type=inputs.q.dtype,
        o_data_type=OUTPUT_DTYPE,
        block_tables=block_tables,
    )
    counter_buffer = getattr(wrapper, "_trtllm_gen_multi_ctas_kv_counter_buffer", None)
    if counter_buffer is not None and not isinstance(counter_buffer, torch.Tensor):
        raise AssertionError("paged TRTLLM-Gen counter buffer is not a tensor")

    def launch() -> torch.Tensor:
        return wrapper.run(
            inputs.q,
            (inputs.k, inputs.v),
            out=output,
            enable_pdl=True,
        )

    return _Runner(
        wrapper,
        launch,
        output,
        retained=(
            workspace,
            block_tables,
            qo_buf,
            kv_indptr_buf,
            kv_indices_buf,
            last_page_len_buf,
            counter_buffer,
        ),
        counter_buffer=counter_buffer,
    )


def _check_backend_accuracy(helper, inputs, output: torch.Tensor) -> dict[str, object]:
    original = inputs.out
    inputs.out = output
    try:
        return helper._check_accuracy(inputs)
    finally:
        inputs.out = original


@torch.inference_mode()
def _compare_outputs(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    limit: float = FULL_COMPARE_RELATIVE_L2_LIMIT,
) -> dict[str, object]:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        raise AssertionError(
            f"backend output mismatch: {lhs.shape}/{lhs.dtype} vs {rhs.shape}/{rhs.dtype}"
        )
    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    error_sq = 0.0
    reference_sq = 0.0
    max_abs = 0.0
    chunks = 0
    for begin in range(0, lhs_flat.numel(), FULL_COMPARE_CHUNK_ELEMENTS):
        lhs_chunk = lhs_flat[begin : begin + FULL_COMPARE_CHUNK_ELEMENTS].float()
        rhs_chunk = rhs_flat[begin : begin + FULL_COMPARE_CHUNK_ELEMENTS].float()
        difference = lhs_chunk - rhs_chunk
        if not bool(torch.isfinite(difference).all().item()):
            raise AssertionError("backend output difference contains nonfinite values")
        error_sq += float(torch.sum(difference * difference).item())
        reference_sq += float(torch.sum(rhs_chunk * rhs_chunk).item())
        max_abs = max(max_abs, float(difference.abs().max().item()))
        chunks += 1
    relative_l2 = math.sqrt(error_sq) / max(math.sqrt(reference_sq), 1e-6)
    if relative_l2 > limit:
        raise AssertionError(
            f"full TS/TRTLLM-Gen relative L2 {relative_l2:.6g} exceeds {limit:.6g}"
        )
    return {
        "method": "full_chunked_ts_minus_trtllm_gen",
        "chunks": chunks,
        "chunk_elements": FULL_COMPARE_CHUNK_ELEMENTS,
        "relative_l2": relative_l2,
        "relative_l2_limit": limit,
        "max_abs_error": max_abs,
    }


def _capture_graph(runner: _Runner, device: torch.device):
    current_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            runner.launch()
    capture_stream.synchronize()
    current_stream.wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        runner.launch()
    torch.cuda.synchronize(device)
    return graph, capture_stream


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sample_summary(values: Sequence[float]) -> dict[str, float | int]:
    median = statistics.median(values)
    absolute_deviations = [abs(value - median) for value in values]
    return {
        "count": len(values),
        "min": min(values),
        "p10": _percentile(values, 0.10),
        "median": median,
        "mean": statistics.fmean(values),
        "p90": _percentile(values, 0.90),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "mad": statistics.median(absolute_deviations),
    }


def _duration_summary_ms(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize backend durations with explicit additive statistics."""

    return {
        **_sample_summary(values),
        "total_duration_ms": float(sum(values)),
        "arithmetic_mean_ms": float(statistics.fmean(values)),
    }


def _summarize_paired_samples(
    samples_by_backend: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, object]:
    """Build the aggregate gate and paired/order diagnostics from raw samples."""

    required_backends = {"ts", "trtllm_gen"}
    if set(samples_by_backend) != required_backends:
        raise ValueError("paired samples must contain exactly TS and TRTLLM-Gen")
    ts_rounds = samples_by_backend["ts"]
    trt_rounds = samples_by_backend["trtllm_gen"]
    if len(ts_rounds) != len(trt_rounds) or not ts_rounds:
        raise ValueError("paired samples require equal nonempty backend rounds")

    flattened = {"ts": [], "trtllm_gen": []}
    paired_ratios: list[float] = []
    ratios_by_order: dict[str, list[float]] = {
        "ts_first": [],
        "trtllm_gen_first": [],
    }
    backend_by_position: dict[str, dict[str, list[float]]] = {
        "ts": {"first": [], "second": []},
        "trtllm_gen": {"first": [], "second": []},
    }
    for round_index, (ts_values, trt_values) in enumerate(
        zip(ts_rounds, trt_rounds, strict=True)
    ):
        if len(ts_values) != len(trt_values) or not ts_values:
            raise ValueError(
                "paired samples require equal nonempty backend samples per round"
            )
        for sample_index, (ts_value_raw, trt_value_raw) in enumerate(
            zip(ts_values, trt_values, strict=True)
        ):
            ts_value = float(ts_value_raw)
            trt_value = float(trt_value_raw)
            if (
                not math.isfinite(ts_value)
                or not math.isfinite(trt_value)
                or ts_value <= 0.0
                or trt_value <= 0.0
            ):
                raise ValueError("paired latency samples must be finite and positive")
            flattened["ts"].append(ts_value)
            flattened["trtllm_gen"].append(trt_value)
            ratio = ts_value / trt_value
            paired_ratios.append(ratio)
            ts_first = (round_index + sample_index) % 2 == 0
            order = "ts_first" if ts_first else "trtllm_gen_first"
            ratios_by_order[order].append(ratio)
            backend_by_position["ts"]["first" if ts_first else "second"].append(
                ts_value
            )
            backend_by_position["trtllm_gen"]["second" if ts_first else "first"].append(
                trt_value
            )

    sample_count = len(paired_ratios)
    if sample_count < 2 or sample_count % 2:
        raise ValueError(
            "paired timing requires an even sample count for balanced order positions"
        )
    if any(
        len(position_values) != sample_count // 2
        for backend_positions in backend_by_position.values()
        for position_values in backend_positions.values()
    ):
        raise ValueError("paired timing did not balance first and second positions")

    ts_total_ms = float(sum(flattened["ts"]))
    trt_total_ms = float(sum(flattened["trtllm_gen"]))
    total_duration_ratio = ts_total_ms / trt_total_ms
    return {
        "order_balanced_total_duration": {
            "method": "alternating-order-balanced-total-duration-ratio",
            "gate_metric": PRIMARY_GATE_METRIC,
            "gate_formula": PRIMARY_GATE_FORMULA,
            "sample_count_per_backend": sample_count,
            "ts_total_duration_ms": ts_total_ms,
            "trtllm_gen_total_duration_ms": trt_total_ms,
            "ts_arithmetic_mean_ms": float(statistics.fmean(flattened["ts"])),
            "trtllm_gen_arithmetic_mean_ms": float(
                statistics.fmean(flattened["trtllm_gen"])
            ),
            "ts_latency_over_trtllm_gen": total_duration_ratio,
            "ts_gap_percent": (total_duration_ratio - 1.0) * 100.0,
            "backend_order_conditioned_ms": {
                backend: {
                    position: _duration_summary_ms(position_values)
                    for position, position_values in positions.items()
                }
                for backend, positions in backend_by_position.items()
            },
        },
        "paired_ts_latency_over_trtllm_gen": {
            "interpretation": ">1 means TRTLLM-Gen is faster",
            "diagnostic_only": True,
            "all_samples": _sample_summary(paired_ratios),
            "by_order": {
                name: _sample_summary(values)
                for name, values in ratios_by_order.items()
            },
        },
    }


def _time_pair_cold_l2(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    device: torch.device,
    iterations: int,
    scrubber: ColdL2Scrubber,
    round_index: int,
) -> dict[str, list[float]]:
    stream = torch.cuda.current_stream(device)
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in graphs
    }
    torch.cuda.synchronize(device)
    for sample_index in range(iterations):
        names = (
            ("ts", "trtllm_gen")
            if (round_index + sample_index) % 2 == 0
            else ("trtllm_gen", "ts")
        )
        for name in names:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            scrubber.enqueue()
            start.record(stream)
            graphs[name].replay()
            end.record(stream)
            events[name].append((start, end))
    torch.cuda.synchronize(device)
    return {
        name: [float(start.elapsed_time(end)) for start, end in pairs]
        for name, pairs in events.items()
    }


@torch.inference_mode()
def _benchmark_pair(
    ts_runner: _Runner,
    trt_runner: _Runner,
    *,
    device: torch.device,
    scrubber: ColdL2Scrubber,
    rounds: int,
    iterations: int,
    warmup: int,
) -> dict[str, object]:
    if rounds <= 1 or iterations <= 0 or warmup < 0 or rounds * iterations % 2:
        raise ValueError(
            "rounds must exceed one; iterations positive; warmup nonnegative; "
            "and rounds * iterations even for balanced order positions"
        )
    if scrubber.device_index != device.index:
        raise ValueError(
            f"cold-L2 scrubber belongs to cuda:{scrubber.device_index}, not {device}"
        )
    ts_graph, ts_stream = _capture_graph(ts_runner, device)
    trt_graph, trt_stream = _capture_graph(trt_runner, device)
    graphs = {"ts": ts_graph, "trtllm_gen": trt_graph}

    for index in range(warmup):
        names = ("ts", "trtllm_gen") if index % 2 == 0 else ("trtllm_gen", "ts")
        for name in names:
            graphs[name].replay()
    torch.cuda.synchronize(device)

    samples_by_backend: dict[str, list[list[float]]] = {
        "ts": [],
        "trtllm_gen": [],
    }
    for round_index in range(rounds):
        samples = _time_pair_cold_l2(
            graphs,
            device=device,
            iterations=iterations,
            scrubber=scrubber,
            round_index=round_index,
        )
        for name, values in samples.items():
            samples_by_backend[name].append(values)

    ts_runner.output.fill_(float("nan"))
    scrubber.enqueue()
    ts_graph.replay()
    trt_runner.output.fill_(float("nan"))
    scrubber.enqueue()
    trt_graph.replay()
    torch.cuda.synchronize(device)
    counter_nonzero_after_timing = _counter_nonzero(trt_runner)
    if counter_nonzero_after_timing:
        raise AssertionError(
            "TRTLLM-Gen did not reset its counter buffer after graph replay"
        )

    result: dict[str, object] = {
        "timing": (
            "alternating CUDA events around individually L2-flushed CUDA-graph replay"
        ),
        "compile_plan_timed": False,
        "graph_capture_timed": False,
        "warmup_timed": False,
        "cache_state": "cold L2 before every measured replay",
        "l2_cache_size_bytes": scrubber.l2_cache_bytes,
        "l2_flush": {
            "enabled": True,
            **scrubber.metadata(),
            "event_order": "flush, start event, one graph replay, end event",
            "frequency": "before every measured replay",
        },
        "rounds": rounds,
        "iterations_per_round": iterations,
        "warmup_replays_per_backend": warmup,
        "backend_order": "alternated per sample and inverted at round boundaries",
        "post_timing_outputs_poisoned_and_replayed": True,
    }
    for name, by_round in samples_by_backend.items():
        flattened = [sample for values in by_round for sample in values]
        round_medians = [statistics.median(values) for values in by_round]
        result[name] = {
            "sample_ms_by_round": by_round,
            "sample_summary_ms": _sample_summary(flattened),
            "total_duration_ms": float(sum(flattened)),
            "arithmetic_mean_ms": float(statistics.fmean(flattened)),
            "round_median_ms": round_medians,
            "median_of_round_medians_ms": statistics.median(round_medians),
            "median_ms": statistics.median(flattened),
        }
    counter_buffer = trt_runner.counter_buffer
    result["trtllm_gen"].update(
        {
            "counter_buffer_bytes": (
                counter_buffer.numel() * counter_buffer.element_size()
                if counter_buffer is not None
                else 0
            ),
            "counter_nonzero_after_timing": counter_nonzero_after_timing,
        }
    )

    result.update(_summarize_paired_samples(samples_by_backend))
    result["_retained_graph_objects"] = (
        ts_graph,
        trt_graph,
        ts_stream,
        trt_stream,
        scrubber,
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_fingerprint(repo: Path) -> dict[str, object]:
    """Hash all local runtime/JIT sources that can affect either backend.

    The result deliberately follows current worktree contents rather than a
    handpicked tracked-file list.  This makes resume reject stale rows after a
    dirty edit to an imported helper or CUDA/JIT source that does not change
    Git HEAD.
    """

    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    roots = ("flashinfer", "csrc", "include")
    for root_name in roots:
        root = repo / root_name
        if not root.is_dir():
            raise RuntimeError(f"source fingerprint root is missing: {root}")
        paths = sorted(
            (
                path
                for path in root.rglob("*")
                if "__pycache__" not in path.parts and path.suffix != ".pyc"
            ),
            key=lambda path: path.relative_to(repo).as_posix(),
        )
        for path in paths:
            relative = path.relative_to(repo).as_posix().encode()
            if path.is_symlink():
                payload = os.readlink(path).encode()
                digest.update(b"L\0" + relative + b"\0" + payload + b"\0")
                file_count += 1
                byte_count += len(payload)
                continue
            if not path.is_file():
                continue
            digest.update(b"F\0" + relative + b"\0")
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    byte_count += len(chunk)
            digest.update(b"\0")
            file_count += 1
    return {
        "roots": list(roots),
        "excluded": ["**/__pycache__/**", "**/*.pyc"],
        "file_count": file_count,
        "byte_count": byte_count,
        "sha256": digest.hexdigest(),
    }


def _distribution_provenance(name: str) -> dict[str, object]:
    """Record installed-package identity without hashing large binary payloads."""

    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as error:
        return {"name": name, "error": f"{type(error).__name__}: {error}"}
    metadata_sha256 = {}
    for metadata_name in ("METADATA", "direct_url.json", "RECORD"):
        content = distribution.read_text(metadata_name)
        if content is not None:
            metadata_sha256[metadata_name] = hashlib.sha256(
                content.encode()
            ).hexdigest()
    return {
        "name": name,
        "version": distribution.version,
        "metadata_sha256": metadata_sha256,
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _verify_cutlass_dsl() -> dict[str, object]:
    distribution = importlib.metadata.distribution("nvidia-cutlass-dsl")
    base_version = distribution.version.partition("+")[0]
    if base_version != EXPECTED_WHEEL_VERSION:
        raise RuntimeError(
            f"expected nvidia-cutlass-dsl {EXPECTED_WHEEL_VERSION}, "
            f"got {distribution.version}"
        )
    distribution_root = Path(distribution.locate_file("")).resolve()

    target_value = os.environ.get("FLASHINFER_WHEEL_VALIDATION_TARGET")
    target = Path(target_value).resolve() if target_value is not None else None
    if target is not None and not distribution_root.is_relative_to(target):
        raise RuntimeError(
            "nvidia-cutlass-dsl resolved outside the selected target: "
            f"{distribution_root}"
        )
    wheel_hash = None
    provenance_source = None
    if EXPECTED_WHEEL_SHA256 is not None and EXPECTED_WHEEL_PIP_REPORT is not None:
        report_path = Path(EXPECTED_WHEEL_PIP_REPORT).resolve()
        report = json.loads(report_path.read_text())
        installs = [
            item
            for item in report.get("install", [])
            if item.get("metadata", {}).get("name", "").lower().replace("_", "-")
            == "nvidia-cutlass-dsl"
            and item.get("metadata", {}).get("version") == distribution.version
        ]
        if len(installs) != 1:
            raise RuntimeError(
                "pip report must contain exactly one matching nvidia-cutlass-dsl "
                f"record, got {len(installs)}"
            )
        archive_info = installs[0].get("download_info", {}).get("archive_info", {})
        provenance_source = {
            "kind": "pip-install-report",
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
        wheel_hash = archive_info.get("hashes", {}).get("sha256")
        if wheel_hash is None:
            legacy_hash = archive_info.get("hash", "")
            if legacy_hash.startswith("sha256="):
                wheel_hash = legacy_hash.removeprefix("sha256=")
    elif EXPECTED_WHEEL_SHA256 is not None:
        raise RuntimeError(
            "set FLASHINFER_CUTLASS_DSL_PIP_REPORT when pinning "
            "FLASHINFER_EXPECTED_CUTLASS_DSL_WHEEL_SHA256"
        )
    if EXPECTED_WHEEL_SHA256 is not None and wheel_hash != EXPECTED_WHEEL_SHA256:
        raise RuntimeError(
            f"selected target does not contain the requested wheel: {wheel_hash}"
        )
    return {
        "version": distribution.version,
        "base_version": base_version,
        "wheel_sha256": wheel_hash,
        "provenance_source": provenance_source,
    }


def _fp8_matrix(helper) -> tuple[object, ...]:
    cases = tuple(
        case for case in helper.context_perf_cases() if case.dtype == FP8_DTYPE_NAME
    )
    if len(cases) != 64 or len({case.case_id for case in cases}) != 64:
        raise AssertionError("FP8 causal context comparison matrix must have 64 rows")
    if any(case.mask_type != "causal" for case in cases):
        raise AssertionError("every comparison row must use causal masking")
    paged = tuple(case for case in cases if case.qkv_layout == "paged_kv")
    if len(paged) != 32 or any(case.page_size != PAGE_SIZE for case in paged):
        raise AssertionError("the matrix must have 32 paged page-size-32 rows")
    return cases


def _selected_cases(cases: Sequence[object], args) -> tuple[object, ...]:
    return tuple(
        case
        for case in cases
        if fnmatch.fnmatch(case.case_id, args.case_glob)
        and (not args.head_dim or case.head_dim in args.head_dim)
        and (not args.num_kv_heads or case.num_heads_kv in args.num_kv_heads)
        and (not args.batch_size or case.batch_size in args.batch_size)
        and (not args.seq_len_q or case.max_seq_len_q in args.seq_len_q)
        and (not args.seq_len_kv or case.max_seq_len_kv in args.seq_len_kv)
        and (not args.layout or case.qkv_layout in args.layout)
    )


def _resume_identity(environment: Mapping[str, object]) -> dict[str, object]:
    source = environment.get("source", {})
    cutlass = environment.get("cutlass_dsl", {})
    return {
        "git_commit": environment.get("git_commit"),
        "torch": environment.get("torch"),
        "torch_cuda": environment.get("torch_cuda"),
        "gpu": environment.get("gpu"),
        "source_files_sha256": (
            source.get("files_sha256") if isinstance(source, Mapping) else None
        ),
        "source_tree_fingerprint": environment.get("source_tree_fingerprint"),
        "comparison_script": environment.get("comparison_script"),
        "matrix_helper": environment.get("matrix_helper"),
        "cutlass_distribution": (
            {
                "version": cutlass.get("distribution_version"),
            }
            if isinstance(cutlass, Mapping)
            else None
        ),
        "cutlass_dsl_package": environment.get("cutlass_dsl_package"),
        "external_distributions": environment.get("external_distributions"),
    }


def _parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="FlashInfer source root under test",
    )
    parser.add_argument(
        "--matrix-helper",
        type=Path,
        default=script_dir / "bench_attention_ts_context.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="incrementally updated JSON report (required for an actual run)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--max-gap-percent",
        type=float,
        default=5.0,
        help=(
            "maximum TS gap from the order-balanced total-duration ratio (default: 5)"
        ),
    )
    parser.add_argument("--case-glob", default="*")
    parser.add_argument("--head-dim", type=int, action="append", choices=(128, 256))
    parser.add_argument("--num-kv-heads", type=int, action="append", choices=(4, 32))
    parser.add_argument("--batch-size", type=int, action="append", choices=(1, 4))
    parser.add_argument(
        "--seq-len-q", type=int, action="append", choices=(256, 1024, 4096, 16384)
    )
    parser.add_argument(
        "--seq-len-kv", type=int, action="append", choices=(1024, 4096, 16384)
    )
    parser.add_argument(
        "--layout", action="append", choices=("separate_qkv", "paged_kv")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the selected run contract without using CUDA",
    )
    return parser


def _selection_json(
    cases: Sequence[object], selected: Sequence[object], args
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_count": len(cases),
        "selected_count": len(selected),
        "selected_case_ids": [case.case_id for case in selected],
        "cases": [case.to_json() for case in selected],
        "timing": {
            "rounds": args.rounds,
            "iterations_per_round": args.iterations,
            "warmup_replays_per_backend": args.warmup,
            "cold_l2": True,
            "paired_backend_order": True,
        },
        "performance_gate": {
            "metric": PRIMARY_GATE_METRIC,
            "formula": PRIMARY_GATE_FORMULA,
            "paired_ratio_median_is_diagnostic_only": True,
            "maximum_gap_percent": args.max_gap_percent,
        },
        "interfaces": {
            "ts_separate": "BatchPrefillTSWrapper",
            "ts_paged": "BatchPrefillPagedTSWrapper",
            "trtllm_gen_separate": "trtllm_ragged_attention_deepseek",
            "trtllm_gen_paged": "BatchPrefillWithPagedKVCacheWrapper",
        },
        "dtype": {"qkv": "torch.float8_e4m3fn", "output": "torch.bfloat16"},
        "mask": "bottom-right causal",
        "paged_kv": "HND page size 32 with shuffled nonidentity physical IDs",
        "output": str(args.output.resolve()) if args.output is not None else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    repo = args.repo_root.resolve()
    helper_path = args.matrix_helper.resolve()
    if not helper_path.is_file():
        raise SystemExit(f"context matrix helper is missing: {helper_path}")
    helper = _load_module("_causal_context_matrix_helper", helper_path)
    cases = _fp8_matrix(helper)
    selected = _selected_cases(cases, args)
    if not selected:
        raise SystemExit("filters selected no FP8 causal context rows")
    if (
        args.rounds <= 1
        or args.iterations <= 0
        or args.warmup < 0
        or args.rounds * args.iterations % 2
    ):
        raise SystemExit(
            "rounds must exceed one; iterations positive; warmup nonnegative; "
            "and rounds * iterations even for balanced order positions"
        )
    if not math.isfinite(args.max_gap_percent) or args.max_gap_percent < 0:
        raise SystemExit("--max-gap-percent must be finite and nonnegative")
    selection = _selection_json(cases, selected, args)
    if args.list or args.dry_run:
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --list or --dry-run is used")
    if not (repo / ".git").exists():
        raise SystemExit(f"not a FlashInfer Git checkout: {repo}")
    if not torch.cuda.is_available():
        raise SystemExit("FP8 causal context comparison requires CUDA")

    sys.path.insert(0, str(repo))
    import flashinfer as selected_flashinfer

    flashinfer_file = Path(selected_flashinfer.__file__).resolve()
    expected_flashinfer_root = (repo / "flashinfer").resolve()
    if not flashinfer_file.is_relative_to(expected_flashinfer_root):
        raise SystemExit(
            f"flashinfer import did not resolve under --repo-root: {flashinfer_file}"
        )

    cutlass_dsl_package = _verify_cutlass_dsl()
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability not in SUPPORTED_CAPABILITIES:
        raise SystemExit(f"comparison requires SM100/SM103, got {capability}")
    cold_l2_scrubber = prepare_cold_l2_scrubber(torch, args.device)

    script_path = Path(__file__).resolve()
    environment = helper._environment(repo, device)
    environment["comparison_script"] = {
        "name": script_path.name,
        "sha256": _sha256(script_path),
    }
    environment["matrix_helper"] = {
        "name": helper_path.name,
        "sha256": _sha256(helper_path),
    }
    environment["cutlass_dsl_package"] = cutlass_dsl_package
    environment["source_tree_fingerprint"] = _source_tree_fingerprint(repo)
    environment["external_distributions"] = {
        "flashinfer-cubin": _distribution_provenance("flashinfer-cubin")
    }
    protocol = {
        "matrix": "64 FP8 bottom-right-causal context rows",
        "selected_case_ids": [case.case_id for case in selected],
        "matrix_axes": {
            "head_dim": [128, 256],
            "num_heads_q": [32],
            "num_heads_kv": [32, 4],
            "batch_size": [1, 4],
            "sequence_shapes_q_kv": [
                [1024, 1024],
                [4096, 4096],
                [16384, 16384],
                [256, 4096],
            ],
            "qkv_layout": ["separate_qkv", "paged_kv"],
        },
        "qkv_dtype": "torch.float8_e4m3fn for both backends",
        "output_dtype": "torch.bfloat16 for both backends",
        "mask_type": "bottom-right causal",
        "input_pairing": "identical logical Q/K/V and metadata per backend",
        "ts_separate_interface": (
            "flashinfer.attention.prims_ts.BatchPrefillTSWrapper"
        ),
        "ts_paged_interface": (
            "flashinfer.attention.prims_ts.BatchPrefillPagedTSWrapper"
        ),
        "trtllm_separate_interface": (
            "flashinfer.prefill.trtllm_ragged_attention_deepseek(backend='trtllm-gen')"
        ),
        "trtllm_paged_interface": (
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper"
            "(backend='trtllm-gen')"
        ),
        "paged_layout": (
            "HND, page size 32, unique shuffled nonidentity physical pages, "
            "poisoned guard pages"
        ),
        "workspace_bytes": WORKSPACE_BYTES,
        "trtllm_counter_buffer": (
            "optional paged-wrapper implementation detail when exposed by the "
            "installed FlashInfer baseline; otherwise no caller-visible buffer"
        ),
        "trtllm_counter_buffer_size": (
            "when exposed by the installed paged wrapper: 4 * round_up(max("
            "batch_size * num_qo_heads, sm_count), 8) bytes; otherwise 0 "
            "caller-visible bytes"
        ),
        "trtllm_counter_lifecycle": (
            "zero-initialized before first use, retained outside capture, "
            "kernel-reset state checked after direct and graph-replay launches"
        ),
        "trtllm_enable_pdl": True,
        "timing": (
            "paired alternating CUDA-graph replay with a non-compressible "
            "2x-L2 scrub before each individually timed replay; scrub, "
            "planning, compilation, capture, and warmup excluded"
        ),
        "cold_l2_scrub": cold_l2_scrubber.metadata(),
        "rounds": args.rounds,
        "iterations_per_round": args.iterations,
        "warmup_replays_per_backend": args.warmup,
        "primary_gap_metric": PRIMARY_GATE_METRIC,
        "primary_gap_formula": PRIMARY_GATE_FORMULA,
        "paired_ratio_median_is_diagnostic_only": True,
        "raw_samples_recorded": True,
        "backend_total_durations_recorded": True,
        "backend_arithmetic_means_recorded": True,
        "backend_order_conditioned_summaries_recorded": True,
        "maximum_gap_percent": args.max_gap_percent,
        "correctness": (
            "eight exact FP32 bottom-right-causal samples and full finiteness "
            "per backend, plus full chunked TS/TRTLLM-Gen output comparison, "
            "before capture and after output-poisoned graph replay"
        ),
        "seed": args.seed,
    }

    old_results: dict[str, dict[str, object]] = {}
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("schema_version") != SCHEMA_VERSION:
            raise SystemExit(
                "cannot resume: report schema differs; start a fresh aggregate-"
                "gated report"
            )
        if previous.get("protocol") != protocol:
            raise SystemExit("cannot resume: report protocol differs")
        if _resume_identity(previous.get("environment", {})) != _resume_identity(
            environment
        ):
            raise SystemExit(
                "cannot resume: source, comparator, wheel, runtime, or GPU differs"
            )
        for row in previous.get("results", []):
            if row.get("status") != "ok" or "ts_policy" not in row:
                continue
            performance = row.get("performance", {})
            aggregate = (
                performance.get("order_balanced_total_duration", {})
                if isinstance(performance, Mapping)
                else {}
            )
            if (
                not isinstance(aggregate, Mapping)
                or aggregate.get("gate_metric") != PRIMARY_GATE_METRIC
                or aggregate.get("gate_formula") != PRIMARY_GATE_FORMULA
                or not isinstance(aggregate.get("ts_gap_percent"), (int, float))
            ):
                raise SystemExit(
                    f"cannot resume: row {row.get('case_id')} lacks the stable "
                    "aggregate gate contract"
                )
            old_results[row["case_id"]] = row

    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "protocol": protocol,
        "matrix": [case.to_json() for case in cases],
        "results": [],
        "summary": {
            "status": "running",
            "matrix_count": len(cases),
            "selected": len(selected),
            "completed": 0,
            "failed": 0,
            "gate_metric": PRIMARY_GATE_METRIC,
            "gate_formula": PRIMARY_GATE_FORMULA,
            "paired_ratio_median_is_diagnostic_only": True,
            "over_order_balanced_total_duration_gap_limit": 0,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    rows: list[dict[str, object]] = []
    failures = 0
    for case in selected:
        if case.case_id in old_results:
            rows.append(old_results[case.case_id])
            print(f"[resume] {case.case_id}", flush=True)
            continue
        print(f"[run] {case.case_id}", flush=True)
        started = time.time()
        row: dict[str, object] = {
            "case_id": case.case_id,
            "case": case.to_json(),
            "input_seed": helper._case_seed(case, args.seed),
            "status": "running",
        }
        inputs = ts_runner = trt_runner = performance = None
        try:
            inputs = helper._make_inputs(
                case,
                device=device,
                seed=helper._case_seed(case, args.seed),
            )
            # Release the helper's FP8 output.  Both compared public interfaces
            # own independent BF16 outputs under the common signoff contract.
            inputs.out = torch.empty(0, dtype=OUTPUT_DTYPE, device=device)
            if inputs.paged_kv_indices_host is not None:
                row["paged_kv_indices_unique"] = len(
                    set(inputs.paged_kv_indices_host)
                ) == len(inputs.paged_kv_indices_host)
                row["paged_kv_indices_nonidentity"] = (
                    inputs.paged_kv_indices_host
                    != tuple(range(len(inputs.paged_kv_indices_host)))
                )

            ts_runner = _plan_ts(inputs)
            trt_runner = _plan_trtllm_gen(inputs)
            row["ts_policy"] = _selected_ts_policy(ts_runner.owner)

            counter_nonzero_before_first = _counter_nonzero(trt_runner)
            if counter_nonzero_before_first:
                raise AssertionError(
                    "TRTLLM-Gen counter buffer was not zero before first use"
                )

            ts_runner.launch()
            trt_runner.launch()
            torch.cuda.synchronize(device)
            counter_nonzero_after_first = _counter_nonzero(trt_runner)
            if counter_nonzero_after_first:
                raise AssertionError(
                    "TRTLLM-Gen did not reset its counter buffer after first call"
                )
            row["direct_accuracy"] = {
                "ts": _check_backend_accuracy(helper, inputs, ts_runner.output),
                "trtllm_gen": _check_backend_accuracy(
                    helper, inputs, trt_runner.output
                ),
                "cross_backend": _compare_outputs(ts_runner.output, trt_runner.output),
            }

            performance = _benchmark_pair(
                ts_runner,
                trt_runner,
                device=device,
                scrubber=cold_l2_scrubber,
                rounds=args.rounds,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            performance["trtllm_gen"].update(
                {
                    "counter_nonzero_before_first": counter_nonzero_before_first,
                    "counter_nonzero_after_first": counter_nonzero_after_first,
                }
            )
            row["graph_replay_accuracy"] = {
                "ts": _check_backend_accuracy(helper, inputs, ts_runner.output),
                "trtllm_gen": _check_backend_accuracy(
                    helper, inputs, trt_runner.output
                ),
                "cross_backend": _compare_outputs(ts_runner.output, trt_runner.output),
            }
            performance.pop("_retained_graph_objects", None)
            flops = helper._causal_flops(case, inputs.q_lengths, inputs.kv_lengths)
            ts_median = performance["ts"]["median_ms"]
            trt_median = performance["trtllm_gen"]["median_ms"]
            aggregate = performance["order_balanced_total_duration"]
            aggregate_ratio = aggregate["ts_latency_over_trtllm_gen"]
            gap_percent = aggregate["ts_gap_percent"]
            paired_median_ratio = performance["paired_ts_latency_over_trtllm_gen"][
                "all_samples"
            ]["median"]
            paired_median_gap_percent = (paired_median_ratio - 1.0) * 100.0
            performance.update(
                {
                    "causal_flops": flops,
                    "ts_tflops_from_median": flops / ts_median / 1e9,
                    "trtllm_gen_tflops_from_median": flops / trt_median / 1e9,
                    "ts_latency_over_trtllm_gen_ratio_of_medians_diagnostic": (
                        ts_median / trt_median
                    ),
                    "ts_latency_over_trtllm_gen_paired_median_diagnostic": (
                        paired_median_ratio
                    ),
                    "ts_gap_percent_paired_median_diagnostic": (
                        paired_median_gap_percent
                    ),
                    "performance_gate_metric": PRIMARY_GATE_METRIC,
                    "performance_gate_formula": PRIMARY_GATE_FORMULA,
                    "ts_latency_over_trtllm_gen_order_balanced_total_duration": (
                        aggregate_ratio
                    ),
                    "ts_gap_percent_order_balanced_total_duration": gap_percent,
                    "maximum_allowed_gap_percent": args.max_gap_percent,
                    "within_gap_limit": gap_percent <= args.max_gap_percent,
                    "within_five_percent_of_trtllm_gen": gap_percent <= 5.0,
                    "winner_order_balanced_total_duration": (
                        "ts" if aggregate_ratio < 1.0 else "trtllm_gen"
                    ),
                }
            )
            row["performance"] = performance
            row["status"] = "ok"
            row["elapsed_seconds"] = time.time() - started
            print(
                f"[ok] {case.case_id}: TS {ts_median:.6f} ms, "
                f"TRTLLM-Gen {trt_median:.6f} ms, aggregate gap "
                f"{gap_percent:+.3f}% (means "
                f"{aggregate['ts_arithmetic_mean_ms']:.6f}/"
                f"{aggregate['trtllm_gen_arithmetic_mean_ms']:.6f} ms; "
                f"paired-median diagnostic {paired_median_gap_percent:+.3f}%)",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - preserve partial diagnostics
            failures += 1
            row["status"] = "error"
            row["error"] = f"{type(error).__name__}: {error}"
            row["traceback"] = traceback.format_exc()
            row["elapsed_seconds"] = time.time() - started
            print(f"[error] {case.case_id}: {row['error']}", flush=True)
        rows.append(row)
        rows.sort(key=lambda item: item["case"]["matrix_index"])
        completed = sum(item.get("status") == "ok" for item in rows)
        failed = sum(item.get("status") == "error" for item in rows)
        over_limit = sum(
            item.get("status") == "ok" and not item["performance"]["within_gap_limit"]
            for item in rows
        )
        report["results"] = rows
        report["summary"] = {
            **report["summary"],
            "completed": completed,
            "failed": failed,
            "over_order_balanced_total_duration_gap_limit": over_limit,
        }
        _write_report(args.output, report)
        inputs = ts_runner = trt_runner = performance = None
        gc.collect()
        torch.cuda.empty_cache()
        if failures and args.fail_fast:
            break

    completed = sum(row.get("status") == "ok" for row in rows)
    failed = sum(row.get("status") == "error" for row in rows)
    over_limit_ids = [
        row["case_id"]
        for row in rows
        if row.get("status") == "ok" and not row["performance"]["within_gap_limit"]
    ]
    successful_rows = [row for row in rows if row.get("status") == "ok"]
    worst_gate_row = (
        max(
            successful_rows,
            key=lambda row: row["performance"][
                "ts_gap_percent_order_balanced_total_duration"
            ],
        )
        if successful_rows
        else None
    )
    if failed or completed != len(selected):
        status = "failed"
    elif over_limit_ids:
        status = "performance_gap"
    else:
        status = "ok"
    report["results"] = rows
    report["summary"] = {
        **report["summary"],
        "status": status,
        "matrix_count": len(cases),
        "selected": len(selected),
        "completed": completed,
        "failed": failed,
        "gate_metric": PRIMARY_GATE_METRIC,
        "gate_formula": PRIMARY_GATE_FORMULA,
        "over_order_balanced_total_duration_gap_limit": len(over_limit_ids),
        "over_order_balanced_total_duration_gap_limit_case_ids": over_limit_ids,
        "worst_order_balanced_total_duration_case_id": (
            None if worst_gate_row is None else worst_gate_row["case_id"]
        ),
        "worst_order_balanced_total_duration_gap_percent": (
            None
            if worst_gate_row is None
            else worst_gate_row["performance"][
                "ts_gap_percent_order_balanced_total_duration"
            ]
        ),
        "all_selected_within_order_balanced_total_duration_gap_limit": (
            not over_limit_ids and not failed and completed == len(selected)
        ),
        "all_selected_within_gap_limit": not over_limit_ids
        and not failed
        and completed == len(selected),
        "all_64_completed": len(selected) == 64 and completed == 64 and failed == 0,
        "all_64_within_order_balanced_total_duration_gap_limit": len(selected) == 64
        and completed == 64
        and failed == 0
        and not over_limit_ids,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_report(args.output, report)
    if status == "ok":
        return 0
    if status == "performance_gap":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
