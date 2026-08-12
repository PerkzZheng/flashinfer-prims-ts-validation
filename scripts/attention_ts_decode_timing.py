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

"""Backend-neutral CUDA timing mechanics for Attention-TS benchmarks."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

_TS_BACKEND = "attention_ts"
_REFERENCE_BACKEND = "trtllm_gen"
_COLD_L2_SCRUB_MULTIPLIER = 2
_COLD_L2_SCRUB_SEED = 0x5EED_C01D
_COLD_L2_SCRUB_STRATEGY = (
    "external-run-scoped-2x-l2-seeded-random-int8-add-before-event-v1"
)


def _positive_finite_samples(values: Sequence[float], *, label: str) -> list[float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError(f"cannot summarize an empty {label}")
    if any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError(f"{label} must contain only finite positive durations")
    return samples


def _finite_samples(values: Sequence[float], *, label: str) -> list[float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError(f"cannot summarize an empty {label}")
    if any(not math.isfinite(value) for value in samples):
        raise ValueError(f"{label} must contain only finite values")
    return samples


@dataclass(frozen=True)
class ColdL2Scrubber:
    """Run-scoped, non-compressible same-stream L2 scrub storage."""

    buffer: Any
    device_index: int
    l2_cache_bytes: int
    scrub_bytes: int

    def enqueue(self) -> None:
        """Read and rewrite every random byte on the current CUDA stream."""

        self.buffer.add_(1)

    def metadata(self) -> dict[str, Any]:
        return {
            "strategy": _COLD_L2_SCRUB_STRATEGY,
            "l2_cache_bytes": self.l2_cache_bytes,
            "scrub_bytes": self.scrub_bytes,
            "scrub_multiplier": _COLD_L2_SCRUB_MULTIPLIER,
            "dtype": "int8",
            "initialization": "seeded-uniform-full-int8-range",
            "seed": _COLD_L2_SCRUB_SEED,
            "operation": "in-place-add-one",
            "compressible_pattern": False,
            "same_current_stream": True,
            "timed": False,
            "lifecycle": "one-buffer-prepared-before-cases-and-reused-for-run",
        }


def prepare_cold_l2_scrubber(torch: Any, device: int) -> ColdL2Scrubber:
    """Allocate, initialize, and warm one portable run-scoped L2 scrubber."""

    l2_bytes = int(torch.cuda.get_device_properties(device).L2_cache_size)
    scrub_bytes = _COLD_L2_SCRUB_MULTIPLIER * l2_bytes
    buffer = torch.empty(scrub_bytes, dtype=torch.int8, device=device)
    generator = torch.Generator(device=f"cuda:{device}")
    generator.manual_seed(_COLD_L2_SCRUB_SEED)
    buffer.random_(-128, 128, generator=generator)
    buffer.add_(1)
    torch.cuda.synchronize(device)
    return ColdL2Scrubber(
        buffer=buffer,
        device_index=device,
        l2_cache_bytes=l2_bytes,
        scrub_bytes=scrub_bytes,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(_finite_samples(values, label="sample"))
    if len(ordered) == 1:
        return ordered[0]
    position = percentile / 100.0 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_times(times_ms: Sequence[float], batch_size: int) -> dict[str, float]:
    """Summarize per-call milliseconds without discarding throughput context."""

    samples = _positive_finite_samples(times_ms, label="timing sample")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    median_ms = float(statistics.median(samples))
    arithmetic_mean_ms = float(statistics.fmean(samples))
    return {
        "median_us": median_ms * 1000.0,
        "arithmetic_mean_us": arithmetic_mean_ms * 1000.0,
        "p95_us": _percentile(samples, 95.0) * 1000.0,
        "min_us": min(samples) * 1000.0,
        "max_us": max(samples) * 1000.0,
        "total_duration_ms": float(sum(samples)),
        "tokens_per_second": batch_size / (median_ms / 1000.0),
    }


def _summarize_position(times_ms: Sequence[float]) -> dict[str, float | int]:
    """Return descriptive statistics for one backend/order position."""

    samples = _positive_finite_samples(times_ms, label="timing position")
    return {
        "sample_count": len(samples),
        "total_duration_ms": float(sum(samples)),
        "arithmetic_mean_us": float(statistics.fmean(samples)) * 1000.0,
        "median_us": float(statistics.median(samples)) * 1000.0,
        "p95_us": _percentile(samples, 95.0) * 1000.0,
        "min_us": min(samples) * 1000.0,
        "max_us": max(samples) * 1000.0,
    }


def summarize_paired_two_order_cycles(
    ts_times_ms: Sequence[float], reference_times_ms: Sequence[float]
) -> dict[str, Any]:
    """Summarize an order-balanced total ratio and per-cycle diagnostics.

    Paired timing alternates which backend replays first.  Consequently, two
    adjacent samples form one complete order cycle: each backend ran once in
    the first position and once in the second position.  The regression gate
    is exactly ``sum(TS) / sum(reference) - 1`` over all complete cycles.  This
    gives every recorded duration its natural weight and avoids the unstable
    50th-percentile vote produced by a median of cycle ratios.  Cycle-ratio
    percentiles remain descriptive diagnostics only.
    """

    sample_count = len(ts_times_ms)
    if sample_count != len(reference_times_ms):
        raise ValueError("paired timing requires equal TS and reference sample counts")
    if sample_count < 2:
        raise ValueError("paired timing requires at least two samples per backend")
    if sample_count % 2 != 0:
        raise ValueError(
            "paired timing requires an even sample count for complete order cycles"
        )

    ts_samples = _positive_finite_samples(ts_times_ms, label="TS timing sample")
    reference_samples = _positive_finite_samples(
        reference_times_ms, label="reference timing sample"
    )
    cycle_gaps_percent: list[float] = []
    for cycle_start in range(0, sample_count, 2):
        ts_cycle_ms = ts_samples[cycle_start] + ts_samples[cycle_start + 1]
        reference_cycle_ms = (
            reference_samples[cycle_start] + reference_samples[cycle_start + 1]
        )
        cycle_gaps_percent.append((ts_cycle_ms / reference_cycle_ms - 1.0) * 100.0)

    ts_total_ms = float(sum(ts_samples))
    reference_total_ms = float(sum(reference_samples))
    total_duration_gap_percent = (ts_total_ms / reference_total_ms - 1.0) * 100.0

    # Even samples execute TS first; odd samples execute the reference first.
    position_summaries = {
        _TS_BACKEND: {
            "first": _summarize_position(ts_samples[0::2]),
            "second": _summarize_position(ts_samples[1::2]),
        },
        _REFERENCE_BACKEND: {
            "first": _summarize_position(reference_samples[1::2]),
            "second": _summarize_position(reference_samples[0::2]),
        },
    }
    return {
        "method": "alternating-order-balanced-total-duration-ratio",
        "gate_formula": ("(sum(attention_ts_ms) / sum(trtllm_gen_ms) - 1) * 100"),
        "sample_count_per_backend": sample_count,
        "cycle_count": len(cycle_gaps_percent),
        "gap_percent_total_duration_ratio": total_duration_gap_percent,
        "backend_total_duration_ms": {
            _TS_BACKEND: ts_total_ms,
            _REFERENCE_BACKEND: reference_total_ms,
        },
        "backend_arithmetic_mean_us": {
            _TS_BACKEND: float(statistics.fmean(ts_samples)) * 1000.0,
            _REFERENCE_BACKEND: float(statistics.fmean(reference_samples)) * 1000.0,
        },
        "position_summaries": position_summaries,
        "cycle_gap_percent_median": float(statistics.median(cycle_gaps_percent)),
        "cycle_gap_percent_p95": _percentile(cycle_gaps_percent, 95.0),
        "cycle_gap_percent_min": min(cycle_gaps_percent),
        "cycle_gap_percent_max": max(cycle_gaps_percent),
        "raw_samples_ms": {
            _TS_BACKEND: ts_samples,
            _REFERENCE_BACKEND: reference_samples,
        },
    }


def time_backend(
    run: Callable[[], Any],
    *,
    bench_gpu_time: Callable[..., Sequence[float]],
    batch_size: int,
    warmup_iters: int,
    repeat_iters: int,
    enable_cupti: bool,
    cold_l2_cache: bool,
) -> dict[str, float | bool | int | str]:
    """Time one backend through FlashInfer's event/CUPTI compatibility path."""

    use_cuda_graph = not enable_cupti and not cold_l2_cache
    times_ms = bench_gpu_time(
        run,
        dry_run_iters=warmup_iters,
        repeat_iters=repeat_iters,
        enable_cupti=enable_cupti,
        use_cuda_graph=use_cuda_graph,
        cold_l2_cache=cold_l2_cache,
    )
    summary: dict[str, float | bool | int | str] = summarize_times(times_ms, batch_size)
    summary["use_cuda_graph"] = use_cuda_graph
    summary["paired_timing"] = False
    summary["alternating_replay_order"] = False
    summary["sample_count"] = repeat_iters
    summary["calls_per_sample"] = 1
    summary["timing_mode"] = (
        "cupti"
        if enable_cupti
        else "cold-l2-cuda-events"
        if cold_l2_cache
        else "cuda-graph"
    )
    return summary


def paired_backend_order(sample_index: int) -> tuple[str, str]:
    """Alternate replay order so neither backend always runs first."""

    if sample_index % 2 == 0:
        return (_TS_BACKEND, _REFERENCE_BACKEND)
    return (_REFERENCE_BACKEND, _TS_BACKEND)


def capture_repeated_graph(
    run: Callable[[], Any], calls_per_graph: int, torch: Any
) -> Any:
    """Capture an identical number of public calls in one CUDA graph."""

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            run()
    return graph


def time_paired_cuda_graphs(
    run_ts: Callable[[], Any],
    run_reference: Callable[[], Any],
    *,
    torch: Any,
    batch_size: int,
    warmup_replays: int,
    sample_count: int,
    calls_per_graph: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Time two equal-size hot-cache graphs with alternating replay order."""

    torch.cuda.synchronize()
    graphs = {
        _TS_BACKEND: capture_repeated_graph(run_ts, calls_per_graph, torch),
        _REFERENCE_BACKEND: capture_repeated_graph(
            run_reference, calls_per_graph, torch
        ),
    }
    torch.cuda.synchronize()

    for warmup_index in range(warmup_replays):
        for backend in paired_backend_order(warmup_index):
            graphs[backend].replay()
    torch.cuda.synchronize()

    samples_ms: dict[str, list[float]] = {
        _TS_BACKEND: [],
        _REFERENCE_BACKEND: [],
    }
    for sample_index in range(sample_count):
        pending_events = {}
        for backend in paired_backend_order(sample_index):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[backend].replay()
            end.record()
            pending_events[backend] = (start, end)
        torch.cuda.synchronize()
        for backend, (start, end) in pending_events.items():
            samples_ms[backend].append(float(start.elapsed_time(end)) / calls_per_graph)

    common = {
        "use_cuda_graph": True,
        "paired_timing": True,
        "alternating_replay_order": True,
        "sample_count": sample_count,
        "calls_per_sample": calls_per_graph,
        "timing_mode": "paired-alternating-cuda-graphs",
    }
    ts_summary = {**summarize_times(samples_ms[_TS_BACKEND], batch_size), **common}
    reference_summary = {
        **summarize_times(samples_ms[_REFERENCE_BACKEND], batch_size),
        **common,
    }
    comparison_summary = summarize_paired_two_order_cycles(
        samples_ms[_TS_BACKEND], samples_ms[_REFERENCE_BACKEND]
    )
    return ts_summary, reference_summary, comparison_summary


def time_paired_cold_l2_cuda_graphs(
    run_ts: Callable[[], Any],
    run_reference: Callable[[], Any],
    *,
    torch: Any,
    device: int,
    batch_size: int,
    warmup_replays: int,
    sample_count: int,
    scrubber: ColdL2Scrubber,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Time one-call graphs after an untimed same-stream 2x-L2 scrub.

    The scrub buffer starts with non-compressible random bytes and is updated
    in place before every replay. This avoids the allocation-order-dependent
    partial eviction observed when a uniformly zero buffer is compressed by
    Blackwell L2. Stream ordering keeps the scrub outside ``elapsed_time``;
    alternating backend order avoids a fixed position bias.
    """

    torch.cuda.synchronize(device)
    graphs = {
        _TS_BACKEND: capture_repeated_graph(run_ts, 1, torch),
        _REFERENCE_BACKEND: capture_repeated_graph(run_reference, 1, torch),
    }
    torch.cuda.synchronize(device)

    if scrubber.device_index != device:
        raise ValueError(
            f"cold-L2 scrubber belongs to device {scrubber.device_index}, "
            f"not requested device {device}"
        )

    for warmup_index in range(warmup_replays):
        for backend in paired_backend_order(warmup_index):
            scrubber.enqueue()
            graphs[backend].replay()
    torch.cuda.synchronize(device)

    samples_ms: dict[str, list[float]] = {
        _TS_BACKEND: [],
        _REFERENCE_BACKEND: [],
    }
    pending_events: list[tuple[str, Any, Any]] = []
    for sample_index in range(sample_count):
        for backend in paired_backend_order(sample_index):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            scrubber.enqueue()
            start.record()
            graphs[backend].replay()
            end.record()
            pending_events.append((backend, start, end))
    torch.cuda.synchronize(device)
    for backend, start, end in pending_events:
        samples_ms[backend].append(float(start.elapsed_time(end)))

    common = {
        "use_cuda_graph": True,
        "paired_timing": True,
        "alternating_replay_order": True,
        "sample_count": sample_count,
        "calls_per_sample": 1,
        "timing_mode": "paired-alternating-cold-l2-cuda-graphs",
        "cold_l2_cache": True,
        "cold_l2_strategy": _COLD_L2_SCRUB_STRATEGY,
        "l2_cache_bytes": scrubber.l2_cache_bytes,
        "l2_flush_bytes": scrubber.scrub_bytes,
        "l2_flush_timed": False,
        "l2_flush_compressible_pattern": False,
        "cold_l2_scrub": scrubber.metadata(),
    }
    ts_summary = {**summarize_times(samples_ms[_TS_BACKEND], batch_size), **common}
    reference_summary = {
        **summarize_times(samples_ms[_REFERENCE_BACKEND], batch_size),
        **common,
    }
    comparison_summary = summarize_paired_two_order_cycles(
        samples_ms[_TS_BACKEND], samples_ms[_REFERENCE_BACKEND]
    )
    return ts_summary, reference_summary, comparison_summary


def first_call(run: Callable[[], Any], torch: Any) -> tuple[Any, float]:
    """Run once and report synchronized host-observed latency in milliseconds."""

    torch.cuda.synchronize()
    start = time.perf_counter()
    result = run()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000.0
