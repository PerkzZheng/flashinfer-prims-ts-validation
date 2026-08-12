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

"""CPU-only tests for Attention-TS paired timing summaries."""

import math
import statistics

import pytest

from .timing import summarize_paired_two_order_cycles, summarize_times


def test_paired_aggregate_cancels_order_bias_despite_raw_median_conflict() -> None:
    ts_times_ms = [1.0, 94.0, 2.0, 3.0]
    reference_times_ms = [45.0, 50.0, 1.0, 4.0]

    assert statistics.median(ts_times_ms) != statistics.median(reference_times_ms)
    summary = summarize_paired_two_order_cycles(ts_times_ms, reference_times_ms)

    assert summary["method"] == "alternating-order-balanced-total-duration-ratio"
    assert summary["sample_count_per_backend"] == 4
    assert summary["cycle_count"] == 2
    assert summary["gap_percent_total_duration_ratio"] == 0.0
    assert summary["cycle_gap_percent_median"] == 0.0
    assert summary["cycle_gap_percent_p95"] == 0.0
    assert summary["cycle_gap_percent_min"] == 0.0
    assert summary["cycle_gap_percent_max"] == 0.0
    assert summary["backend_total_duration_ms"] == {
        "attention_ts": 100.0,
        "trtllm_gen": 100.0,
    }
    assert summary["backend_arithmetic_mean_us"] == {
        "attention_ts": 25_000.0,
        "trtllm_gen": 25_000.0,
    }
    assert (
        summary["position_summaries"]["attention_ts"]["first"]["arithmetic_mean_us"]
        == 1_500.0
    )
    assert (
        summary["position_summaries"]["attention_ts"]["second"]["arithmetic_mean_us"]
        == 48_500.0
    )
    assert (
        summary["position_summaries"]["trtllm_gen"]["first"]["arithmetic_mean_us"]
        == 27_000.0
    )
    assert (
        summary["position_summaries"]["trtllm_gen"]["second"]["arithmetic_mean_us"]
        == 23_000.0
    )
    assert summary["raw_samples_ms"] == {
        "attention_ts": ts_times_ms,
        "trtllm_gen": reference_times_ms,
    }


def test_paired_cycles_report_known_ten_percent_gap() -> None:
    summary = summarize_paired_two_order_cycles(
        [11.0, 22.0, 5.5, 5.5],
        [10.0, 20.0, 5.0, 5.0],
    )

    assert summary["sample_count_per_backend"] == 4
    assert summary["cycle_count"] == 2
    assert summary["gap_percent_total_duration_ratio"] == pytest.approx(10.0)
    assert summary["cycle_gap_percent_median"] == pytest.approx(10.0)
    assert summary["cycle_gap_percent_p95"] == pytest.approx(10.0)
    assert summary["cycle_gap_percent_min"] == pytest.approx(10.0)
    assert summary["cycle_gap_percent_max"] == pytest.approx(10.0)


def test_bimodal_49_51_cycle_median_flip_does_not_flip_five_percent_gate() -> None:
    def samples(slow_cycle_count: int) -> tuple[list[float], list[float]]:
        # Each cycle has the same reference duration. Changing one cycle from
        # 0.9x to 1.1x flips the cycle median at 49/51, while the aggregate
        # total-duration ratio moves continuously from -0.2% to +0.2%.
        ratios = [1.1] * slow_cycle_count + [0.9] * (100 - slow_cycle_count)
        ts = [ratio for ratio in ratios for _ in range(2)]
        reference = [1.0] * len(ts)
        return ts, reference

    low_ts, low_reference = samples(49)
    high_ts, high_reference = samples(51)
    low = summarize_paired_two_order_cycles(low_ts, low_reference)
    high = summarize_paired_two_order_cycles(high_ts, high_reference)

    assert low["cycle_gap_percent_median"] == pytest.approx(-10.0)
    assert high["cycle_gap_percent_median"] == pytest.approx(10.0)
    assert low["gap_percent_total_duration_ratio"] == pytest.approx(-0.2)
    assert high["gap_percent_total_duration_ratio"] == pytest.approx(0.2)
    assert low["gap_percent_total_duration_ratio"] <= 5.0
    assert high["gap_percent_total_duration_ratio"] <= 5.0


@pytest.mark.parametrize(
    ("ts_times_ms", "reference_times_ms", "message"),
    [
        ([1.0, 2.0], [1.0, 2.0, 3.0], "equal"),
        ([], [], "at least two"),
        ([1.0], [1.0], "at least two"),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "even"),
    ],
)
def test_paired_cycles_reject_invalid_sample_counts(
    ts_times_ms: list[float],
    reference_times_ms: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_paired_two_order_cycles(ts_times_ms, reference_times_ms)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_paired_cycles_reject_invalid_durations(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        summarize_paired_two_order_cycles([1.0, invalid], [1.0, 1.0])
    with pytest.raises(ValueError, match="finite positive"):
        summarize_paired_two_order_cycles([1.0, 1.0], [1.0, invalid])
    with pytest.raises(ValueError, match="finite positive"):
        summarize_times([1.0, invalid], batch_size=1)
