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

"""CPU-only checks for the causal-context comparison performance gate."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_driver() -> ModuleType:
    path = Path(__file__).with_name("bench_attention_ts_context_vs_trtllm_gen.py")
    name = "_attention_ts_context_comparison_metrics_test_driver"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load context comparison driver from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DRIVER = _load_driver()


def _bimodal_summary(high_ratio_count: int) -> dict[str, object]:
    ratios = [1.1] * high_ratio_count + [0.9] * (100 - high_ratio_count)
    return DRIVER._summarize_paired_samples(
        {
            "ts": [ratios[:50], ratios[50:]],
            "trtllm_gen": [[1.0] * 50, [1.0] * 50],
        }
    )


def test_49_51_paired_median_flip_does_not_destabilize_aggregate_gate() -> None:
    low = _bimodal_summary(49)
    high = _bimodal_summary(51)

    low_gate = low["order_balanced_total_duration"]
    high_gate = high["order_balanced_total_duration"]
    assert low_gate["ts_gap_percent"] == pytest.approx(-0.2)
    assert high_gate["ts_gap_percent"] == pytest.approx(0.2)
    assert low_gate["ts_total_duration_ms"] == pytest.approx(99.8)
    assert high_gate["ts_total_duration_ms"] == pytest.approx(100.2)
    assert low_gate["trtllm_gen_total_duration_ms"] == pytest.approx(100.0)
    assert low_gate["ts_arithmetic_mean_ms"] == pytest.approx(0.998)
    assert low_gate["trtllm_gen_arithmetic_mean_ms"] == pytest.approx(1.0)

    assert low["paired_ts_latency_over_trtllm_gen"]["all_samples"][
        "median"
    ] == pytest.approx(0.9)
    assert high["paired_ts_latency_over_trtllm_gen"]["all_samples"][
        "median"
    ] == pytest.approx(1.1)
    assert low["paired_ts_latency_over_trtllm_gen"]["diagnostic_only"] is True
    assert low_gate["gate_metric"] == DRIVER.PRIMARY_GATE_METRIC
    assert low_gate["gate_formula"] == DRIVER.PRIMARY_GATE_FORMULA


def test_backend_order_conditioned_summaries_are_exactly_balanced() -> None:
    summary = _bimodal_summary(49)["order_balanced_total_duration"]
    by_position = summary["backend_order_conditioned_ms"]
    for backend in ("ts", "trtllm_gen"):
        assert by_position[backend]["first"]["count"] == 50
        assert by_position[backend]["second"]["count"] == 50
        assert by_position[backend]["first"]["total_duration_ms"] == pytest.approx(
            by_position[backend]["first"]["arithmetic_mean_ms"] * 50
        )
        assert by_position[backend]["second"]["total_duration_ms"] == pytest.approx(
            by_position[backend]["second"]["arithmetic_mean_ms"] * 50
        )


@pytest.mark.parametrize(
    "samples",
    [
        {"ts": [[1.0, 1.0]], "trtllm_gen": [[1.0]]},
        {"ts": [[1.0, 1.0, 1.0]], "trtllm_gen": [[1.0, 1.0, 1.0]]},
        {"ts": [[1.0, 0.0]], "trtllm_gen": [[1.0, 1.0]]},
    ],
)
def test_invalid_or_unbalanced_samples_are_rejected(samples) -> None:
    with pytest.raises(ValueError):
        DRIVER._summarize_paired_samples(samples)
