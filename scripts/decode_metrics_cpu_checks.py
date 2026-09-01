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

"""CPU-only tests for FMHA/MLA decode disposition and aggregation."""

import copy
from typing import Any

import pytest

from .bench_attention_ts_decode import (
    _performance_disposition,
    _summarize_results,
)
from .bench_attention_ts_mla_decode import (
    MLAPerformanceCaseSpec,
    _groups_tokens_heads_q_matrix,
    _validate_compile_reuse,
)
from .bench_attention_ts_mla_decode import (
    _balanced_gap_percent as _mla_balanced_gap_percent,
)
from .bench_attention_ts_mla_decode import (
    _case_contract_record as _mla_case_contract_record,
)
from .bench_attention_ts_mla_decode import (
    _performance_disposition as _mla_performance_disposition,
)
from .bench_attention_ts_mla_decode import (
    _validate_row_case_contract as _validate_mla_row_case_contract,
)
from .routines.attention_ts_benchmark.timing import (
    summarize_paired_two_order_cycles,
)


def _paired_cycles(gap_percent: float) -> dict[str, Any]:
    return {
        "method": "alternating-order-balanced-total-duration-ratio",
        "sample_count_per_backend": 4,
        "cycle_count": 2,
        "gap_percent_total_duration_ratio": gap_percent,
        "cycle_gap_percent_median": gap_percent,
        "cycle_gap_percent_p95": gap_percent,
        "cycle_gap_percent_min": gap_percent,
        "cycle_gap_percent_max": gap_percent,
    }


def test_paired_disposition_uses_balanced_total_duration_ratio() -> None:
    passed = _performance_disposition(4.0, _paired_cycles(3.0), 5.0)
    assert passed == {
        "threshold_percent": 5.0,
        "raw_gap_percent": 4.0,
        "balanced_gap_percent": 3.0,
        "raw_exceeds_threshold": False,
        "balanced_exceeds_threshold": False,
        "classification": "pass",
        "gate_metric": ("paired_two_order_cycles.gap_percent_total_duration_ratio"),
    }

    raw_exception = _performance_disposition(8.0, _paired_cycles(2.0), 5.0)
    assert raw_exception["raw_exceeds_threshold"] is True
    assert raw_exception["balanced_exceeds_threshold"] is False
    assert raw_exception["classification"] == "raw-median-exception-balanced-pass"

    balanced_regression = _performance_disposition(4.0, _paired_cycles(6.0), 5.0)
    assert balanced_regression["raw_exceeds_threshold"] is False
    assert balanced_regression["balanced_exceeds_threshold"] is True
    assert balanced_regression["classification"] == "regression"


def test_mla_disposition_reads_aggregate_instead_of_cycle_median() -> None:
    paired = _paired_cycles(2.0)
    paired["cycle_gap_percent_median"] = 20.0
    result = {"paired_two_order_cycles": paired}

    assert _mla_balanced_gap_percent(result) == 2.0
    disposition = _mla_performance_disposition(
        8.0, _mla_balanced_gap_percent(result), 5.0
    )
    assert disposition["classification"] == "raw-median-exception-balanced-pass"
    assert disposition["gate_metric"] == (
        "paired_two_order_cycles.gap_percent_total_duration_ratio"
    )


def _valid_mla_resume_row() -> tuple[dict[str, Any], dict[str, Any]]:
    spec = MLAPerformanceCaseSpec(
        case_id="mla-fp8-b4-h32-kv2048-ps32",
        num_heads=32,
        batch_size=4,
        max_seq_len=2048,
        dtype_name="fp8",
        seed=17,
    )
    expected = _mla_case_contract_record(spec, "source-session", "wrapper")
    paired = summarize_paired_two_order_cycles(
        [1.04, 0.96, 1.02, 0.98],
        [1.0, 1.0, 1.0, 1.0],
    )
    balanced_gap = paired["gap_percent_total_duration_ratio"]
    row = {
        "status": "ok",
        "case_id": spec.case_id,
        "case_contract": expected["contract"],
        "case_contract_sha256": expected["sha256"],
        "shape": {
            "batch_size": 4,
            "q_len": 1,
            "num_qo_heads": 32,
            "qk_nope_head_dim": 512,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "page_size": 32,
            "max_seq_len": 2048,
            "input_dtype": "fp8",
            "output_dtype": "bf16",
            "fixture_seed": 17,
        },
        "attention_ts": {"interface": "wrapper"},
        "trtllm_gen": {
            "backend": "trtllm-gen",
            "enable_pdl": False,
            "internal_shape_auto_selector": True,
        },
        "paired_two_order_cycles": paired,
        "performance_disposition": _mla_performance_disposition(0.0, balanced_gap, 5.0),
    }
    return row, expected


def test_mla_resume_accepts_the_signed_aggregate_contract() -> None:
    row, expected = _valid_mla_resume_row()
    _validate_mla_row_case_contract(row, expected)


def test_mla_resume_rejects_legacy_median_gate_and_corrupt_raw_samples() -> None:
    row, expected = _valid_mla_resume_row()
    legacy = copy.deepcopy(row)
    paired = legacy["paired_two_order_cycles"]
    paired["gap_percent_median"] = paired.pop("gap_percent_total_duration_ratio")
    legacy["performance_disposition"]["gate_metric"] = (
        "paired_two_order_cycles.gap_percent_median"
    )
    with pytest.raises(ValueError, match="paired aggregate"):
        _validate_mla_row_case_contract(legacy, expected)

    corrupt = copy.deepcopy(row)
    corrupt["paired_two_order_cycles"]["raw_samples_ms"]["attention_ts"].pop()
    with pytest.raises(ValueError, match="raw samples"):
        _validate_mla_row_case_contract(corrupt, expected)


def test_mla_groups_tokens_heads_q_catalog_is_focused_and_unique() -> None:
    specs = _groups_tokens_heads_q_matrix()

    assert len(specs) == 22
    assert len({spec.case_id for spec in specs}) == len(specs)
    assert {spec.num_heads for spec in specs} == {12, 24, 48, 96}
    assert {spec.dtype_name for spec in specs} == {"bf16", "fp8"}
    assert {spec.num_heads * spec.seq_len_q for spec in specs} >= {12, 48, 96}
    assert all(((spec.max_seq_len + 31) // 32) % 4 == 0 for spec in specs)

    split_anchors = [spec for spec in specs if spec.max_seq_len == 32896]
    assert {(spec.num_heads, spec.seq_len_q) for spec in split_anchors} == {
        (12, 1),
        (96, 1),
    }
    assert {spec.dtype_name for spec in split_anchors} == {"bf16", "fp8"}

    reuse_groups: dict[str, list[MLAPerformanceCaseSpec]] = {}
    for spec in specs:
        if spec.compile_reuse_group is not None:
            reuse_groups.setdefault(spec.compile_reuse_group, []).append(spec)
    assert len(reuse_groups) == 2
    for group in reuse_groups.values():
        assert [spec.batch_size for spec in group] == [3, 4]
        assert (
            len({(spec.num_heads, spec.seq_len_q, spec.max_seq_len) for spec in group})
            == 1
        )


def test_mla_compile_reuse_gate_applies_only_after_group_anchor() -> None:
    first, second = [
        spec
        for spec in _groups_tokens_heads_q_matrix()
        if spec.compile_reuse_group == "bf16-h12-q1-kv1024"
    ]
    completed: set[str] = set()
    compiled = {
        "status": "ok",
        "attention_ts": {
            "compiled_during_setup": True,
            "compiled_on_first_call": False,
        },
    }
    reused = {
        "status": "ok",
        "attention_ts": {
            "compiled_during_setup": False,
            "compiled_on_first_call": False,
        },
    }

    _validate_compile_reuse(first, compiled, completed)
    _validate_compile_reuse(second, reused, completed)
    with pytest.raises(AssertionError, match="batch-only topology change recompiled"):
        _validate_compile_reuse(second, compiled, completed)


def test_unpaired_disposition_is_explicitly_raw_only() -> None:
    passed = _performance_disposition(4.0, None, 5.0)
    regressed = _performance_disposition(6.0, None, 5.0)

    assert passed["balanced_gap_percent"] is None
    assert passed["balanced_exceeds_threshold"] is None
    assert passed["classification"] == "raw-only-pass"
    assert passed["gate_metric"] == "attention_ts_gap_percent_vs_trtllm_gen"
    assert regressed["classification"] == "raw-only-regression"
    assert regressed["raw_exceeds_threshold"] is True


def test_threshold_equality_is_not_a_regression() -> None:
    paired = _performance_disposition(5.0, _paired_cycles(5.0), 5.0)
    raw_only = _performance_disposition(5.0, None, 5.0)

    assert paired["raw_exceeds_threshold"] is False
    assert paired["balanced_exceeds_threshold"] is False
    assert paired["classification"] == "pass"
    assert raw_only["raw_exceeds_threshold"] is False
    assert raw_only["classification"] == "raw-only-pass"


def test_summary_aggregates_raw_balanced_exception_gate_and_error_counts() -> None:
    def row(
        case_id: str,
        raw_gap_percent: float,
        balanced_gap_percent: float | None,
    ) -> dict[str, Any]:
        paired = (
            None
            if balanced_gap_percent is None
            else _paired_cycles(balanced_gap_percent)
        )
        return {
            "status": "ok",
            "case_id": case_id,
            "performance_disposition": _performance_disposition(
                raw_gap_percent, paired, 5.0
            ),
        }

    results = [
        row("balanced-pass", 4.0, 4.0),
        row("raw-exception", 8.0, 2.0),
        row("balanced-only-regression", 4.0, 6.0),
        row("both-regress", 7.0, 9.0),
        row("raw-only-pass", 4.0, None),
        row("raw-only-regress", 6.0, None),
        {"status": "error", "case_id": "failed-case"},
    ]

    summary = _summarize_results(results, 5.0)

    assert summary == {
        "row_count": 7,
        "successful_row_count": 6,
        "error_row_count": 1,
        "error_case_ids": ["failed-case"],
        "gap_threshold_percent": 5.0,
        "raw_independent_median_regression_row_count": 3,
        "order_balanced_total_duration_evaluable_row_count": 4,
        "order_balanced_total_duration_regression_row_count": 2,
        "raw_median_exception_row_count": 1,
        "gate_regression_row_count": 3,
        "worst_order_balanced_case_id": "both-regress",
        "worst_order_balanced_gap_percent": 9.0,
    }
