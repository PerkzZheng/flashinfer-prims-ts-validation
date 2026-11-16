# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for workload selection, failure accounting, and comparison safety."""

import copy
import json

import pytest
from audit_attention_ts_decode_dispatch import compare_policy
from bench_attention_ts_nvfp4_decode import ROOT, main
from nvfp4_decode_suite import load_suite, policy_cases, result_summary, timing_summary
from summarize_nvfp4_decode import paired_rows, validate_results


def test_inventory_and_selection(capsys):
    """Retain the full session workload, including small batches and Q9."""
    expected = {"issue9311": 16, "issue6979": 64, "q9": 1, "fp8_controls": 6}
    for name, count in expected.items():
        spec = load_suite(ROOT / f"suites/nvfp4_decode/{name}.json")
        assert len(spec["cases"]) == count
        chosen = spec["cases"][-1]["id"]
        assert (
            main(
                [
                    "--suite",
                    str(ROOT / f"suites/nvfp4_decode/{name}.json"),
                    "--case",
                    chosen,
                    "--list",
                ]
            )
            == 0
        )
        assert [r["id"] for r in json.loads(capsys.readouterr().out)["cases"]] == [
            chosen
        ]
    with pytest.raises(ValueError):
        main(["--list", "--case", "missing"])
    cases = policy_cases()
    assert len(cases) == len(set(cases)) == 2016
    assert sum(c[0] == "fp8" for c in cases) == 1824


@pytest.mark.parametrize("samples", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_timings(samples):
    """Invalid durations must never become speedup evidence."""
    with pytest.raises(ValueError):
        timing_summary(samples)


def result():
    """A small complete report whose invariants can be broken independently."""
    case = {"id": "one"}
    check = {"oracle_pass": True, "all_outputs_finite": True}
    row = {
        "case": case,
        "backend": "prims_nvfp4",
        "status": "ok",
        "timing": timing_summary([1.0, 2.0, 3.0]),
        "before": check,
        "after": check,
        "input_sample_sha256": "input",
        "out_dtype": "torch.float16",
        "mask": "causal",
    }
    return {
        "rows": [row],
        "suite": {"cases": [case]},
        "controls": {"protocol": "cold-cupti"},
        "selected_cases": ["one"],
        "backends": ["prims_nvfp4"],
        "summary": {},
        "metadata": dict(
            gpu="B200",
            capability=[10, 0],
            sms=148,
            torch="x",
            cuda="x",
            dsl="x",
            harness_sha256="x",
            audit_sha256="x",
        ),
    }


def test_missing_duplicate_and_failed_rows():
    """A partial run, missing reference, or failed row cannot pass."""
    data = result()
    assert validate_results(data)["passed"]
    row = data["rows"][0]
    assert not result_summary([], [row["case"]], ["prims_nvfp4"])["passed"]
    assert not result_summary([row, row], [row["case"]], ["prims_nvfp4"])["passed"]
    assert not result_summary([row], [row["case"]], ["prims_nvfp4", "trt_fp8"])[
        "passed"
    ]
    row["status"] = "failed"
    assert not validate_results(data)["passed"]
    row["status"] = "ok"
    row["after"] = {"all_outputs_finite": True, "oracle_pass": False}
    with pytest.raises(ValueError):
        validate_results(data)


@pytest.mark.parametrize("field", ["suite", "controls", "selected_cases", "backends"])
def test_comparison_contract_drift(field):
    """Different workloads or sampling protocols must not produce a ratio."""
    old = result()
    new = copy.deepcopy(old)
    new[field] = "changed"
    with pytest.raises(ValueError):
        paired_rows(old, new)


def test_comparison_input_and_capacity_drift():
    """Reject different fixtures/hardware and report genuine dispatch changes."""
    old = result()
    new = copy.deepcopy(old)
    new["rows"][0]["input_sample_sha256"] = "other"
    with pytest.raises(ValueError):
        paired_rows(old, new)
    new = copy.deepcopy(old)
    new["metadata"]["sms"] += 1
    with pytest.raises(ValueError):
        paired_rows(old, new)
    with pytest.raises(ValueError):
        compare_policy(old, new)
    new = copy.deepcopy(old)
    new["rows"][0]["policy"] = {"splits_kv": 2}
    assert len(compare_policy(old, new)) == 1


def test_empty_selection_and_stale_timing_summary():
    """Empty runs and edited aggregates cannot be published as qualified evidence."""
    data = result()
    data["rows"] = []
    data["selected_cases"] = []
    with pytest.raises(ValueError):
        validate_results(data)
    assert not result_summary([], [], [])["passed"]
    data = result()
    data["rows"][0]["timing"]["median_us"] = 0.1
    with pytest.raises(ValueError):
        validate_results(data)
