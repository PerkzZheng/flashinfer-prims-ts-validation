# SPDX-License-Identifier: Apache-2.0
"""CPU-only workload and failure-accounting checks for scaled MLA comparisons."""

import json

import pytest
from bench_attention_ts_scaled_mla import (
    ROOT,
    load_cases,
    main,
    summarize,
    timing_summary,
)


def test_workload_inventory():
    """Keep the requested large/small head coverage disjoint and complete."""
    large = load_cases(ROOT / "suites/scaled_mla/decode.json")
    small = load_cases(ROOT / "suites/scaled_mla/small_heads.json")
    assert len(large) == 48 and len(small) == 72
    assert {r["heads"] for r in large} == {64, 128}
    assert {r["heads"] for r in small} == {8, 16, 32}
    assert not {r["id"] for r in large} & {r["id"] for r in small}
    assert {(r["batch"], r["queries"]) for r in large} == {
        (b, q) for b in (4, 16, 32, 64) for q in (4, 8)
    }
    assert all(
        r["context_tokens"] == 65536 and r["topk"] == 2048 and r["page_size"] == 64
        for r in large + small
    )


def test_list_and_select(capsys):
    """Inventory selection must not import Torch or require an installed source."""
    assert main(["--list", "--case", "d576-b4-q4-h64"]) == 0
    assert [r["id"] for r in json.loads(capsys.readouterr().out)] == ["d576-b4-q4-h64"]
    with pytest.raises(SystemExit) as exc:
        main(["--list", "--case", "typo"])
    assert exc.value.code == 2


@pytest.mark.parametrize("samples", [[], [0.0], [float("nan")], [float("inf")], [-1.0]])
def test_invalid_timings(samples):
    """Missing/invalid measurements must never become reported speedups."""
    with pytest.raises(ValueError):
        timing_summary(samples)


def paired_row():
    """A complete synthetic result for testing comparison accounting."""
    return dict(
        id="row",
        variant="d576",
        status="ok",
        backends={
            n: dict(status="ok") for n in ("prims-nvfp4", "prims-fp8", "flashmla")
        },
        speedups={"prims-fp8": 1.1, "flashmla": 1.3},
    )


def test_missing_failed_and_duplicate_rows_cannot_pass():
    """A partial campaign or failed comparator cannot be mistaken for a win."""
    row = paired_row()
    assert summarize([row], ["row"], without_flashmla=False)["passed"]
    assert not summarize([row], ["row", "missing"], without_flashmla=False)["passed"]
    assert not summarize([row, row], ["row"], without_flashmla=False)["passed"]
    row["backends"]["flashmla"]["status"] = "failed"
    assert not summarize([row], ["row"], without_flashmla=False)["passed"]
    assert summarize([row], ["row"], without_flashmla=True)["passed"]


def test_optional_target_and_unsupported_reference():
    """A declared native-format gap is explicit; target checks use available pairs."""
    row = paired_row()
    assert not summarize([row], ["row"], without_flashmla=False, min_speedup=1.2)[
        "passed"
    ]
    row["variant"] = "d512-nvfp4-extra"
    row["backends"]["flashmla"] = dict(status="unsupported")
    row["speedups"].pop("flashmla")
    summary = summarize([row], ["row"], without_flashmla=False)
    assert summary["passed"] and summary["flashmla_unsupported"] == ["row"]
    assert summarize([row], ["row"], without_flashmla=False, min_speedup=1.05)["passed"]
