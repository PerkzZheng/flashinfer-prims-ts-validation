# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the sparse MLA workload inventory and qualification gate."""

import json

import pytest
from bench_attention_ts_sparse_mla import cases, main, summarize


def test_model_matrix():
    matrix = cases()
    assert len(matrix) == 480
    assert [c["id"] for c in matrix] == list(range(480))
    prefill = [c for c in matrix if c["phase"] == "prefill"]
    decode = [c for c in matrix if c["phase"] == "decode"]
    assert len(prefill) == 30 and len(decode) == 450
    assert {(c["batch"], c["queries"]) for c in prefill} == {(1, 8192)}
    assert {(c["batch"], c["queries"]) for c in decode} == {
        (b, q) for b in (1, 4, 16, 64, 256) for q in (1, 4, 8)
    }
    assert {(c["topk"], c["heads"], c["dtype"]) for c in matrix} == {
        (k, h, d)
        for k in (512, 1024, 2048)
        for h in (8, 16, 32, 64, 128)
        for d in ("bf16", "fp8")
    }


def test_list_and_select_without_cuda(capsys):
    assert main(["--list", "--indices", "239,0"]) == 0
    assert [r["id"] for r in json.loads(capsys.readouterr().out)] == [0, 239]


@pytest.mark.parametrize("indices", ["", "480", "-1", "bad"])
def test_invalid_selection(indices):
    with pytest.raises(SystemExit) as exc:
        main(["--list", "--indices", indices])
    assert exc.value.code == 2


def result(case_id, speedup, comparator="ok"):
    return {
        "id": case_id,
        "speedup": speedup,
        "backends": {"ts-auto": {"status": "ok"}, "trtllm-gen": {"status": comparator}},
    }


def test_latency_gate_and_unavailable_comparator():
    # A >1 ratio is a speedup. A failing comparator must not improve signoff.
    rows = [result(0, 1.5), result(1, 1 / 1.06), result(2, None, "failed")]
    summary = summarize(rows, 3, 5)
    assert summary["paired"] == 2
    assert summary["cases_over_gate"] == [1]
    assert summary["failed_or_unavailable"] == 1
    assert not summary["passed"]
    assert summarize(rows[:1], 1, 5)["passed"]
    assert not summarize(rows[:1], 2, 5)["passed"]
    assert not summarize([], 0, 5)["passed"]
