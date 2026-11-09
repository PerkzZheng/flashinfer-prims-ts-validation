# SPDX-License-Identifier: Apache-2.0
"""CPU-only inventory, workload and accounting checks for the QSA NVFP4 suites."""

import json

import numpy as np
import pytest
from attention_ts_qsa_nvfp4_fixtures import (
    NVFP4_ROW_BYTES,
    accounting,
    build_indices,
    tail_tokens,
)
from bench_q_token_kv_block_sparse_nvfp4 import (
    BACKENDS,
    ROOT,
    backends_for,
    load_cases,
    main,
    summarize,
    timing_summary,
)
from summarize_qsa_nvfp4 import render

DECODE = ROOT / "suites/qsa_nvfp4/decode.json"
PREFILL = ROOT / "suites/qsa_nvfp4/prefill.json"
DECODE_HKV4 = ROOT / "suites/qsa_nvfp4/decode_hkv4.json"
PREFILL_HKV4 = ROOT / "suites/qsa_nvfp4/prefill_hkv4.json"


def test_inventory():
    """Cover the agreed decode/prefill grid with unique, disjoint IDs."""
    decode, prefill = load_cases(DECODE), load_cases(PREFILL)
    assert len(decode) == 54 and len(prefill) == 12
    assert {(r["tp"], r["batch"], r["mtp"]) for r in decode} == {
        (t, b, m) for t in (1, 2, 4) for b in (1, 4, 8, 16, 32, 64) for m in (3, 4, 7)
    }
    assert all(r["group_size"] == r["mtp"] + 1 for r in decode)
    assert {(r["heads_q"], r["heads_kv"]) for r in decode} == {
        (64, 16),
        (32, 8),
        (16, 4),
    }
    assert {(r["query_tokens"], r["group_size"]) for r in prefill} == {
        (q, g) for q in (8192, 16384) for g in (4, 8)
    }
    assert not {r["id"] for r in decode} & {r["id"] for r in prefill}
    assert all(r["context_tokens"] == 131072 for r in decode + prefill)


def test_hkv4_inventory():
    """The 64/4 suites keep the grid; BF16-Q NVFP4 has no recipe for any row."""
    decode, prefill = load_cases(DECODE_HKV4), load_cases(PREFILL_HKV4)
    assert [r["id"] for r in decode] == [r["id"] for r in load_cases(DECODE)]
    assert [r["id"] for r in prefill] == [r["id"] for r in load_cases(PREFILL)]
    assert {(r["heads_q"], r["heads_kv"]) for r in decode + prefill} == {
        (64, 4),
        (32, 2),
        (16, 1),
    }
    assert all(backends_for(r) == ("nvfp4-fp8mma", "fp8") for r in decode + prefill)
    assert all(backends_for(r) == BACKENDS for r in load_cases(DECODE))


def test_list_and_select(capsys):
    """Inventory selection must not import Torch or need a source checkout."""
    assert main(["--list", "--case", "decode-tp1-b32-mtp4"]) == 0
    assert [r["id"] for r in json.loads(capsys.readouterr().out)] == [
        "decode-tp1-b32-mtp4"
    ]
    with pytest.raises(SystemExit) as exc:
        main(["--list", "--case", "typo"])
    assert exc.value.code == 2


@pytest.mark.parametrize("samples", [[], [0.0], [float("nan")], [float("inf")], [-1.0]])
def test_invalid_timings(samples):
    """Missing or invalid measurements never become speedups."""
    with pytest.raises(ValueError):
        timing_summary(samples)


def complete_row(row_id="row", heads_kv=16):
    return dict(
        id=row_id,
        group_size=5,
        heads_q=64,
        heads_kv=heads_kv,
        status="ok",
        backends={n: dict(status="ok", median_us=1.0) for n in BACKENDS},
        speedups={"fp8": 1.25, "nvfp4-bf16mma": 1.5},
    )


def test_failure_accounting():
    """Partial, duplicated or failed campaigns cannot pass; targets are optional."""
    row = complete_row()
    assert summarize([row], ["row"])["passed"]
    assert not summarize([row], ["row", "missing"])["passed"]
    assert not summarize([row, row], ["row"])["passed"]
    assert not summarize([row], ["row"], min_speedup=1.3)["passed"]
    assert summarize([row], ["row"], min_speedup=1.2)["passed"]
    row["backends"]["fp8"]["status"] = "failed"
    assert not summarize([row], ["row"])["passed"]


def test_unsupported_backend_accounting():
    """Only a backend without a recipe for the row may be absent."""
    row = complete_row(heads_kv=4)
    row["backends"]["nvfp4-bf16mma"] = dict(status="unsupported")
    assert summarize([row], ["row"])["passed"]
    row = complete_row()
    row["backends"]["nvfp4-bf16mma"] = dict(status="unsupported")
    assert not summarize([row], ["row"])["passed"]


def small_case(**overrides):
    case = dict(
        phase="decode",
        batch=3,
        query_tokens=5,
        context_tokens=4096,
        topk_blocks=64,
        overlap=0.7,
        count_current_block=True,
        seed=7,
        heads_q=64,
        heads_kv=16,
    )
    case.update(overrides)
    return case


def test_indices_are_causal_distinct_and_deterministic():
    """Selections stay in each token's causal range, are distinct and reproducible."""
    case = small_case()
    first, second = build_indices(case), build_indices(case)
    assert np.array_equal(first["ids"], second["ids"])
    for row, position in zip(first["ids"], first["positions"], strict=True):
        live = row[row >= 0]
        assert live.size == np.unique(live).size == case["topk_blocks"]
        assert live.max() < (position + 1) // 4
    assert 0.6 < first["mean_adjacent_overlap"] < 0.8
    assert tail_tokens(4095, True) == 0 and tail_tokens(4094, True) == 3


def test_accounting():
    """Unique needed K/V counts once; NVFP4 rows are payload plus scales."""
    case = small_case()
    indices = build_indices(case)
    work = accounting(case, indices)
    unique_tokens = int(indices["unique_blocks"].sum()) * 4
    tokens = case["batch"] * case["query_tokens"]
    assert NVFP4_ROW_BYTES == 144
    assert (
        work["bytes"]["nvfp4-qfp8"] == unique_tokens * 16 * 288 + tokens * 64 * 256 * 3
    )
    assert work["bytes"]["fp8-qfp8"] == unique_tokens * 16 * 512 + tokens * 64 * 256 * 3
    assert work["flops"] == 4 * 256 * 64 * indices["attended_tokens"].sum()


def test_render_keeps_missing_rows_visible():
    """The summary lists failed and missing rows instead of dropping them."""
    row = complete_row("decode-a")
    for name in BACKENDS:
        row["backends"][name].update(achieved_tbps=1.0, achieved_tflops=2.0)
    report = dict(
        schema="qsa-nvfp4-v1",
        cases=[row, dict(id="decode-b", status="failed")],
        requested=["decode-a", "decode-b", "decode-c"],
    )
    text = render([report])
    assert "1.250×" in text and "decode-b | N/A (failed)" in text
    assert "decode-c | N/A (missing)" in text


def test_nvfp4_round_trip():
    """The quantizer's codes decode to E2M1 * E4M3-scale values (CPU, Torch)."""
    torch = pytest.importorskip("torch")
    from attention_ts_qsa_nvfp4_fixtures import dequantize_nvfp4_rows, quantize_nvfp4

    x = torch.tensor([[0.0, 0.5, -1.0, 1.5, 2.0, -3.0, 4.0, 6.0] * 32])
    payload, scales = quantize_nvfp4(x)
    assert payload.shape == (1, 128) and scales.shape == (1, 16)
    assert int(payload[0, 0]) == 0x10 and int(payload[0, 1]) == 0x3A
    rows = torch.cat([payload, scales.view(torch.uint8)], -1)
    assert torch.equal(dequantize_nvfp4_rows(rows), x)
