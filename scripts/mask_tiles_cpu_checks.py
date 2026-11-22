# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the mask-tile manifest, length expansion and A/B comparison."""

import copy
import json

import pytest
from bench_attention_ts_mask_tiles import DEFAULT_SUITE, main
from mask_tiles_suite import (
    compare_runs,
    comparison_markdown,
    expand_lengths,
    load_suite,
    local_kv_length,
    owned_positions,
)


def test_inventory_covers_every_family(capsys):
    spec = load_suite(DEFAULT_SUITE)
    families = {c["family"] for c in spec["cases"]}
    assert families == {"context", "decode", "mla"}
    assert any(c.get("cp") for c in spec["cases"])
    assert any(c.get("window_left", -1) >= 0 for c in spec["cases"])
    assert any(c.get("variable_window") for c in spec["cases"])
    chosen = spec["cases"][-1]["id"]
    assert main(["--case", chosen, "--list"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == chosen


def test_lengths_are_deterministic_and_aligned():
    spec = {"lo": 1000, "hi": 8000, "seed": 5, "align": 64}
    first = expand_lengths(spec, 16, "kv")
    assert first == expand_lengths(spec, 16, "kv")
    assert all(x % 64 == 0 and 1000 - 64 <= x < 8000 for x in first)
    assert expand_lengths(4, 3, "q") == [4, 4, 4]
    with pytest.raises(ValueError):
        expand_lengths([1, 2], 3, "q")


@pytest.mark.parametrize("cp", [[8, 64, 3], [4, 1, 1], [2, 16, 0], [1, 1, 0]])
def test_local_kv_length_matches_owned_positions(cp):
    world, interleave, rank = cp
    for length in range(0, 700, 13):
        owned = [p for p in range(length) if (p // interleave) % world == rank]
        local = local_kv_length(length, cp)
        assert local == len(owned)
        assert owned_positions(local, cp) == owned


def test_manifest_rejects_bad_cases(tmp_path):
    spec = json.loads(DEFAULT_SUITE.read_text())
    bad = copy.deepcopy(spec)
    bad["cases"][0]["mask"] = "dense"
    bad["cases"][0]["window_left"] = 128
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="sliding windows"):
        load_suite(path)
    bad = copy.deepcopy(spec)
    bad["cases"][1]["id"] = bad["cases"][0]["id"]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unique"):
        load_suite(path)


def _run(commit, rows):
    return {
        "schema_version": 1,
        "suite": {"name": "s"},
        "selected_cases": [r["case_id"] for r in rows],
        "metadata": {
            "source_commit": commit,
            "gpu": "GPU",
            "capability": [10, 3],
            "sms": 148,
            "torch": "t",
            "cuda": "c",
            "dsl": "d",
            "harness_sha256": {"x": "1"},
        },
        "rows": rows,
    }


def _row(case_id, us, out="o", status="passed"):
    return {
        "case_id": case_id,
        "family": "decode",
        "status": status,
        "input_sha256": "i",
        "output_sha256": out,
        "timing": {"median_us": us},
    }


def test_compare_gates_on_median_paired_ratio():
    base = [_run("a", [_row("x", 100.0), _row("y", 50.0)]) for _ in range(3)]
    cand = [_run("b", [_row("x", us), _row("y", 50.0)]) for us in (100.5, 103.0, 100.8)]
    result = compare_runs(base, cand, max_regression=0.01, allow_output_change=False)
    assert result["passed"]
    assert result["rows"][0]["ratio"] == pytest.approx(1.008)
    cand[0]["rows"][0]["timing"]["median_us"] = 102.0
    result = compare_runs(base, cand, max_regression=0.01, allow_output_change=False)
    assert not result["passed"]
    assert result["rows"][0]["issues"] == ["regression"]
    assert "FAIL" in comparison_markdown(result)


def test_compare_requires_identical_outputs_unless_allowed():
    base = [_run("a", [_row("x", 10.0, out="o1")])]
    cand = [_run("b", [_row("x", 10.0, out="o2")])]
    assert not compare_runs(base, cand, max_regression=0.01, allow_output_change=False)[
        "passed"
    ]
    assert compare_runs(base, cand, max_regression=0.01, allow_output_change=True)[
        "passed"
    ]


def test_compare_rejects_mixed_environments_and_failures():
    base = [_run("a", [_row("x", 10.0)])]
    cand = [_run("b", [_row("x", 10.0)])]
    cand[0]["metadata"]["gpu"] = "Other"
    with pytest.raises(ValueError, match="gpu"):
        compare_runs(base, cand, max_regression=0.01, allow_output_change=False)
    cand = [_run("b", [_row("x", 10.0, status="failed")])]
    result = compare_runs(base, cand, max_regression=0.01, allow_output_change=False)
    assert not result["passed"] and result["rows"][0]["issues"] == ["failed"]
