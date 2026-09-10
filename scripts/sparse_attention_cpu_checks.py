# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, PerkzZheng.
"""CPU-only inventory, trace-integrity, launch, and summary contract checks."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from run_sparse_attention_suite import ROOT, build_command, main
from sparse_attention_suites import (
    _expand_manifest,
    _select_cases,
    _validate_declared_trace_hashes,
)
from summarize_sparse_attention import render


def test_scrub_initializes_random_bytes_once_without_global_rng():
    # Load only CPU-callable setup helpers, not Torch/FlashInfer/vLLM imports.
    tree = ast.parse(
        (ROOT / "scripts/bench_q_token_kv_block_sparse_attention.py").read_text()
    )
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_l2_cache_size_bytes", "_l2_flush_buffer")
    ]
    buffer, generator = Mock(), Mock()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda _: SimpleNamespace(L2_cache_size=192 * 1024**2)
        ),
        accelerator=SimpleNamespace(current_device_index=lambda: 3),
        Tensor=object,
        uint8="uint8",
        empty=Mock(return_value=buffer),
        Generator=Mock(return_value=generator),
    )
    namespace = {
        "torch": fake_torch,
        "MIN_L2_FLUSH_BYTES": 256 * 1024**2,
        "L2_FLUSH_MULTIPLIER": 2,
        "_L2_FLUSH_BUFFERS": {},
    }
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), "scrub-helpers", "exec"),
        namespace,
    )
    assert namespace["_l2_flush_buffer"]() is buffer
    assert namespace["_l2_flush_buffer"]() is buffer
    fake_torch.empty.assert_called_once_with(384 * 1024**2, dtype="uint8", device=3)
    fake_torch.Generator.assert_called_once_with(device="cuda:3")
    generator.manual_seed.assert_called_once_with(0x5EED_C01D)
    buffer.random_.assert_called_once_with(0, 256, generator=generator)


def test_q5_inventory():
    manifest, cases = _expand_manifest(
        ROOT / "suites/sparse_attention/q5_tp2.json",
        Path("/not-required"),
        validate_traces=False,
    )
    assert len(cases) == 2
    assert {case.batch_size for case in cases} == {16, 32}
    assert all(
        case.tp == 2 and case.seq_len_q == case.group_size == 5 for case in cases
    )
    assert all(case.phase == "grouped_decode_proxy" for case in cases)
    assert manifest["model_len"] == 131072


def test_tp2_inventory():
    _, cases = _expand_manifest(
        ROOT / "suites/sparse_attention/tp2_prefill_decode.json",
        Path("/not-required"),
        validate_traces=False,
    )
    prefill = _select_cases(cases, ["prefill-*"], None)
    decode = _select_cases(cases, ["decode-*"], None)
    assert len(prefill) == 6 and len(decode) == 48
    assert {case.kv_length for case in cases} == {8192, 16384, 32768}
    assert {case.dtype_name for case in cases} == {"bf16", "fp8_e4m3"}
    assert all(
        case.query_layout == "packed" and case.seq_len_q == case.kv_length
        for case in prefill
    )
    assert {case.batch_size for case in decode} == {1, 8, 64, 256}
    assert {case.seq_len_q for case in decode} == {1, 4}
    assert all(case.tp == 2 for case in cases)
    with pytest.raises(ValueError, match="selected no"):
        _select_cases(cases, ["not-a-case"], None)


def test_trace_hashes_are_checked(tmp_path):
    trace = tmp_path / "trace.pt"
    trace.write_bytes(b"tensor fixture")
    digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    _validate_declared_trace_hashes(
        {"files": ["trace.pt"], "sha256": [digest]}, tmp_path
    )
    _validate_declared_trace_hashes(
        {"trace_sets": {"8192": ["trace.pt"]}, "trace_sha256": {"trace.pt": digest}},
        tmp_path,
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        _validate_declared_trace_hashes(
            {"files": ["trace.pt"], "sha256": ["0" * 64]}, tmp_path
        )
    with pytest.raises(FileNotFoundError, match="missing top-k"):
        _validate_declared_trace_hashes(
            {"files": ["absent.pt"], "sha256": [digest]}, tmp_path
        )
    with pytest.raises(ValueError, match="equal length"):
        _validate_declared_trace_hashes({"files": ["trace.pt"], "sha256": []}, tmp_path)
    with pytest.raises(ValueError, match="match exactly"):
        _validate_declared_trace_hashes(
            {"trace_sets": {"8": ["trace.pt"]}, "trace_sha256": {}}, tmp_path
        )


def test_cli_inventory_and_dry_run(capsys, tmp_path):
    assert main(["list"]) == 0
    inventory = capsys.readouterr().out
    assert "q5: 2 cases" in inventory and "tp2: 54 cases" in inventory
    assert "prefill-g4:" not in inventory
    assert (
        main(
            [
                "run",
                "--dry-run",
                "--suite",
                "tp2",
                "--suite",
                "prefill-g4",
                "--source-root",
                str(tmp_path / "flashinfer"),
                "--vllm-root",
                str(tmp_path / "vllm"),
                "--trace-root",
                str(tmp_path / "traces"),
            ]
        )
        == 0
    )
    commands = capsys.readouterr().out
    assert commands.count("--auto-group-size") == 1
    assert "--case-id 'prefill-*'" in commands
    assert "--warmup-iterations 20 --iterations 300" in commands


def test_command_keeps_default_q5_group_and_resume(tmp_path):
    args = argparse.Namespace(
        trace_root=tmp_path,
        source_root=tmp_path,
        vllm_root=tmp_path,
        output_dir=tmp_path,
        device=0,
        warmup_iterations=20,
        iterations=300,
        seed=42,
        resume=True,
    )
    command = build_command(args, "q5")
    assert "--auto-group-size" not in command
    assert "--resume" in command
    assert command[command.index("--seed") + 1] == "42"


@pytest.mark.parametrize(
    "argv",
    [
        ["validate"],
        ["run", "--trace-root", "."],
        ["list", "--iterations", "0"],
        ["list", "--warmup-iterations", "-1"],
    ],
)
def test_cli_rejects_incomplete_arguments(argv):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


def result_fixture():
    return {
        "status": "complete",
        "failures": [],
        "selected_case_ids": ["test"],
        "cases": [
            {
                "case_id": "test",
                "query_group_size": 5,
                "resolved_splits_kv": 4,
                "correctness": {"post_cuda_graph_replay": True},
                "timings": {
                    name: {"mean_us": value}
                    for name, value in (
                        ("metadata", 10),
                        ("prims_ts_attention", 30),
                        ("prims_ts_combined", 35),
                        ("triton_expand_indices", 2),
                        ("triton_attention", 48),
                        ("triton_combined", 50),
                    )
                },
            }
        ],
    }


def test_summary_uses_complete_means_not_component_sums():
    text = render(result_fixture())
    assert "1.429x" in text  # 50 / 35, not (48 + 2) / (30 + 10).
    assert "35.000" in text and "10.000" in text


@pytest.mark.parametrize(
    "mutation", ["partial", "failure", "missing", "duplicate", "unchecked", "nan"]
)
def test_summary_rejects_invalid_evidence(mutation):
    document = copy.deepcopy(result_fixture())
    if mutation == "partial":
        document["status"] = "partial"
    elif mutation == "failure":
        document["failures"] = [{"case_id": "test"}]
    elif mutation == "missing":
        document["selected_case_ids"].append("absent")
    elif mutation == "duplicate":
        document["cases"] *= 2
    elif mutation == "unchecked":
        document["cases"][0]["correctness"]["post_cuda_graph_replay"] = False
    else:
        document["cases"][0]["timings"]["metadata"]["mean_us"] = float("nan")
    with pytest.raises(ValueError):
        render(document)
