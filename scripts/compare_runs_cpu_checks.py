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

"""CPU-only checks for the cross-run regression gate."""

import copy
import json
import math
from pathlib import Path

import compare_runs
import pytest


def _context_artifact(latencies: list[float]) -> dict[str, object]:
    return {
        "environment": {
            "python": "3.12",
            "torch": "2.10",
            "torch_cuda": "13.0",
            "flashinfer": "0.7",
            "cutlass_dsl": {"distribution_version": "4.7.0"},
            "gpu": {"name": "B200"},
            "source": {
                "benchmark_sha256": "a" * 64,
                "timing_helper_sha256": "e" * 64,
            },
        },
        "protocol": {"timing": "cold-l2"},
        "results": [
            {
                "case_id": "context-row",
                "status": "ok",
                "case": {"head_dim": 128},
                "performance": {"samples_ms_by_round": [latencies]},
            }
        ],
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, allow_nan=True))


def test_compare_rejects_nonfinite_latency(tmp_path: Path) -> None:
    baseline = _context_artifact([1.0, 1.0])
    candidate = _context_artifact([1.0, math.nan])
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write(baseline_path, baseline)
    _write(candidate_path, candidate)

    with pytest.raises(SystemExit, match="nonfinite or nonpositive"):
        compare_runs.main(
            [
                str(baseline_path),
                str(candidate_path),
                "--output",
                str(tmp_path / "comparison.json"),
            ]
        )


def test_fmha_environment_contract_includes_versions_and_reference() -> None:
    provenance = {
        "benchmark": {"sha256": "b" * 64},
        "timing_helper": {"sha256": "e" * 64},
        "python": "3.12",
        "flashinfer_version": "0.7",
        "torch_version": "2.10",
        "torch_cuda_version": "13.0",
        "torch_git_version": "torch-git",
        "gpu": {"name": "B200"},
        "cutlass_dsl": {
            "cutlass_version": "4.7.0",
            "distribution_version": "4.7.0",
        },
        "attention_ts": {"interface": "wrapper", "public_api": "public.ts"},
        "trtllm_gen": {
            "public_api": "public.reference",
            "resolved_manifest_sha256": "c" * 64,
            "flashinfer_cubin_version": "0.7",
        },
        "timing": {"method": "paired-alternating-cold-l2-cuda-graphs"},
    }
    baseline = compare_runs._environment_contract({"provenance": provenance})
    compare_runs._validate_environment_contract(baseline)

    changed_payload = {"provenance": copy.deepcopy(provenance)}
    changed_payload["provenance"]["torch_version"] = "2.11"
    changed = compare_runs._environment_contract(changed_payload)
    assert baseline != changed

    changed_payload = {"provenance": copy.deepcopy(provenance)}
    changed_payload["provenance"]["timing_helper"]["sha256"] = "f" * 64
    changed = compare_runs._environment_contract(changed_payload)
    assert baseline != changed

    changed_payload = {"provenance": copy.deepcopy(provenance)}
    changed_payload["provenance"]["trtllm_gen"]["resolved_manifest_sha256"] = "d" * 64
    changed = compare_runs._environment_contract(changed_payload)
    assert baseline != changed
