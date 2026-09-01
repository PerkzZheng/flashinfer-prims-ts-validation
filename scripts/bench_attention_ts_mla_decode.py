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

"""Compare public Attention-TS against an explicit public MLA reference.

The default ``signoff`` suite contains exactly 100 SQ1 rows; ``--q-len 4``
runs the same 100-row geometry as a distinct grouped-Q catalog::

    Hq(8,16,32,64,128) * B(1,4,8,64,128)
    * maxKV(2048,8192) * input(BF16,E4M3)

The focused 22-row ``groups-tokens-heads-q`` suite compares PrimTS auto-dispatch
with CuTe DSL and covers the public flat-query
contract added for non-power-of-two query-head counts. It crosses heads
12/24/48/96 with equivalent 48-row 1CTA and 96-row 2CTA geometries, exercises
both split-reduction families, and verifies that otherwise identical B3/B4
plans reuse one compiled topology. It intentionally records, but does not pin,
the exact automatic policy selected for the performance rows.

Every row is built once. Both public backends receive the same query/cache
storage, two-dimensional page table, runtime sequence lengths, fused BMM
scales, and backend-neutral FP32 reference. Output is BF16 and page size is
32. Batches larger than one have deterministic unequal runtime lengths.

Timing captures one public call in a separate CUDA graph for each backend and
alternates replay order. The signoff suite scrubs 2x L2 with non-compressible
data immediately before every replay; the focused feature suite uses the hot
graph contract from the feature's public-backend campaign. Consecutive
opposite-order samples form balanced two-order cycles. The regression gate is
the ratio of total TS duration to total reference duration over all complete
cycles; per-cycle percentiles remain diagnostics. Planning, reference
construction, and first calls are reported but never timed. The default
``wrapper`` PrimTS interface preserves historical runs;
``--prims-ts-interface standalone`` selects the public caller-workspace API.

Examples
--------
Run a short public-interface smoke::

    python benchmarks/bench_attention_ts_mla_decode.py \
      --case mla-bf16-b1-h8-kv2048-ps32 --warmup-iters 2 --iters 10

List the distinct 100-row grouped-Q catalog::

    python benchmarks/bench_attention_ts_mla_decode.py --q-len 4 --list-cases

Run the complete signoff matrix and write durable outputs::

    python benchmarks/bench_attention_ts_mla_decode.py \
      --expect-cases 100 \
      --json-output /tmp/mla.json \
      --csv-output /tmp/mla.csv \
      --markdown-output /tmp/mla.md

Resume an interrupted run only when its source and timing signature match::

    python benchmarks/bench_attention_ts_mla_decode.py \
      --resume /tmp/mla.json --csv-output /tmp/mla.csv \
      --markdown-output /tmp/mla.md

Freshly rerun only rows above the previous 5% threshold after a source edit::

    python benchmarks/bench_attention_ts_mla_decode.py \
      --regressions-from /tmp/mla.json --gap-threshold 5 \
      --json-output /tmp/mla-rerun.json

Pitfalls, regressions, limitations, and fallbacks
-------------------------------------------------
* ``qk_nope_head_dim`` is 512 for this absorbed MLA interface. It is distinct
  from the pre-absorption 128-wide query head used in the scale denominator.
* TRTLLM-gen signoff rows own a separate, initially zeroed 128 MiB workspace and a
  shape-sized counter buffer. The counter buffer must self-reset; it is never
  zeroed inside timing.
* ``flashinfer.autotune(False)`` disables only Python cross-backend/bucket
  profiling. TRTLLM-gen tactic -1 still runs its internal shape auto-selector,
  while PrimTS's independent automatic policy remains enabled.
* B1 supplies a runtime sequence-length tensor but cannot be ragged relative
  to another request. B>1 covers unequal, non-page-aligned lengths.
* This benchmark covers one fixed query length per invocation, bottom-right
  causal decode, latent/RoPE 512/64, page32, shared page indices, BF16/E4M3
  input, and BF16 output. Dense/causal equivalence applies only to SQ1.
* No backend fallback is allowed: TS uses the selected public PrimTS interface;
  the signoff suite explicitly selects TRTLLM-gen and the focused feature suite
  explicitly selects monolithic CuTe DSL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.routines.attention_ts_benchmark import (  # noqa: E402, I001
    atomic_write_json as _write_json_atomic,
    atomic_write_text as _write_text_atomic,
    cache_info_record as _cache_info_dict,
    effective_command,
    first_call as _first_call,
    git_value,
    prepare_cold_l2_scrubber as _prepare_cold_l2_scrubber,
    sha256_file as _sha256_file,
    time_paired_cold_l2_cuda_graphs as _time_paired_cold_l2_cuda_graphs,
    time_paired_cuda_graphs as _time_paired_cuda_graphs,
)

_HEADS = (8, 16, 32, 64, 128)
_BATCH_SIZES = (1, 4, 8, 64, 128)
_MAX_SEQ_LENS = (2048, 8192)
_GROUPED_QUERY_HEADS = (12, 24, 48, 96)
_GROUPED_QUERY_BATCH_SIZES = (3, 4)
_GROUPED_QUERY_MAX_SEQ_LENS = (512, 1024, 32896)
_SUITES = ("signoff", "groups-tokens-heads-q")
_DTYPES = ("bf16", "fp8")
_PAGE_SIZE = 32
_Q_LEN = 1
_QK_NOPE_HEAD_DIM = 512
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_TRTLLM_WORKSPACE_BYTES = 128 * 1024 * 1024
_TRTLLM_BACKEND = "trtllm-gen"
_CUTE_DSL_BACKEND = "cute-dsl"
_TRTLLM_ENABLE_PDL = False
_TRTLLM_IS_VAR_SEQ = True
_TRTLLM_USES_SHARED_PAGED_KV_IDX = True
_TRTLLM_SPARSE_MLA_TOP_K = 0
_TRTLLM_PYTHON_AUTOTUNE = False
_TRTLLM_INTERNAL_SHAPE_AUTO_SELECTOR = True
_PRIMS_TS_INTERFACES = ("wrapper", "standalone")
_EXPECTED_CUTLASS_DSL_VERSION = "4.7.0"
_ORACLE_CUDA_MATMUL_FP32_PRECISION = "ieee"
_REFERENCE_TOLERANCES = {
    "bf16": (1e-2, 5e-4, 2e-2),
    "fp8": (1e-1, 2e-3, 1e-1),
}
_SCHEMA_VERSION = 8
_PAIRED_GATE_METRIC = "paired_two_order_cycles.gap_percent_total_duration_ratio"


def _reference_backend_for_suite(suite: str) -> str:
    return _CUTE_DSL_BACKEND if suite == "groups-tokens-heads-q" else _TRTLLM_BACKEND


def _reference_label(backend: str) -> str:
    return backend.replace("-", "_")


def _reference_display_name(backend: str) -> str:
    return "TRTLLM-gen" if backend == _TRTLLM_BACKEND else "CuTe DSL"


@dataclass(frozen=True)
class MLAPerformanceCaseSpec:
    """One deterministic row in the public MLA performance matrix."""

    case_id: str
    num_heads: int
    batch_size: int
    max_seq_len: int
    dtype_name: str
    seed: int
    seq_len_q: int = _Q_LEN
    compile_reuse_group: str | None = None


def _stable_seed(case_id: str) -> int:
    return (
        int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:4], "little")
        & 0x7FFFFFFF
    )


def _timing_cache_mode_for_suite(suite: str) -> str:
    return "hot" if suite == "groups-tokens-heads-q" else "cold-l2"


def _case_id(
    dtype_name: str,
    batch_size: int,
    num_heads: int,
    max_seq_len: int,
    seq_len_q: int = _Q_LEN,
):
    base = f"mla-{dtype_name}-b{batch_size}-h{num_heads}-kv{max_seq_len}-ps{_PAGE_SIZE}"
    return base if seq_len_q == 1 else f"{base}-q{seq_len_q}"


def _full_matrix(seq_len_q: int = _Q_LEN) -> list[MLAPerformanceCaseSpec]:
    if seq_len_q <= 0:
        raise ValueError("seq_len_q must be positive")
    specs = []
    for dtype_name in _DTYPES:
        for max_seq_len in _MAX_SEQ_LENS:
            for num_heads in _HEADS:
                for batch_size in _BATCH_SIZES:
                    case_id = _case_id(
                        dtype_name,
                        batch_size,
                        num_heads,
                        max_seq_len,
                        seq_len_q,
                    )
                    specs.append(
                        MLAPerformanceCaseSpec(
                            case_id=case_id,
                            num_heads=num_heads,
                            batch_size=batch_size,
                            max_seq_len=max_seq_len,
                            dtype_name=dtype_name,
                            seed=_stable_seed(case_id),
                            seq_len_q=seq_len_q,
                        )
                    )
    assert len(specs) == 100
    return specs


def _groups_tokens_heads_q_matrix() -> list[MLAPerformanceCaseSpec]:
    """Return focused public-auto rows without duplicating the signoff grid."""

    shapes = (
        # Hold B and K fixed across equal-row factorizations so the only
        # structural change is the logical flat-query geometry.
        # Equal 48-row factorizations exercise 1CTA without structural Q padding.
        (4, 12, 4, 512, None),
        (4, 24, 2, 512, None),
        (4, 48, 1, 512, None),
        # Equal 96-row factorizations exercise a partial M128 tile in 2CTA.
        (4, 12, 8, 512, None),
        (4, 24, 4, 512, None),
        (4, 48, 2, 512, None),
        (4, 96, 1, 512, None),
        # Long-K anchors cover each family's split-reduction output path.
        (4, 12, 1, 32896, None),
        (4, 96, 1, 32896, None),
    )
    specs = []
    for dtype_name in _DTYPES:
        for batch_size, num_heads, seq_len_q, max_seq_len, reuse_group in shapes:
            case_id = _case_id(
                dtype_name,
                batch_size,
                num_heads,
                max_seq_len,
                seq_len_q,
            )
            specs.append(
                MLAPerformanceCaseSpec(
                    case_id=case_id,
                    num_heads=num_heads,
                    batch_size=batch_size,
                    max_seq_len=max_seq_len,
                    dtype_name=dtype_name,
                    seed=_stable_seed(case_id),
                    seq_len_q=seq_len_q,
                    compile_reuse_group=reuse_group,
                )
            )

    # Batch is a runtime value. Keep two same-topology BF16 pairs adjacent so
    # the second plan can prove cache reuse without turning policy into a test.
    for num_heads in (12, 96):
        reuse_group = f"bf16-h{num_heads}-q1-kv1024"
        for batch_size in (3, 4):
            case_id = _case_id("bf16", batch_size, num_heads, 1024, 1)
            specs.append(
                MLAPerformanceCaseSpec(
                    case_id=case_id,
                    num_heads=num_heads,
                    batch_size=batch_size,
                    max_seq_len=1024,
                    dtype_name="bf16",
                    seed=_stable_seed(case_id),
                    compile_reuse_group=reuse_group,
                )
            )

    if len(specs) != 22 or len({spec.case_id for spec in specs}) != len(specs):
        raise AssertionError("groups-tokens-heads-q must contain 22 unique rows")
    return specs


def _catalog(suite: str, seq_len_q: int) -> list[MLAPerformanceCaseSpec]:
    if suite == "signoff":
        return _full_matrix(seq_len_q)
    if suite == "groups-tokens-heads-q":
        return _groups_tokens_heads_q_matrix()
    raise ValueError(f"unknown suite {suite!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare public Attention-TS and an explicit MLA reference with "
            "paired cold-L2 one-call CUDA graphs."
        )
    )
    parser.add_argument(
        "--suite",
        choices=_SUITES,
        default="signoff",
        help=(
            "Case catalog: the 100-row signoff product or the focused 22-row "
            "non-power-of-two grouped-query regression suite."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        metavar="ID",
        help="Exact case ID to run; repeat to select multiple rows.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List selected suite case IDs and exit.",
    )
    parser.add_argument(
        "--q-len",
        type=int,
        default=_Q_LEN,
        help=(
            "Fixed query length for all 100 rows. SQ1 preserves historical "
            "case IDs; grouped-Q rows append -qN (default: 1)."
        ),
    )
    parser.add_argument(
        "--num-heads",
        action="append",
        type=int,
        choices=tuple(sorted(set(_HEADS) | set(_GROUPED_QUERY_HEADS))),
        help="Query-head filter; repeat to select multiple values.",
    )
    parser.add_argument(
        "--batch-size",
        action="append",
        type=int,
        choices=tuple(sorted(set(_BATCH_SIZES) | set(_GROUPED_QUERY_BATCH_SIZES))),
        help="Batch-size filter; repeat to select multiple values.",
    )
    parser.add_argument(
        "--kv-len",
        "--max-seq-len",
        action="append",
        dest="max_seq_lens",
        type=int,
        choices=tuple(sorted(set(_MAX_SEQ_LENS) | set(_GROUPED_QUERY_MAX_SEQ_LENS))),
        help="Maximum KV-length filter; repeat to select multiple values.",
    )
    parser.add_argument(
        "--dtype",
        action="append",
        dest="dtype_names",
        choices=_DTYPES,
        help="Input dtype filter; repeat to select BF16 and FP8.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--prims-ts-interface",
        choices=_PRIMS_TS_INTERFACES,
        default="wrapper",
        help=(
            "PrimTS public interface to benchmark: planned wrapper or direct "
            "caller-workspace standalone call (default: wrapper)."
        ),
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Untimed paired graph replays per backend (default: 10).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help=(
            "Timed paired cold-L2 samples per backend. Must be even and at "
            "least 2 so opposite replay orders form complete cycles (default: 200)."
        ),
    )
    parser.add_argument(
        "--expect-cases",
        type=int,
        help="Fail before GPU work unless selection has exactly this many rows.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed row and continue instead of raising immediately.",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=5.0,
        help="Regression threshold relative to the suite reference (default: 5%%).",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help=(
            "Return status 2 after writing outputs if any successful row's "
            "order-balanced total-duration gap exceeds the threshold."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Resume completed rows from an existing JSON file. Source, matrix, "
            "and timing signatures must match exactly."
        ),
    )
    parser.add_argument(
        "--regressions-from",
        type=Path,
        help=(
            "Freshly rerun prior errors and rows whose order-balanced total-"
            "duration TS gap exceeds --gap-threshold. Legacy rows without the "
            "paired metric are rerun conservatively; source signatures may differ."
        ),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def _dedupe(values: Sequence[Any] | None) -> set[Any] | None:
    return None if not values else set(values)


def _balanced_gap_percent(result: dict[str, Any]) -> float | None:
    """Return the order-balanced total-duration gate, if it is valid."""

    paired = result.get("paired_two_order_cycles")
    if not isinstance(paired, dict):
        return None
    value = paired.get("gap_percent_total_duration_ratio")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and abs(value) != float("inf") else None


def _raw_gap_percent(result: dict[str, Any]) -> float | None:
    value = result.get(
        "attention_ts_gap_percent_vs_reference",
        result.get("attention_ts_gap_percent_vs_trtllm_gen"),
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and abs(value) != float("inf") else None


def _performance_disposition(
    raw_gap_percent: float,
    balanced_gap_percent: float,
    threshold_percent: float,
) -> dict[str, Any]:
    """Describe raw and paired outcomes while gating on total duration."""

    raw_exceeds = raw_gap_percent > threshold_percent
    balanced_exceeds = balanced_gap_percent > threshold_percent
    if balanced_exceeds:
        classification = "regression"
    elif raw_exceeds:
        classification = "raw-median-exception-balanced-pass"
    else:
        classification = "pass"
    return {
        "threshold_percent": threshold_percent,
        "raw_gap_percent": raw_gap_percent,
        "balanced_gap_percent": balanced_gap_percent,
        "raw_exceeds_threshold": raw_exceeds,
        "balanced_exceeds_threshold": balanced_exceeds,
        "classification": classification,
        "gate_metric": _PAIRED_GATE_METRIC,
    }


def _select_specs(
    parser: argparse.ArgumentParser, args
) -> list[MLAPerformanceCaseSpec]:
    specs = _catalog(args.suite, args.q_len)
    by_id = {spec.case_id: spec for spec in specs}
    if args.case_ids:
        unknown = sorted(set(args.case_ids) - set(by_id))
        if unknown:
            parser.error("unknown --case value(s): " + ", ".join(unknown))
        requested_ids = set(args.case_ids)
        specs = [spec for spec in specs if spec.case_id in requested_ids]

    heads = _dedupe(args.num_heads)
    batches = _dedupe(args.batch_size)
    max_seq_lens = _dedupe(args.max_seq_lens)
    dtypes = _dedupe(args.dtype_names)
    specs = [
        spec
        for spec in specs
        if (heads is None or spec.num_heads in heads)
        and (batches is None or spec.batch_size in batches)
        and (max_seq_lens is None or spec.max_seq_len in max_seq_lens)
        and (dtypes is None or spec.dtype_name in dtypes)
    ]

    if args.regressions_from is not None:
        prior = _load_json(args.regressions_from)
        regression_ids = {
            row["case_id"]
            for row in prior.get("results", [])
            if row.get("status") != "ok"
            or _balanced_gap_percent(row) is None
            or _balanced_gap_percent(row) > args.gap_threshold
        }
        specs = [spec for spec in specs if spec.case_id in regression_ids]

    if args.expect_cases is not None and len(specs) != args.expect_cases:
        parser.error(f"selected {len(specs)} rows, expected {args.expect_cases}")
    return specs


def _dtype_from_name(dtype_name: str, torch):
    return {
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
    }[dtype_name]


def _cuda_runtime_error(error: BaseException) -> BaseException:
    """Add a concrete remedy for the cached TRT module's CTK13 dependency."""

    message = str(error)
    if "libcudart.so.13" not in message:
        return error
    cuda_home = os.environ.get("CUDA_HOME")
    expected = "$CUDA_HOME/lib64/libcudart.so.13"
    if cuda_home:
        expected = str(Path(cuda_home) / "lib64/libcudart.so.13")
    return RuntimeError(
        "TRTLLM-gen could not load libcudart.so.13. The cached module needs "
        "the CTK13 runtime on the dynamic-loader path. Before starting this "
        "benchmark, set `LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH` "
        f"(expected library: {expected}). The benchmark does not mutate the "
        "process environment. Original error: "
        f"{message}"
    )


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_value(*args: str) -> str | None:
    return git_value(REPO_ROOT, *args)


def _sha256_source_tree(
    root: Path, suffixes: tuple[str, ...] = (".json", ".py", ".toml")
) -> str | None:
    """Hash source files by relative path and contents without exposing paths."""

    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for source in sorted(root.rglob("*")):
        if not source.is_file() or source.suffix not in suffixes:
            continue
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with source.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_dependency_immutability(provenance: dict[str, Any], *, phase: str) -> None:
    """Reject source-control changes while a benchmark matrix is running."""

    expected = provenance["execution_dependencies"]["source_control"]
    actual = {
        "flashinfer_git_head": _git_value("rev-parse", "HEAD"),
        "flashinfer_tracked_worktree_dirty": bool(
            _git_value("status", "--porcelain", "--untracked-files=no")
        ),
    }
    if actual != expected:
        raise RuntimeError(
            f"source-control drift detected {phase}: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def _verify_completion_source_hashes(provenance: dict[str, Any]) -> None:
    """Rehash public source inputs once after the complete matrix."""

    expected = provenance["execution_dependencies"]["source_hashes"]
    actual = _public_source_hashes()
    if actual != expected:
        raise RuntimeError(
            "execution source hash drift detected at completion: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def _public_source_hashes() -> dict[str, str | None]:
    return {
        "benchmark": _sha256_file(Path(__file__)),
        "timing_helper": _sha256_file(
            REPO_ROOT / "benchmarks/routines/attention_ts_benchmark/timing.py"
        ),
        "attention_ts": _sha256_source_tree(
            REPO_ROOT / "flashinfer/attention/prims_ts"
        ),
        "flashinfer_mla": _sha256_source_tree(REPO_ROOT / "flashinfer/mla"),
        "flashinfer_jit": _sha256_source_tree(REPO_ROOT / "flashinfer/jit"),
    }


def _validate_trt_artifact_manifest(
    manifest: Path,
    expected_sha256: str,
    checksum_disabled: str | None,
) -> str:
    """Require authenticated TRT metadata before signing or executing a run."""

    if checksum_disabled:
        raise RuntimeError(
            "FLASHINFER_CUBIN_CHECKSUM_DISABLED is set. Attention-TS MLA "
            "comparisons require artifact checksum verification and refuse to run."
        )
    if not manifest.is_file():
        raise FileNotFoundError(
            "TRTLLM-gen artifact manifest is missing: "
            f"{manifest}. Install the matching flashinfer-cubin package before "
            "starting the benchmark."
        )
    actual_sha256 = _sha256_file(manifest)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "TRTLLM-gen artifact manifest checksum mismatch: "
            f"expected {expected_sha256}, got {actual_sha256} at {manifest}"
        )
    return actual_sha256


def _execution_semantics(
    prims_ts_interface: str = "wrapper",
    seq_len_q: int = _Q_LEN,
    reference_backend: str = _TRTLLM_BACKEND,
) -> dict[str, Any]:
    """Machine-readable public comparison contract shared by rows/signatures."""

    if prims_ts_interface not in _PRIMS_TS_INTERFACES:
        raise ValueError(f"unsupported PrimTS interface {prims_ts_interface!r}")
    if seq_len_q <= 0:
        raise ValueError("seq_len_q must be positive")
    semantics = {
        "query_length": seq_len_q,
        "page_size": _PAGE_SIZE,
        "qk_nope_head_dim": _QK_NOPE_HEAD_DIM,
        "kv_lora_rank": _KV_LORA_RANK,
        "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
        "input_dtypes": list(_DTYPES),
        "output_dtype": "bf16",
        "mask_type": "causal",
        "query_layout": "B,SQ,H,D",
        "output_layout": "B,SQ,H,D",
        "bottom_right_causal_reference": True,
        "sq1_causal_dense_equivalence": seq_len_q == 1,
        "public_scales": ["bmm1_scale", "bmm2_scale"],
        "scale_semantics": (
            "FP8 Q/KV storage scales are fused into bmm1_scale/bmm2_scale "
            "before both public calls"
        ),
        "reference_tolerances": {
            name: {"rtol": values[0], "atol": values[1], "relative_l2": values[2]}
            for name, values in _REFERENCE_TOLERANCES.items()
        },
        "shared_input_objects": [
            "query",
            "kv_cache",
            "block_tables",
            "seq_lens",
        ],
        "separate_stable_output_buffers": True,
        "separate_backend_workspaces": True,
        "prims_ts_interface": prims_ts_interface,
        "prims_ts_public_api": (
            "flashinfer.mla.prims_ts_batch_decode_with_kv_cache_mla"
            if prims_ts_interface == "standalone"
            else "flashinfer.attention.prims_ts.BatchMLADecodePagedTSWrapper"
        ),
        "prims_ts_standalone_explicit_seq_lens": (prims_ts_interface == "standalone"),
        "prims_ts_standalone_queried_caller_workspace": (
            prims_ts_interface == "standalone"
        ),
        "attention_ts_policy": "automatic-family-profile-split-persistence",
        "reference_backend": reference_backend,
        "reference_public_api": (
            "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla"
        ),
        "reference_workspace_bytes": _TRTLLM_WORKSPACE_BYTES,
    }
    if reference_backend == _CUTE_DSL_BACKEND:
        semantics.update(
            {
                "cute_dsl_impl": "monolithic",
                "cute_dsl_python_cross_backend_bucket_autotune": False,
            }
        )
        return semantics
    if reference_backend != _TRTLLM_BACKEND:
        raise ValueError(f"unsupported reference backend {reference_backend!r}")
    semantics.update(
        {
            "trtllm_backend": _TRTLLM_BACKEND,
            "trtllm_enable_pdl": _TRTLLM_ENABLE_PDL,
            "trtllm_is_var_seq": _TRTLLM_IS_VAR_SEQ,
            "trtllm_uses_shared_paged_kv_idx": _TRTLLM_USES_SHARED_PAGED_KV_IDX,
            "trtllm_sparse_mla_top_k": _TRTLLM_SPARSE_MLA_TOP_K,
            "trtllm_sinks": None,
            "trtllm_skip_softmax_threshold_scale_factor": None,
            "trtllm_return_lse": False,
            "trtllm_cum_seq_lens_q": None,
            "trtllm_python_cross_backend_bucket_autotune": _TRTLLM_PYTHON_AUTOTUNE,
            "trtllm_tactic": -1,
            "trtllm_internal_shape_auto_selector": (
                _TRTLLM_INTERNAL_SHAPE_AUTO_SELECTOR
            ),
            "trtllm_workspace_bytes": _TRTLLM_WORKSPACE_BYTES,
            "trtllm_counter_buffer": "separate shape-sized uint8 storage",
            "trtllm_counter_buffer_size": (
                "4 * round_up(max(batch_size * num_qo_heads, sm_count), 8) bytes"
            ),
            "trtllm_workspace_zeroed_once_before_first_use": True,
            "variable_seq_lens": (
                "deterministic max-first, approximately half-to-full, "
                "non-page-aligned tails for B>1"
            ),
        }
    )
    return semantics


def _case_contract(
    spec: MLAPerformanceCaseSpec,
    source_session_sha256: str,
    prims_ts_interface: str = "wrapper",
    reference_backend: str = _TRTLLM_BACKEND,
) -> dict[str, Any]:
    """Canonical immutable identity for one selected matrix row."""

    return {
        "case_id": spec.case_id,
        "batch_size": spec.batch_size,
        "num_qo_heads": spec.num_heads,
        "max_seq_len": spec.max_seq_len,
        "input_dtype": spec.dtype_name,
        "fixture_seed": spec.seed,
        "page_size": _PAGE_SIZE,
        "q_len": spec.seq_len_q,
        "qk_nope_head_dim": _QK_NOPE_HEAD_DIM,
        "kv_lora_rank": _KV_LORA_RANK,
        "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
        "output_dtype": "bf16",
        "compile_reuse_group": spec.compile_reuse_group,
        "semantic_settings": _execution_semantics(
            prims_ts_interface,
            spec.seq_len_q,
            reference_backend,
        ),
        "source_session_sha256": source_session_sha256,
    }


def _case_contract_record(
    spec: MLAPerformanceCaseSpec,
    source_session_sha256: str,
    prims_ts_interface: str = "wrapper",
    reference_backend: str = _TRTLLM_BACKEND,
) -> dict[str, Any]:
    contract = _case_contract(
        spec,
        source_session_sha256,
        prims_ts_interface,
        reference_backend,
    )
    return {"contract": contract, "sha256": _canonical_sha256(contract)}


def _validate_row_case_contract(
    row: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    case_id = expected["contract"]["case_id"]
    if row.get("case_contract") != expected["contract"]:
        raise ValueError(f"resume row {case_id} canonical case contract differs")
    if row.get("case_contract_sha256") != expected["sha256"]:
        raise ValueError(f"resume row {case_id} case contract digest differs")
    if _canonical_sha256(row["case_contract"]) != row["case_contract_sha256"]:
        raise ValueError(f"resume row {case_id} case contract digest is corrupt")

    contract = expected["contract"]
    shape = row.get("shape", {})
    expected_shape_metadata = {
        "batch_size": contract["batch_size"],
        "q_len": contract["q_len"],
        "num_qo_heads": contract["num_qo_heads"],
        "qk_nope_head_dim": contract["qk_nope_head_dim"],
        "kv_lora_rank": contract["kv_lora_rank"],
        "qk_rope_head_dim": contract["qk_rope_head_dim"],
        "page_size": contract["page_size"],
        "max_seq_len": contract["max_seq_len"],
        "input_dtype": contract["input_dtype"],
        "output_dtype": contract["output_dtype"],
        "fixture_seed": contract["fixture_seed"],
    }
    for name, value in expected_shape_metadata.items():
        if shape.get(name) != value:
            raise ValueError(f"resume row {case_id} shape field {name} differs")
    if row.get("case_id") != case_id:
        raise ValueError(f"resume row key {row.get('case_id')} differs from {case_id}")
    expected_interface = contract["semantic_settings"]["prims_ts_interface"]
    if row.get("attention_ts", {}).get("interface") != expected_interface:
        raise ValueError(f"resume row {case_id} PrimTS interface metadata differs")
    reference_backend = contract["semantic_settings"]["reference_backend"]
    reference_label = _reference_label(reference_backend)
    reference = row.get(reference_label, {})
    if reference.get("backend") != reference_backend:
        raise ValueError(f"resume row {case_id} reference backend differs")
    if reference_backend == _TRTLLM_BACKEND:
        if reference.get("enable_pdl") is not _TRTLLM_ENABLE_PDL:
            raise ValueError(f"resume row {case_id} TRTLLM-gen PDL metadata differs")
        if (
            reference.get("internal_shape_auto_selector")
            is not _TRTLLM_INTERNAL_SHAPE_AUTO_SELECTOR
        ):
            raise ValueError(
                f"resume row {case_id} TRTLLM-gen auto-selector metadata differs"
            )
    if row.get("status") == "ok":
        paired = row.get("paired_two_order_cycles")
        required_paired_keys = {
            "method",
            "gate_formula",
            "sample_count_per_backend",
            "cycle_count",
            "gap_percent_total_duration_ratio",
            "backend_total_duration_ms",
            "backend_arithmetic_mean_us",
            "position_summaries",
            "cycle_gap_percent_median",
            "cycle_gap_percent_p95",
            "cycle_gap_percent_min",
            "cycle_gap_percent_max",
            "raw_samples_ms",
        }
        if not isinstance(paired, dict) or not required_paired_keys <= paired.keys():
            raise ValueError(
                f"resume row {case_id} lacks complete paired aggregate metrics"
            )
        if (
            paired["method"] != "alternating-order-balanced-total-duration-ratio"
            or paired["gate_formula"]
            != f"(sum(attention_ts_ms) / sum({reference_label}_ms) - 1) * 100"
        ):
            raise ValueError(f"resume row {case_id} has a different paired estimator")
        if _balanced_gap_percent(row) is None:
            raise ValueError(
                f"resume row {case_id} has an invalid balanced gate metric"
            )
        if (
            paired["sample_count_per_backend"] < 2
            or paired["sample_count_per_backend"] % 2
            or paired["cycle_count"] * 2 != paired["sample_count_per_backend"]
        ):
            raise ValueError(
                f"resume row {case_id} has an invalid two-order cycle count"
            )
        backend_names = {"attention_ts", reference_label}
        for field in (
            "backend_total_duration_ms",
            "backend_arithmetic_mean_us",
            "position_summaries",
            "raw_samples_ms",
        ):
            value = paired[field]
            if not isinstance(value, dict) or set(value) != backend_names:
                raise ValueError(
                    f"resume row {case_id} has invalid paired {field} metadata"
                )
        sample_count = paired["sample_count_per_backend"]
        backend_totals: dict[str, float] = {}
        for backend in backend_names:
            samples = paired["raw_samples_ms"][backend]
            positions = paired["position_summaries"][backend]
            if (
                not isinstance(samples, list)
                or len(samples) != sample_count
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in samples
                )
            ):
                raise ValueError(
                    f"resume row {case_id} has invalid {backend} raw samples"
                )
            if not isinstance(positions, dict) or set(positions) != {
                "first",
                "second",
            }:
                raise ValueError(
                    f"resume row {case_id} has invalid {backend} position summaries"
                )
            if any(
                not isinstance(positions[position], dict)
                or positions[position].get("sample_count") != sample_count // 2
                for position in ("first", "second")
            ):
                raise ValueError(
                    f"resume row {case_id} has incomplete {backend} position samples"
                )
            total_ms = sum(float(value) for value in samples)
            backend_totals[backend] = total_ms
            recorded_total = paired["backend_total_duration_ms"][backend]
            recorded_mean_us = paired["backend_arithmetic_mean_us"][backend]
            if not math.isclose(
                float(recorded_total), total_ms, rel_tol=1e-12, abs_tol=1e-12
            ) or not math.isclose(
                float(recorded_mean_us),
                total_ms / sample_count * 1000.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"resume row {case_id} has inconsistent {backend} aggregates"
                )
        reference_total_ms = backend_totals[reference_label]
        if reference_total_ms == 0.0:
            raise ValueError(f"resume row {case_id} has zero reference duration")
        expected_gap_percent = (
            backend_totals["attention_ts"] / reference_total_ms - 1.0
        ) * 100.0
        if not math.isclose(
            float(paired["gap_percent_total_duration_ratio"]),
            expected_gap_percent,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"resume row {case_id} has an inconsistent paired gate metric"
            )
        disposition = row.get("performance_disposition")
        required_disposition_keys = {
            "threshold_percent",
            "raw_gap_percent",
            "balanced_gap_percent",
            "raw_exceeds_threshold",
            "balanced_exceeds_threshold",
            "classification",
            "gate_metric",
        }
        if (
            not isinstance(disposition, dict)
            or not required_disposition_keys <= disposition.keys()
            or disposition["gate_metric"] != _PAIRED_GATE_METRIC
            or not math.isclose(
                float(disposition["balanced_gap_percent"]),
                expected_gap_percent,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"resume row {case_id} lacks a valid performance disposition"
            )


def _timing_contract(args) -> dict[str, Any]:
    reference_label = _reference_label(args.reference_backend)
    contract = {
        "method": f"paired-alternating-{args.timing_cache_mode}-cuda-graphs",
        "cache_mode": args.timing_cache_mode,
        "separate_backend_graphs": True,
        "public_calls_per_graph": 1,
        "alternating_replay_order": True,
        "complete_opposite_order_cycles": True,
        "samples_per_backend": args.iters,
        "two_order_cycle_count": args.iters // 2,
        "regression_gate_metric": _PAIRED_GATE_METRIC,
        "regression_gate_formula": (
            f"(sum(attention_ts_ms) / sum({reference_label}_ms) - 1) * 100"
        ),
        "reference_backend": args.reference_backend,
        "cycle_ratio_percentiles_are_diagnostic_only": True,
        "raw_samples_recorded": True,
        "backend_arithmetic_means_recorded": True,
        "first_second_position_summaries_recorded": True,
        "warmup_replays": args.warmup_iters,
        "sample_count": args.iters,
        "output_preallocated": True,
        "planning_and_first_calls_excluded": True,
        "prims_ts_interface": args.prims_ts_interface,
    }
    if args.timing_cache_mode == "cold-l2":
        contract.update(
            {
                "cold_l2_strategy": (
                    "external-run-scoped-2x-l2-seeded-random-int8-add-before-event-v1"
                ),
                "l2_flush_multiplier": 2,
                "l2_flush_timed": False,
                "cold_l2_scrub": args.cold_l2_scrub_contract,
            }
        )
    return contract


def _oracle_precision_contract(torch) -> dict[str, Any]:
    return {
        "input_accumulation_dtype": "float32",
        "cuda_matmul_fp32_precision_api": ("torch.backends.cuda.matmul.fp32_precision"),
        "cuda_matmul_fp32_precision_forced": (_ORACLE_CUDA_MATMUL_FP32_PRECISION),
        "tf32_enabled_during_reference": False,
        "entry_cuda_matmul_fp32_precision": (torch.backends.cuda.matmul.fp32_precision),
        "state_restored_after_reference": True,
    }


@contextmanager
def _forced_oracle_precision(torch):
    """Force IEEE FP32 oracle matmuls locally, then restore caller state."""

    previous_precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = _ORACLE_CUDA_MATMUL_FP32_PRECISION
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous_precision


def _backend_neutral_reference(case, torch):
    """Evaluate ideal MLA math in FP32 from the shared stored tensors/scales."""

    from benchmarks.routines.attention_ts_benchmark.mla_fixtures import (
        MLA_LATENT_DIM,
        MLA_PAGE_SIZE,
        MLA_QK_DIM,
    )

    with _forced_oracle_precision(torch):
        outputs = []
        seq_lens = case.seq_lens.tolist()
        for batch_idx, seq_len in enumerate(seq_lens):
            page_count = (int(seq_len) + MLA_PAGE_SIZE - 1) // MLA_PAGE_SIZE
            page_ids = case.block_tables[batch_idx, :page_count].to(torch.long)
            cache = (
                case.kv_cache[page_ids, 0]
                .reshape(-1, MLA_QK_DIM)[:seq_len]
                .to(torch.float32)
            )
            request_outputs = []
            for query_idx in range(case.query.shape[1]):
                visible_kv_len = int(seq_len) - (case.query.shape[1] - 1 - query_idx)
                if visible_kv_len <= 0:
                    raise ValueError(
                        "bottom-right causal attention requires every sequence "
                        "length to be at least q_len"
                    )
                query = case.query[batch_idx, query_idx].to(torch.float32)
                visible_cache = cache[:visible_kv_len]
                q_latent = query[:, :MLA_LATENT_DIM]
                q_rope = query[:, MLA_LATENT_DIM:]
                c_latent = visible_cache[:, :MLA_LATENT_DIM]
                c_rope = visible_cache[:, MLA_LATENT_DIM:]
                scores = q_latent @ c_latent.T + q_rope @ c_rope.T
                probabilities = torch.softmax(scores * float(case.bmm1_scale), dim=-1)
                request_outputs.append(
                    probabilities @ c_latent * float(case.bmm2_scale)
                )
            outputs.append(torch.stack(request_outputs, dim=0))
        return torch.stack(outputs, dim=0)


def _reference_tolerances(dtype_name: str) -> tuple[float, float, float]:
    # The FP8 envelope accounts for the kernels' E4M3 probability paths while
    # the shared oracle evaluates ideal IEEE FP32 softmax. Relative-L2 guards
    # both dtype paths against an all-zero result satisfying elementwise atol.
    return _REFERENCE_TOLERANCES[dtype_name]


def _check_result(name: str, result, reference, spec, torch) -> dict[str, float]:
    expected_shape = (
        spec.batch_size,
        spec.seq_len_q,
        spec.num_heads,
        _KV_LORA_RANK,
    )
    if tuple(result.shape) != expected_shape:
        raise AssertionError(
            f"{name} returned shape {tuple(result.shape)}, expected {expected_shape}"
        )
    if result.dtype != torch.bfloat16:
        raise AssertionError(f"{name} returned {result.dtype}, expected torch.bfloat16")
    actual = result.to(torch.float32)
    if not bool(torch.isfinite(actual).all().item()):
        raise AssertionError(f"{name} produced non-finite output for {spec.case_id}")
    rtol, atol, relative_l2_limit = _reference_tolerances(spec.dtype_name)
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    error = actual - reference
    relative_l2 = float(
        (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)).item()
    )
    if relative_l2 > relative_l2_limit:
        raise AssertionError(
            f"{name} relative L2 {relative_l2:.6g} exceeds "
            f"{relative_l2_limit:.6g} for {spec.case_id}"
        )
    absolute = error.abs()
    return {
        "max_abs_error": float(absolute.max().item()),
        "mean_abs_error": float(absolute.mean().item()),
        "relative_l2_error": relative_l2,
    }


def _counter_nonzero(counter_buffer, torch) -> int:
    return int(torch.count_nonzero(counter_buffer).item())


def _validate_case(case, spec, torch) -> None:
    from benchmarks.routines.attention_ts_benchmark.mla_fixtures import MLA_QK_DIM

    if tuple(case.query.shape) != (
        spec.batch_size,
        spec.seq_len_q,
        spec.num_heads,
        MLA_QK_DIM,
    ):
        raise ValueError("fixture query shape does not match the matrix row")
    if case.kv_cache.ndim != 4 or case.kv_cache.shape[1] != 1:
        raise ValueError("fixture KV cache must use [pages,1,page,576]")
    if case.kv_cache.shape[-2:] != (_PAGE_SIZE, MLA_QK_DIM):
        raise ValueError("fixture KV cache page geometry is not page32/576")
    for name, tensor, ndim in (
        ("block_tables", case.block_tables, 2),
        ("seq_lens", case.seq_lens, 1),
    ):
        if tensor.dtype != torch.int32 or tensor.ndim != ndim:
            raise TypeError(f"{name} must be rank-{ndim} int32")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA storage")
    if int(case.seq_lens.max().item()) != spec.max_seq_len:
        raise ValueError("fixture must contain its advertised maximum KV length")
    if not bool((case.seq_lens >= spec.seq_len_q).all().item()):
        raise ValueError("bottom-right causal rows require every KV length >= q_len")
    if spec.batch_size > 1:
        if torch.unique(case.seq_lens).numel() == 1:
            raise ValueError("B>1 rows must use variable runtime sequence lengths")
        if not bool((case.seq_lens[1:] % _PAGE_SIZE != 0).all().item()):
            raise ValueError("non-leading runtime lengths must exercise page tails")


def _prepare_ts_call(case, spec, args, runtime, *, qkv_dtype, out) -> dict[str, Any]:
    """Own the selected PrimTS interface, setup resources, and hot callable."""

    torch = runtime["torch"]
    cache_before_setup = runtime["cache_info"]()
    torch.cuda.synchronize(case.query.device)
    setup_start = time.perf_counter()

    if args.prims_ts_interface == "wrapper":
        wrapper = runtime["wrapper_type"]()
        wrapper.plan(
            case.block_tables,
            case.seq_lens,
            spec.num_heads,
            _KV_LORA_RANK,
            _QK_ROPE_HEAD_DIM,
            _PAGE_SIZE,
            q_data_type=qkv_dtype,
            kv_data_type=qkv_dtype,
            o_data_type=torch.bfloat16,
            mask_type="causal",
            max_kv_len=spec.max_seq_len,
            seq_len_q=spec.seq_len_q,
        )
        policy = dict(wrapper._policy)

        def run_ts():
            return wrapper.run(
                case.query,
                case.kv_cache,
                bmm1_scale=case.bmm1_scale,
                bmm2_scale=case.bmm2_scale,
                out=out,
            )

        setup_kind = "plan"
        workspace_bytes = None
        public_api = "flashinfer.attention.prims_ts.BatchMLADecodePagedTSWrapper"
    else:
        workspace_bytes = runtime["get_ts_workspace_size"](
            spec.batch_size,
            spec.num_heads,
            _KV_LORA_RANK,
            _QK_ROPE_HEAD_DIM,
            _PAGE_SIZE,
            spec.max_seq_len,
            q_dtype=qkv_dtype,
            kv_dtype=qkv_dtype,
            out_dtype=torch.bfloat16,
            mask_type="causal",
            device=args.device,
            seq_len_q=spec.seq_len_q,
        )
        ts_workspace = torch.empty(
            workspace_bytes, dtype=torch.int8, device=case.query.device
        )
        policy = {
            "source": "auto",
            "diagnostic_available": False,
            "diagnostic_reason": (
                "the standalone public API does not expose its resolved recipe"
            ),
        }

        def run_ts():
            return runtime["standalone_ts_decode"](
                case.query,
                case.kv_cache,
                ts_workspace,
                _KV_LORA_RANK,
                _QK_ROPE_HEAD_DIM,
                case.block_tables,
                case.seq_lens,
                spec.max_seq_len,
                out=out,
                bmm1_scale=case.bmm1_scale,
                bmm2_scale=case.bmm2_scale,
                mask_type="causal",
                out_dtype=torch.bfloat16,
            )

        setup_kind = "workspace-query-and-allocation"
        public_api = "flashinfer.mla.prims_ts_batch_decode_with_kv_cache_mla"

    torch.cuda.synchronize(case.query.device)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    cache_after_setup = runtime["cache_info"]()
    return {
        "call": run_ts,
        "interface": args.prims_ts_interface,
        "public_api": public_api,
        "setup_kind": setup_kind,
        "setup_ms": setup_ms,
        "workspace_bytes": workspace_bytes,
        "policy": policy,
        "cache_before_setup": cache_before_setup,
        "cache_after_setup": cache_after_setup,
    }


def _run_case(spec, args, runtime, cold_l2_scrubber) -> dict[str, Any]:
    torch = runtime["torch"]
    make_case = runtime["make_case"]
    reference_decode = runtime["reference_decode"]
    cache_info = runtime["cache_info"]
    flashinfer = runtime["flashinfer"]
    reference_backend = args.reference_backend
    reference_label = _reference_label(reference_backend)

    qkv_dtype = _dtype_from_name(spec.dtype_name, torch)
    case = make_case(
        batch_size=spec.batch_size,
        num_qo_heads=spec.num_heads,
        max_seq_len=spec.max_seq_len,
        qkv_dtype=qkv_dtype,
        seq_len_q=spec.seq_len_q,
        device=f"cuda:{args.device}",
        seed=spec.seed,
    )
    _validate_case(case, spec, torch)
    reference = _backend_neutral_reference(case, torch)

    ts_out = torch.empty(
        (spec.batch_size, spec.seq_len_q, spec.num_heads, _KV_LORA_RANK),
        dtype=torch.bfloat16,
        device=case.query.device,
    )
    reference_out = torch.empty_like(ts_out)
    reference_workspace = torch.empty(
        _TRTLLM_WORKSPACE_BYTES, dtype=torch.int8, device=case.query.device
    )
    if reference_backend == _TRTLLM_BACKEND:
        reference_counter_bytes = runtime["get_trtllm_counter_bytes"](
            spec.batch_size,
            spec.num_heads,
            runtime["get_device_sm_count"](case.query.device),
        )
        reference_counter_buffer = torch.zeros(
            reference_counter_bytes, dtype=torch.uint8, device=case.query.device
        )
    else:
        reference_counter_bytes = 0
        reference_counter_buffer = None

    prepared_ts = _prepare_ts_call(
        case,
        spec,
        args,
        runtime,
        qkv_dtype=qkv_dtype,
        out=ts_out,
    )
    run_ts = prepared_ts["call"]

    def run_reference():
        try:
            return reference_decode(
                query=case.query,
                kv_cache=case.kv_cache,
                workspace_buffer=reference_workspace,
                qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
                kv_lora_rank=_KV_LORA_RANK,
                qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
                block_tables=case.block_tables,
                seq_lens=case.seq_lens,
                max_seq_len=spec.max_seq_len,
                sparse_mla_top_k=_TRTLLM_SPARSE_MLA_TOP_K,
                out=reference_out,
                bmm1_scale=case.bmm1_scale,
                bmm2_scale=case.bmm2_scale,
                sinks=None,
                skip_softmax_threshold_scale_factor=None,
                enable_pdl=_TRTLLM_ENABLE_PDL,
                backend=reference_backend,
                cute_dsl_impl="monolithic",
                is_var_seq=_TRTLLM_IS_VAR_SEQ,
                uses_shared_paged_kv_idx=_TRTLLM_USES_SHARED_PAGED_KV_IDX,
                lse=None,
                return_lse=False,
                cum_seq_lens_q=None,
                max_q_len=None,
                multi_ctas_kv_counter_buffer=reference_counter_buffer,
            )
        except (ImportError, OSError, RuntimeError) as error:
            diagnostic = _cuda_runtime_error(error)
            if diagnostic is error:
                raise
            raise diagnostic from error

    # Keep both explicit public backends out of FlashInfer's Python
    # cross-backend profiler. PrimTS retains its independent automatic policy;
    # TRTLLM-gen retains its internal shape selector when it is the reference.
    with flashinfer.autotune(_TRTLLM_PYTHON_AUTOTUNE):
        ts_result, ts_first_call_ms = _first_call(run_ts, torch)
        cache_after_first_call = cache_info()
        reference_result, reference_first_call_ms = _first_call(run_reference, torch)
        ts_error = _check_result("Attention-TS", ts_result, reference, spec, torch)
        reference_error = _check_result(
            reference_backend,
            reference_result,
            reference,
            spec,
            torch,
        )
        counter_nonzero_after_first = (
            0
            if reference_counter_buffer is None
            else _counter_nonzero(reference_counter_buffer, torch)
        )
        if counter_nonzero_after_first:
            raise AssertionError(
                "TRTLLM-gen did not reset its workspace counter region after first call"
            )

        if args.timing_cache_mode == "cold-l2":
            ts_timing, reference_timing, paired_two_order_cycles = (
                _time_paired_cold_l2_cuda_graphs(
                    run_ts,
                    run_reference,
                    torch=torch,
                    device=args.device,
                    batch_size=spec.batch_size,
                    warmup_replays=args.warmup_iters,
                    sample_count=args.iters,
                    scrubber=cold_l2_scrubber,
                    reference_label=reference_label,
                )
            )
        else:
            ts_timing, reference_timing, paired_two_order_cycles = (
                _time_paired_cuda_graphs(
                    run_ts,
                    run_reference,
                    torch=torch,
                    batch_size=spec.batch_size,
                    warmup_replays=args.warmup_iters,
                    sample_count=args.iters,
                    calls_per_graph=1,
                    reference_label=reference_label,
                )
            )

    ts_post = _check_result("Attention-TS after timing", ts_out, reference, spec, torch)
    reference_post = _check_result(
        f"{reference_backend} after timing",
        reference_out,
        reference,
        spec,
        torch,
    )
    counter_nonzero_after_timing = (
        0
        if reference_counter_buffer is None
        else _counter_nonzero(reference_counter_buffer, torch)
    )
    if counter_nonzero_after_timing:
        raise AssertionError(
            "TRTLLM-gen did not reset its workspace counter region after graph replay"
        )

    backend_difference = (ts_out.float() - reference_out.float()).abs()
    gap_us = ts_timing["median_us"] - reference_timing["median_us"]
    gap_percent = gap_us / reference_timing["median_us"] * 100.0
    balanced_gap_percent = paired_two_order_cycles["gap_percent_total_duration_ratio"]
    performance_disposition = _performance_disposition(
        gap_percent,
        balanced_gap_percent,
        args.gap_threshold,
    )
    seq_lens = [int(value) for value in case.seq_lens.tolist()]
    attention_ts_record = {
        "interface": prepared_ts["interface"],
        "public_api": prepared_ts["public_api"],
        "setup_kind": prepared_ts["setup_kind"],
        "setup_ms": prepared_ts["setup_ms"],
        # Compatibility field retained for existing CSV consumers.
        "plan_ms": prepared_ts["setup_ms"],
        "workspace_bytes": prepared_ts["workspace_bytes"],
        "first_call_ms": ts_first_call_ms,
        "compiled_during_plan": (
            prepared_ts["interface"] == "wrapper"
            and prepared_ts["cache_after_setup"].misses
            > prepared_ts["cache_before_setup"].misses
        ),
        "compiled_during_setup": (
            prepared_ts["cache_after_setup"].misses
            > prepared_ts["cache_before_setup"].misses
        ),
        "compiled_on_first_call": (
            cache_after_first_call.misses > prepared_ts["cache_after_setup"].misses
        ),
        "policy": prepared_ts["policy"],
        "cache_before_plan": _cache_info_dict(prepared_ts["cache_before_setup"]),
        "cache_after_plan": _cache_info_dict(prepared_ts["cache_after_setup"]),
        "cache_after_first_call": _cache_info_dict(cache_after_first_call),
        **ts_timing,
        **ts_error,
        "post_timing_max_abs_error": ts_post["max_abs_error"],
        "post_timing_mean_abs_error": ts_post["mean_abs_error"],
        "post_timing_relative_l2_error": ts_post["relative_l2_error"],
    }
    reference_record = {
        "first_call_ms": reference_first_call_ms,
        "backend": reference_backend,
        "workspace_bytes": _TRTLLM_WORKSPACE_BYTES,
        "counter_buffer_bytes": reference_counter_bytes,
        "counter_nonzero_after_first": counter_nonzero_after_first,
        "counter_nonzero_after_timing": counter_nonzero_after_timing,
        **reference_timing,
        **reference_error,
        "post_timing_max_abs_error": reference_post["max_abs_error"],
        "post_timing_mean_abs_error": reference_post["mean_abs_error"],
        "post_timing_relative_l2_error": reference_post["relative_l2_error"],
    }
    if reference_backend == _TRTLLM_BACKEND:
        reference_record.update(
            {
                "enable_pdl": _TRTLLM_ENABLE_PDL,
                "is_var_seq": _TRTLLM_IS_VAR_SEQ,
                "uses_shared_paged_kv_idx": _TRTLLM_USES_SHARED_PAGED_KV_IDX,
                "python_cross_backend_bucket_autotune": _TRTLLM_PYTHON_AUTOTUNE,
                "tactic": -1,
                "internal_shape_auto_selector": (_TRTLLM_INTERNAL_SHAPE_AUTO_SELECTOR),
            }
        )
    else:
        reference_record.update(
            {
                "cute_dsl_impl": "monolithic",
                "python_cross_backend_bucket_autotune": False,
            }
        )

    comparison_contract = {
        "execution_apis_public": True,
        "implementation_state_used_only_for_diagnostics": True,
        "selected_prims_ts_interface": prepared_ts["interface"],
        "standalone_uses_explicit_seq_lens": (prepared_ts["interface"] == "standalone"),
        "standalone_uses_queried_caller_workspace": (
            prepared_ts["interface"] == "standalone"
        ),
        "reference_backend": reference_backend,
        "single_fixture_instance": True,
        "same_query_object": True,
        "same_query_layout": True,
        "query_layout": "B,SQ,H,D",
        "same_kv_cache_object": True,
        "same_block_tables_object": True,
        "same_seq_lens_object": True,
        "same_bmm1_scale": True,
        "same_bmm2_scale": True,
        "same_backend_neutral_fp32_reference": True,
        "stable_separate_output_buffers": True,
        "same_output_layout": True,
        "output_layout": "B,SQ,H,D",
        "separate_backend_workspaces": True,
        "bottom_right_causal_reference": True,
        "sq1_causal_dense_equivalence": spec.seq_len_q == 1,
        "attention_ts_auto_policy": prepared_ts["policy"].get("source") == "auto",
    }
    if reference_backend == _TRTLLM_BACKEND:
        comparison_contract.update(
            {
                "trtllm_backend_forced": _TRTLLM_BACKEND,
                "trtllm_enable_pdl": _TRTLLM_ENABLE_PDL,
                "trtllm_python_cross_backend_bucket_autotune": (
                    _TRTLLM_PYTHON_AUTOTUNE
                ),
                "trtllm_internal_shape_auto_selector": (
                    _TRTLLM_INTERNAL_SHAPE_AUTO_SELECTOR
                ),
            }
        )
    else:
        comparison_contract["cute_dsl_impl"] = "monolithic"

    result = {
        "status": "ok",
        "case_id": spec.case_id,
        "reference_backend": reference_backend,
        "shape": {
            "batch_size": spec.batch_size,
            "q_len": spec.seq_len_q,
            "num_qo_heads": spec.num_heads,
            "qk_nope_head_dim": _QK_NOPE_HEAD_DIM,
            "kv_lora_rank": _KV_LORA_RANK,
            "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
            "page_size": _PAGE_SIZE,
            "max_seq_len": spec.max_seq_len,
            "seq_lens": seq_lens,
            "min_seq_len": min(seq_lens),
            "mean_seq_len": sum(seq_lens) / len(seq_lens),
            "input_dtype": spec.dtype_name,
            "output_dtype": "bf16",
            "fixture_seed": spec.seed,
            "physical_page_capacity": int(case.kv_cache.shape[0]),
        },
        "scales": {
            "q_scale": case.q_scale,
            "kv_scale": case.kv_scale,
            "bmm1_scale": case.bmm1_scale,
            "bmm2_scale": case.bmm2_scale,
        },
        "attention_ts": attention_ts_record,
        reference_label: reference_record,
        "comparison_contract": comparison_contract,
        "reference_over_attention_ts": (
            reference_timing["median_us"] / ts_timing["median_us"]
        ),
        "attention_ts_gap_us": gap_us,
        "attention_ts_gap_percent_vs_reference": gap_percent,
        "paired_two_order_cycles": paired_two_order_cycles,
        "performance_disposition": performance_disposition,
        "backend_max_abs_difference": float(backend_difference.max().item()),
        "backend_mean_abs_difference": float(backend_difference.mean().item()),
    }
    if reference_backend == _TRTLLM_BACKEND:
        result["attention_ts_speedup"] = result["reference_over_attention_ts"]
        result["attention_ts_gap_percent_vs_trtllm_gen"] = gap_percent
    return result


def _error_result(
    spec,
    error: BaseException,
    prims_ts_interface: str = "wrapper",
    gap_threshold: float = 5.0,
    reference_backend: str = _TRTLLM_BACKEND,
) -> dict[str, Any]:
    reference_label = _reference_label(reference_backend)
    reference_record = {"backend": reference_backend}
    if reference_backend == _TRTLLM_BACKEND:
        reference_record.update(
            {
                "enable_pdl": _TRTLLM_ENABLE_PDL,
                "is_var_seq": _TRTLLM_IS_VAR_SEQ,
                "uses_shared_paged_kv_idx": _TRTLLM_USES_SHARED_PAGED_KV_IDX,
                "python_cross_backend_bucket_autotune": _TRTLLM_PYTHON_AUTOTUNE,
                "tactic": -1,
                "internal_shape_auto_selector": (_TRTLLM_INTERNAL_SHAPE_AUTO_SELECTOR),
            }
        )
    else:
        reference_record["cute_dsl_impl"] = "monolithic"
    return {
        "status": "error",
        "case_id": spec.case_id,
        "reference_backend": reference_backend,
        "shape": {
            "batch_size": spec.batch_size,
            "q_len": spec.seq_len_q,
            "num_qo_heads": spec.num_heads,
            "qk_nope_head_dim": _QK_NOPE_HEAD_DIM,
            "kv_lora_rank": _KV_LORA_RANK,
            "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
            "page_size": _PAGE_SIZE,
            "max_seq_len": spec.max_seq_len,
            "input_dtype": spec.dtype_name,
            "output_dtype": "bf16",
            "fixture_seed": spec.seed,
        },
        "attention_ts": {"interface": prims_ts_interface},
        reference_label: reference_record,
        "paired_two_order_cycles": None,
        "performance_disposition": {
            "threshold_percent": gap_threshold,
            "raw_gap_percent": None,
            "balanced_gap_percent": None,
            "raw_exceeds_threshold": None,
            "balanced_exceeds_threshold": None,
            "classification": "error",
            "gate_metric": _PAIRED_GATE_METRIC,
        },
        "reason": f"{type(error).__name__}: {error}",
    }


def _validate_compile_reuse(
    spec: MLAPerformanceCaseSpec,
    result: dict[str, Any],
    completed_groups: set[str],
) -> None:
    """Require a later row in an explicit group to reuse its compiled topology."""

    group = spec.compile_reuse_group
    if group is None or result.get("status") != "ok":
        return
    if group in completed_groups:
        attention_ts = result["attention_ts"]
        if (
            attention_ts["compiled_during_setup"]
            or attention_ts["compiled_on_first_call"]
        ):
            raise AssertionError(
                f"batch-only topology change recompiled group {group} at {spec.case_id}"
            )
    completed_groups.add(group)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark JSON {path} must contain an object")
    return payload


def _reference_record(result: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    backend = result.get("reference_backend", _TRTLLM_BACKEND)
    label = _reference_label(backend)
    return backend, label, result.get(label, {})


def _row_for_csv(result: dict[str, Any], execution_index: int) -> dict[str, Any]:
    shape = result["shape"]
    reference_backend, reference_label, reference = _reference_record(result)
    row: dict[str, Any] = {
        "execution_index": execution_index,
        "case_id": result["case_id"],
        "status": result.get("status", "ok"),
        "input_dtype": shape["input_dtype"],
        "output_dtype": shape["output_dtype"],
        "batch_size": shape["batch_size"],
        "num_qo_heads": shape["num_qo_heads"],
        "max_seq_len": shape["max_seq_len"],
        "min_seq_len": shape.get("min_seq_len", ""),
        "mean_seq_len": shape.get("mean_seq_len", ""),
        "page_size": shape["page_size"],
        "q_len": shape["q_len"],
        "qk_nope_head_dim": shape["qk_nope_head_dim"],
        "kv_lora_rank": shape["kv_lora_rank"],
        "qk_rope_head_dim": shape["qk_rope_head_dim"],
        "fixture_seed": shape["fixture_seed"],
        "source_session_sha256": result.get("source_session_sha256", ""),
        "case_contract_sha256": result.get("case_contract_sha256", ""),
        "prims_ts_interface": result.get("attention_ts", {}).get("interface", ""),
        "reference_backend": reference_backend,
        "trtllm_backend": result.get("trtllm_gen", {}).get("backend", ""),
        "trtllm_enable_pdl": result.get("trtllm_gen", {}).get("enable_pdl", ""),
        "trtllm_python_autotune": result.get("trtllm_gen", {}).get(
            "python_cross_backend_bucket_autotune", ""
        ),
        "trtllm_internal_shape_auto_selector": result.get("trtllm_gen", {}).get(
            "internal_shape_auto_selector", ""
        ),
        "performance_disposition": result.get("performance_disposition", {}).get(
            "classification", "error"
        ),
        "regression_gate_metric": result.get("performance_disposition", {}).get(
            "gate_metric", _PAIRED_GATE_METRIC
        ),
        "reason": result.get("reason", ""),
    }
    if result.get("status") != "ok":
        return row
    ts = result["attention_ts"]
    policy = ts["policy"]
    paired = result.get("paired_two_order_cycles", {})
    disposition = result.get("performance_disposition", {})
    position_summaries = paired.get("position_summaries", {})
    row.update(
        {
            "ts_median_us": ts["median_us"],
            "reference_median_us": reference["median_us"],
            "ts_arithmetic_mean_us": ts["arithmetic_mean_us"],
            "reference_arithmetic_mean_us": reference["arithmetic_mean_us"],
            "ts_p95_us": ts["p95_us"],
            "reference_p95_us": reference["p95_us"],
            "ts_gap_us": result["attention_ts_gap_us"],
            "ts_gap_percent": result["attention_ts_gap_percent_vs_reference"],
            "raw_independent_median_gap_percent": result[
                "attention_ts_gap_percent_vs_reference"
            ],
            "order_balanced_total_duration_gap_percent": paired.get(
                "gap_percent_total_duration_ratio", ""
            ),
            "cycle_gap_median_percent": paired.get("cycle_gap_percent_median", ""),
            "cycle_gap_p95_percent": paired.get("cycle_gap_percent_p95", ""),
            "cycle_gap_min_percent": paired.get("cycle_gap_percent_min", ""),
            "cycle_gap_max_percent": paired.get("cycle_gap_percent_max", ""),
            "paired_aggregate_method": paired.get("method", ""),
            "paired_sample_count_per_backend": paired.get(
                "sample_count_per_backend", ""
            ),
            "paired_two_order_cycle_count": paired.get("cycle_count", ""),
            "ts_first_position_mean_us": position_summaries.get("attention_ts", {})
            .get("first", {})
            .get("arithmetic_mean_us", ""),
            "ts_second_position_mean_us": position_summaries.get("attention_ts", {})
            .get("second", {})
            .get("arithmetic_mean_us", ""),
            "reference_first_position_mean_us": position_summaries.get(
                reference_label, {}
            )
            .get("first", {})
            .get("arithmetic_mean_us", ""),
            "reference_second_position_mean_us": position_summaries.get(
                reference_label, {}
            )
            .get("second", {})
            .get("arithmetic_mean_us", ""),
            "performance_disposition": disposition.get(
                "classification", "missing-paired-metric"
            ),
            "raw_exceeds_threshold": disposition.get("raw_exceeds_threshold", ""),
            "balanced_exceeds_threshold": disposition.get(
                "balanced_exceeds_threshold", ""
            ),
            "regression_gate_metric": disposition.get("gate_metric", ""),
            "reference_over_ts": result["reference_over_attention_ts"],
            "ts_plan_ms": ts["plan_ms"],
            "ts_setup_kind": ts.get("setup_kind", "plan"),
            "ts_workspace_bytes": ts.get("workspace_bytes", ""),
            "ts_first_call_ms": ts["first_call_ms"],
            "reference_first_call_ms": reference["first_call_ms"],
            "ts_max_abs_error": ts["max_abs_error"],
            "ts_relative_l2_error": ts["relative_l2_error"],
            "reference_max_abs_error": reference["max_abs_error"],
            "reference_relative_l2_error": reference["relative_l2_error"],
            "backend_max_abs_difference": result["backend_max_abs_difference"],
            "policy_kernel": policy.get("kernel"),
            "policy_source": policy.get("source"),
            "policy_profile": policy.get("profile"),
            "tile_size_q": policy.get("tile_size_q"),
            "tile_size_kv": policy.get("tile_size_kv"),
            "num_insts_kv": policy.get("num_insts_kv"),
            "split_kv": policy.get("split_kv"),
            "num_ctas_per_head_dim": policy.get("num_ctas_per_head_dim"),
            "head_dim_per_cta_v": policy.get("head_dim_per_cta_v"),
            "use_cluster_reduction": policy.get("use_cluster_reduction"),
            "use_persistent_scheduler": policy.get("use_persistent_scheduler"),
            "use_clc_dynamic_persistent_scheduler": policy.get(
                "use_clc_dynamic_persistent_scheduler"
            ),
            "sample_count": ts["sample_count"],
            "timing_mode": ts["timing_mode"],
        }
    )
    if reference_backend == _TRTLLM_BACKEND:
        row.update(
            {
                "trtllm_gen_median_us": reference["median_us"],
                "trtllm_gen_arithmetic_mean_us": reference["arithmetic_mean_us"],
                "trtllm_gen_p95_us": reference["p95_us"],
                "trtllm_gen_first_position_mean_us": row[
                    "reference_first_position_mean_us"
                ],
                "trtllm_gen_second_position_mean_us": row[
                    "reference_second_position_mean_us"
                ],
                "trt_over_ts": result["reference_over_attention_ts"],
                "trtllm_gen_first_call_ms": reference["first_call_ms"],
                "trtllm_gen_max_abs_error": reference["max_abs_error"],
                "trtllm_gen_relative_l2_error": reference["relative_l2_error"],
            }
        )
    return row


def _sorted_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(result: dict[str, Any]) -> tuple[bool, float]:
        gap = _balanced_gap_percent(result)
        cannot_gate = result.get("status") != "ok" or gap is None
        return cannot_gate, float("inf") if gap is None else gap

    return sorted(
        results,
        key=sort_key,
        reverse=True,
    )


def _render_csv(results: Sequence[dict[str, Any]]) -> str:
    execution_index = {result["case_id"]: idx for idx, result in enumerate(results)}
    rows = [
        _row_for_csv(result, execution_index[result["case_id"]])
        for result in _sorted_results(results)
    ]
    fieldnames = [
        "execution_index",
        "case_id",
        "status",
        "input_dtype",
        "output_dtype",
        "batch_size",
        "num_qo_heads",
        "max_seq_len",
        "min_seq_len",
        "mean_seq_len",
        "page_size",
        "q_len",
        "qk_nope_head_dim",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "fixture_seed",
        "source_session_sha256",
        "case_contract_sha256",
        "prims_ts_interface",
        "reference_backend",
        "trtllm_backend",
        "trtllm_enable_pdl",
        "trtllm_python_autotune",
        "trtllm_internal_shape_auto_selector",
        "ts_median_us",
        "reference_median_us",
        "trtllm_gen_median_us",
        "ts_arithmetic_mean_us",
        "reference_arithmetic_mean_us",
        "trtllm_gen_arithmetic_mean_us",
        "ts_p95_us",
        "reference_p95_us",
        "trtllm_gen_p95_us",
        "ts_gap_us",
        "ts_gap_percent",
        "raw_independent_median_gap_percent",
        "order_balanced_total_duration_gap_percent",
        "cycle_gap_median_percent",
        "cycle_gap_p95_percent",
        "cycle_gap_min_percent",
        "cycle_gap_max_percent",
        "paired_aggregate_method",
        "paired_sample_count_per_backend",
        "paired_two_order_cycle_count",
        "ts_first_position_mean_us",
        "ts_second_position_mean_us",
        "reference_first_position_mean_us",
        "reference_second_position_mean_us",
        "trtllm_gen_first_position_mean_us",
        "trtllm_gen_second_position_mean_us",
        "performance_disposition",
        "raw_exceeds_threshold",
        "balanced_exceeds_threshold",
        "regression_gate_metric",
        "reference_over_ts",
        "trt_over_ts",
        "ts_plan_ms",
        "ts_setup_kind",
        "ts_workspace_bytes",
        "ts_first_call_ms",
        "reference_first_call_ms",
        "trtllm_gen_first_call_ms",
        "ts_max_abs_error",
        "ts_relative_l2_error",
        "reference_max_abs_error",
        "reference_relative_l2_error",
        "trtllm_gen_max_abs_error",
        "trtllm_gen_relative_l2_error",
        "backend_max_abs_difference",
        "policy_kernel",
        "policy_source",
        "policy_profile",
        "tile_size_q",
        "tile_size_kv",
        "num_insts_kv",
        "split_kv",
        "num_ctas_per_head_dim",
        "head_dim_per_cta_v",
        "use_cluster_reduction",
        "use_persistent_scheduler",
        "use_clc_dynamic_persistent_scheduler",
        "sample_count",
        "timing_mode",
        "reason",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _render_markdown(results: Sequence[dict[str, Any]], gap_threshold: float) -> str:
    successful = [result for result in results if result.get("status") == "ok"]
    raw_regressions = [
        result
        for result in successful
        if _raw_gap_percent(result) is not None
        and _raw_gap_percent(result) > gap_threshold
    ]
    balanced_regressions = [
        result
        for result in successful
        if _balanced_gap_percent(result) is not None
        and _balanced_gap_percent(result) > gap_threshold
    ]
    paired_metric_errors = [
        result for result in successful if _balanced_gap_percent(result) is None
    ]
    raw_exceptions = [
        result
        for result in successful
        if _raw_gap_percent(result) is not None
        and _raw_gap_percent(result) > gap_threshold
        and _balanced_gap_percent(result) is not None
        and _balanced_gap_percent(result) <= gap_threshold
    ]
    errors = [result for result in results if result.get("status") != "ok"]
    source_sessions = {
        result["source_session_sha256"]
        for result in results
        if result.get("source_session_sha256")
    }
    if len(source_sessions) > 1:
        raise ValueError("cannot render results from mixed source sessions")
    source_session = next(iter(source_sessions), "not-recorded")
    interfaces = {
        result.get("attention_ts", {}).get("interface", "wrapper") for result in results
    }
    if len(interfaces) > 1:
        raise ValueError("cannot render results from mixed PrimTS interfaces")
    prims_ts_interface = next(iter(interfaces), "wrapper")
    timing_modes = {
        result.get("attention_ts", {}).get("timing_mode")
        for result in successful
        if result.get("attention_ts", {}).get("timing_mode")
    }
    if len(timing_modes) > 1:
        raise ValueError("cannot render results with mixed timing modes")
    timing_mode = next(iter(timing_modes), "not-recorded")
    reference_backends = {
        result.get("reference_backend", _TRTLLM_BACKEND) for result in results
    }
    if len(reference_backends) > 1:
        raise ValueError("cannot render results with mixed reference backends")
    reference_backend = next(iter(reference_backends), _TRTLLM_BACKEND)
    reference_name = _reference_display_name(reference_backend)
    lines = [
        f"# Attention-TS versus {reference_name} MLA decode",
        "",
        (
            f"Rows: {len(results)}; successful: {len(successful)}; errors: "
            f"{len(errors)}; raw independent-median above {gap_threshold:g}%: "
            f"{len(raw_regressions)}; order-balanced total-duration above "
            f"{gap_threshold:g}%: {len(balanced_regressions)}; raw-only "
            f"exceptions: {len(raw_exceptions)}; paired-metric errors: "
            f"{len(paired_metric_errors)}."
        ),
        "",
        f"Source session: `{source_session}`. PrimTS interface: "
        f"`{prims_ts_interface}`. Reference backend: `{reference_backend}`.",
        "",
        f"Timing uses `{timing_mode}` with paired alternating one-call CUDA "
        "graphs. Raw gap compares "
        "the two independent sample medians. Balanced gap is exactly the ratio "
        f"of total TS duration to total {reference_name} duration over complete "
        "opposite-order cycles and is the regression gate. Per-cycle "
        "percentiles are diagnostics only. Rows are sorted from worst balanced "
        "TS gap to best.",
        "",
        f"| Rank | Case | B | Q | Hq | KV range | Dtype | TS us | {reference_name} us | Raw gap % | Balanced total gap % | Disposition | Kernel/profile | TileQ/TileKV | Inst/Split | V CTAs/V width | cluster | Persistent |",
        "|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---|",
    ]
    rank = 0
    for result in _sorted_results(results):
        if result.get("status") != "ok":
            shape = result["shape"]
            lines.append(
                f"| error | `{result['case_id']}` | {shape['batch_size']} | "
                f"{shape['q_len']} | {shape['num_qo_heads']} | - | "
                f"{shape['input_dtype']} | - | - | - | - | error | "
                f"{result.get('reason', '')} | - | - | - | - | - |"
            )
            continue
        rank += 1
        shape = result["shape"]
        ts = result["attention_ts"]
        _, _, reference = _reference_record(result)
        policy = ts["policy"]
        profile = policy.get("profile") or "-"
        balanced_gap = _balanced_gap_percent(result)
        disposition = result.get("performance_disposition", {}).get(
            "classification", "missing-paired-metric"
        )
        lines.append(
            f"| {rank} | `{result['case_id']}` | {shape['batch_size']} | "
            f"{shape['q_len']} | {shape['num_qo_heads']} | "
            f"{shape['min_seq_len']}..{shape['max_seq_len']} | "
            f"{shape['input_dtype']}→bf16 | {ts['median_us']:.3f} | "
            f"{reference['median_us']:.3f} | "
            f"{result['attention_ts_gap_percent_vs_reference']:+.3f} | "
            f"{'-' if balanced_gap is None else f'{balanced_gap:+.3f}'} | "
            f"{disposition} | "
            f"{policy.get('kernel')}/{profile} | "
            f"{policy.get('tile_size_q')}/{policy.get('tile_size_kv')} | "
            f"{policy.get('num_insts_kv')}/{policy.get('split_kv')} | "
            f"{policy.get('num_ctas_per_head_dim')}/"
            f"{policy.get('head_dim_per_cta_v')} | "
            f"{policy.get('use_cluster_reduction')} | "
            f"{policy.get('use_persistent_scheduler')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This table covers the fixed query length shown in Q, bottom-right "
            "causal attention, page32, shared page indices, latent/RoPE 512/64, "
            "runtime KV lengths, BF16 or E4M3 input, and BF16 output on the "
            "recorded device/toolchain only. It does not approve a "
            "kernel or heuristic change; each regression requires a separately "
            "documented diagnosis and review.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary(results: Sequence[dict[str, Any]], gap_threshold: float):
    successful = [result for result in results if result.get("status") == "ok"]
    errors = [result for result in results if result.get("status") != "ok"]
    raw_regressions = [
        result
        for result in successful
        if _raw_gap_percent(result) is not None
        and _raw_gap_percent(result) > gap_threshold
    ]
    balanced_evaluable = [
        result for result in successful if _balanced_gap_percent(result) is not None
    ]
    balanced_regressions = [
        result
        for result in balanced_evaluable
        if _balanced_gap_percent(result) > gap_threshold
    ]
    paired_metric_errors = [
        result for result in successful if _balanced_gap_percent(result) is None
    ]
    raw_exceptions = [
        result
        for result in raw_regressions
        if _balanced_gap_percent(result) is not None
        and _balanced_gap_percent(result) <= gap_threshold
    ]
    worst = (
        max(
            balanced_evaluable,
            key=lambda result: _balanced_gap_percent(result),
        )
        if balanced_evaluable
        else None
    )
    gate_regression_count = len(balanced_regressions) + len(paired_metric_errors)
    return {
        "row_count": len(results),
        "successful_row_count": len(successful),
        "error_row_count": len(errors),
        "error_case_ids": [result["case_id"] for result in errors],
        "paired_metric_error_row_count": len(paired_metric_errors),
        "paired_metric_error_case_ids": [
            result["case_id"] for result in paired_metric_errors
        ],
        "gap_threshold_percent": gap_threshold,
        "raw_independent_median_regression_row_count": len(raw_regressions),
        "order_balanced_total_duration_evaluable_row_count": len(balanced_evaluable),
        "order_balanced_total_duration_regression_row_count": len(balanced_regressions),
        "raw_median_exception_row_count": len(raw_exceptions),
        "gate_regression_row_count": gate_regression_count,
        # Compatibility alias now intentionally follows the balanced gate.
        "regression_row_count": gate_regression_count,
        "worst_order_balanced_case_id": None if worst is None else worst["case_id"],
        "worst_order_balanced_gap_percent": (
            None if worst is None else _balanced_gap_percent(worst)
        ),
    }


def _write_outputs(args, payload) -> None:
    results = payload["results"]
    if args.json_output is not None:
        _write_json_atomic(args.json_output, payload)
    if args.csv_output is not None:
        _write_text_atomic(args.csv_output, _render_csv(results))
    if args.markdown_output is not None:
        _write_text_atomic(
            args.markdown_output,
            _render_markdown(results, args.gap_threshold),
        )


def _collect_provenance(
    args,
    runtime,
    specs: Sequence[MLAPerformanceCaseSpec],
) -> dict[str, Any]:
    """Collect portable provenance without serializing local paths or environment."""

    torch = runtime["torch"]
    import cutlass
    from flashinfer.artifacts import ArtifactPath, CheckSumHash
    from flashinfer.jit.env import FLASHINFER_CUBIN_DIR

    properties = torch.cuda.get_device_properties(args.device)
    try:
        cubin_version = importlib.metadata.version("flashinfer-cubin")
    except importlib.metadata.PackageNotFoundError:
        cubin_version = None
    cutlass_version = getattr(cutlass, "__version__", "unknown")
    if cutlass_version.partition("+")[0] != _EXPECTED_CUTLASS_DSL_VERSION:
        raise RuntimeError(
            f"expected imported CUTLASS DSL {_EXPECTED_CUTLASS_DSL_VERSION}, "
            f"got {cutlass_version}"
        )

    artifact = None
    if args.reference_backend == _TRTLLM_BACKEND:
        artifact_manifest = (
            Path(FLASHINFER_CUBIN_DIR) / ArtifactPath.TRTLLM_GEN_FMHA / "checksums.txt"
        )
        artifact_manifest_sha = _validate_trt_artifact_manifest(
            artifact_manifest,
            CheckSumHash.TRTLLM_GEN_FMHA,
            os.environ.get("FLASHINFER_CUBIN_CHECKSUM_DISABLED"),
        )
        artifact = {
            "artifact_path": ArtifactPath.TRTLLM_GEN_FMHA,
            "expected_manifest_sha256": CheckSumHash.TRTLLM_GEN_FMHA,
            "verified_manifest_sha256": artifact_manifest_sha,
            "flashinfer_cubin_version": cubin_version,
            "flashinfer_cubin_matches_flashinfer": (
                cubin_version
                == getattr(runtime["flashinfer"], "__version__", "unknown")
            ),
        }
    source_control = {
        "flashinfer_git_head": _git_value("rev-parse", "HEAD"),
        "flashinfer_tracked_worktree_dirty": bool(
            _git_value("status", "--porcelain", "--untracked-files=no")
        ),
    }
    versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer": getattr(runtime["flashinfer"], "__version__", "unknown"),
        "flashinfer_cubin": cubin_version,
        "nvidia_cutlass_dsl": cutlass_version,
        "nvidia_cutlass_dsl_distribution": importlib.metadata.version(
            "nvidia-cutlass-dsl"
        ),
        "cutlass": cutlass_version,
    }
    gpu = {
        "device_index": args.device,
        "name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(args.device)),
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "l2_cache_bytes": properties.L2_cache_size,
    }
    semantics_by_q_len = {
        str(seq_len_q): _execution_semantics(
            args.prims_ts_interface,
            seq_len_q,
            args.reference_backend,
        )
        for seq_len_q in sorted({spec.seq_len_q for spec in specs})
    }
    public_api = next(iter(semantics_by_q_len.values()))["prims_ts_public_api"]
    timing = _timing_contract(args)
    source_hashes = _public_source_hashes()
    execution_dependencies = {
        "source_control": source_control,
        "versions": versions,
        "gpu": gpu,
        "source_hashes": source_hashes,
        "reference_backend": args.reference_backend,
        "semantic_contracts_by_q_len": semantics_by_q_len,
        "oracle_precision": _oracle_precision_contract(torch),
        "timing_contract": timing,
    }
    if artifact is not None:
        execution_dependencies["trtllm_artifact"] = artifact
    dependencies_sha256 = _canonical_sha256(execution_dependencies)
    reference_label = _reference_label(args.reference_backend)
    reference_provenance = {
        "public_api": "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla",
        "backend": args.reference_backend,
    }
    if args.reference_backend == _TRTLLM_BACKEND:
        reference_provenance.update(
            {
                "enable_pdl": _TRTLLM_ENABLE_PDL,
                "tactic": -1,
                "artifact": artifact,
            }
        )
    else:
        reference_provenance["cute_dsl_impl"] = "monolithic"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_dependencies_sha256": dependencies_sha256,
        "execution_dependencies": execution_dependencies,
        "versions": versions,
        "gpu": gpu,
        "attention_ts_mla": {
            "interface": args.prims_ts_interface,
            "public_api": public_api,
            "integration_source_sha256": source_hashes["attention_ts"],
            "compile_cache": _cache_info_dict(runtime["cache_info"]()),
        },
        reference_label: reference_provenance,
        "matrix": {
            "suite": args.suite,
            "num_heads": sorted({spec.num_heads for spec in specs}),
            "batch_sizes": sorted({spec.batch_size for spec in specs}),
            "max_seq_lens": sorted({spec.max_seq_len for spec in specs}),
            "input_dtypes": sorted({spec.dtype_name for spec in specs}),
            "output_dtype": "bf16",
            "q_lens": sorted({spec.seq_len_q for spec in specs}),
            "page_size": _PAGE_SIZE,
        },
        "timing": timing,
        "accuracy": {
            "reference": "backend-neutral FP32 SDPA-equivalent attention",
            "checked_before_and_after_timing": True,
        },
        "run_controls": {
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "gap_threshold_percent": args.gap_threshold,
            "prims_ts_interface": args.prims_ts_interface,
            "reference_backend": args.reference_backend,
            "timing_cache_mode": args.timing_cache_mode,
        },
    }


def _run_signature(
    specs: Sequence[MLAPerformanceCaseSpec], args, provenance
) -> dict[str, Any]:
    dependencies = provenance["execution_dependencies"]
    dependencies_sha256 = _canonical_sha256(dependencies)
    if dependencies_sha256 != provenance["execution_dependencies_sha256"]:
        raise RuntimeError("execution dependency provenance digest is inconsistent")
    contract = {
        "schema_version": _SCHEMA_VERSION,
        "selected_case_specs": [asdict(spec) for spec in specs],
        "execution_dependencies_sha256": dependencies_sha256,
        "run_controls": {
            "gap_threshold_percent": args.gap_threshold,
            "continue_on_error": args.continue_on_error,
            "fail_on_regression": args.fail_on_regression,
            "prims_ts_interface": getattr(args, "prims_ts_interface", "wrapper"),
            "reference_backend": args.reference_backend,
            "timing_cache_mode": args.timing_cache_mode,
        },
    }
    source_session_sha256 = _canonical_sha256(contract)
    return {
        "sha256": source_session_sha256,
        "source_session_sha256": source_session_sha256,
        "execution_dependencies_sha256": dependencies_sha256,
        "contract": contract,
    }


def _resume_results(
    path: Path,
    signature: dict[str, Any],
    specs: Sequence[MLAPerformanceCaseSpec],
):
    payload = _load_json(path)
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"resume schema mismatch: {payload.get('schema_version')} != {_SCHEMA_VERSION}"
        )
    prior_signature = payload.get("run_signature", {})
    if prior_signature.get("sha256") != signature["sha256"]:
        raise ValueError(
            "resume run signature differs; use --regressions-from for a fresh "
            "rerun after source, matrix, or timing changes"
        )
    source_session_sha256 = signature["source_session_sha256"]
    if payload.get("source_session_sha256") != source_session_sha256:
        raise ValueError("resume payload source session digest differs")
    if (
        payload.get("provenance", {}).get("execution_dependencies_sha256")
        != signature["execution_dependencies_sha256"]
    ):
        raise ValueError("resume provenance dependency digest differs")
    prims_ts_interface = (
        signature["contract"]
        .get("run_controls", {})
        .get("prims_ts_interface", "wrapper")
    )
    reference_backend = (
        signature["contract"]
        .get("run_controls", {})
        .get("reference_backend", _TRTLLM_BACKEND)
    )
    expected_case_records = {
        spec.case_id: _case_contract_record(
            spec,
            source_session_sha256,
            prims_ts_interface,
            reference_backend,
        )
        for spec in specs
    }
    if [asdict(spec) for spec in specs] != signature["contract"]["selected_case_specs"]:
        raise ValueError("resume specs differ from the signed selected case specs")
    seen = set()
    for row in payload.get("results", []):
        case_id = row.get("case_id")
        if case_id in seen:
            raise ValueError(f"resume payload contains duplicate row {case_id}")
        if case_id not in expected_case_records:
            raise ValueError(f"resume payload contains unexpected row {case_id}")
        if row.get("source_session_sha256") != source_session_sha256:
            raise ValueError(f"resume row {case_id} belongs to a different session")
        _validate_row_case_contract(row, expected_case_records[case_id])
        seen.add(case_id)
    return {
        row["case_id"]: row
        for row in payload.get("results", [])
        if row.get("status") == "ok"
    }


def _print_result(result: dict[str, Any]) -> None:
    shape = result["shape"]
    if result.get("status") != "ok":
        print(
            f"\n[{result['case_id']}] ERROR B={shape['batch_size']} "
            f"Q={shape['q_len']} H={shape['num_qo_heads']} "
            f"KV={shape['max_seq_len']} "
            f"{shape['input_dtype']}→bf16: {result['reason']}"
        )
        return
    ts = result["attention_ts"]
    reference_backend, _, reference = _reference_record(result)
    reference_name = _reference_display_name(reference_backend)
    print(
        f"\n[{result['case_id']}] B={shape['batch_size']} Q={shape['q_len']} "
        f"H={shape['num_qo_heads']} "
        f"KV={shape['min_seq_len']}..{shape['max_seq_len']} "
        f"{shape['input_dtype']}→bf16 page32"
    )
    print(
        f"  Attention-TS ({ts['interface']}) {ts['median_us']:9.3f} us; "
        f"{reference_name} {reference['median_us']:9.3f} us; "
        f"raw gap {result['attention_ts_gap_percent_vs_reference']:+8.3f}%; "
        f"balanced total gap {_balanced_gap_percent(result):+8.3f}%"
    )
    paired = result["paired_two_order_cycles"]
    print(
        f"  paired means TS/reference={ts['arithmetic_mean_us']:.3f}/"
        f"{reference['arithmetic_mean_us']:.3f} us; cycle gap median/p95="
        f"{paired['cycle_gap_percent_median']:+.3f}/"
        f"{paired['cycle_gap_percent_p95']:+.3f}%"
    )
    print(
        "  performance disposition: "
        f"{result['performance_disposition']['classification']} "
        f"(gate={result['performance_disposition']['gate_metric']})"
    )
    print(f"  TS policy: {ts['policy']}")
    print(
        f"  correctness max abs TS/reference={ts['max_abs_error']:.6g}/"
        f"{reference['max_abs_error']:.6g}; backend diff="
        f"{result['backend_max_abs_difference']:.6g}"
    )


def _load_runtime() -> dict[str, Any]:
    try:
        import flashinfer
        import torch
        from benchmarks.routines.attention_ts_benchmark.mla_fixtures import (
            make_attention_ts_mla_decode_case,
        )
        from flashinfer.attention.prims_ts import BatchMLADecodePagedTSWrapper
        from flashinfer.attention.prims_ts.mla_decode import _get_compiled_mla_decode
        from flashinfer.mla import (
            get_prims_ts_batch_decode_mla_workspace_size,
            prims_ts_batch_decode_with_kv_cache_mla,
            trtllm_batch_decode_with_kv_cache_mla,
        )
        from flashinfer.utils import (
            get_device_sm_count,
            get_trtllm_gen_multi_ctas_kv_counter_bytes,
        )
    except (ImportError, OSError, RuntimeError) as error:
        diagnostic = _cuda_runtime_error(error)
        if diagnostic is error:
            raise
        raise diagnostic from error

    return {
        "torch": torch,
        "flashinfer": flashinfer,
        "wrapper_type": BatchMLADecodePagedTSWrapper,
        "standalone_ts_decode": prims_ts_batch_decode_with_kv_cache_mla,
        "get_ts_workspace_size": get_prims_ts_batch_decode_mla_workspace_size,
        "reference_decode": trtllm_batch_decode_with_kv_cache_mla,
        "get_device_sm_count": get_device_sm_count,
        "get_trtllm_counter_bytes": get_trtllm_gen_multi_ctas_kv_counter_bytes,
        "make_case": make_attention_ts_mla_decode_case,
        "cache_info": _get_compiled_mla_decode.cache_info,
        "clear_cache": _get_compiled_mla_decode.cache_clear,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.reference_backend = _reference_backend_for_suite(args.suite)
    args.timing_cache_mode = _timing_cache_mode_for_suite(args.suite)
    args.effective_command = effective_command(Path(__file__), argv)
    if args.q_len <= 0:
        parser.error("--q-len must be positive")
    if args.suite != "signoff" and args.q_len != _Q_LEN:
        parser.error("--q-len applies only to --suite signoff")
    if args.list_cases:
        for spec in _catalog(args.suite, args.q_len):
            print(spec.case_id)
        return 0
    if args.resume is not None:
        if args.regressions_from is not None:
            parser.error("--resume and --regressions-from are mutually exclusive")
        if args.json_output is None:
            args.json_output = args.resume
    if args.warmup_iters <= 0:
        parser.error("--warmup-iters must be positive")
    if args.iters < 2 or args.iters % 2:
        parser.error("--iters must be even and at least 2")
    if not math.isfinite(args.gap_threshold) or args.gap_threshold < 0:
        parser.error("--gap-threshold must be finite and non-negative")
    specs = _select_specs(parser, args)
    if not specs:
        print("No rows selected.")
        return 0

    runtime = _load_runtime()
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error(f"--device {args.device} is outside the visible CUDA range")
    torch.cuda.set_device(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    device_name = torch.cuda.get_device_name(args.device)
    if capability != (10, 0) or "B200" not in device_name.upper():
        parser.error(
            "Attention-TS MLA benchmarking requires NVIDIA B200 / SM100a; "
            f"device {args.device} is {device_name} with capability {capability}"
        )

    cold_l2_scrubber = None
    args.cold_l2_scrub_contract = None
    if args.timing_cache_mode == "cold-l2":
        cold_l2_scrubber = _prepare_cold_l2_scrubber(torch, args.device)
        args.cold_l2_scrub_contract = cold_l2_scrubber.metadata()
    runtime["clear_cache"]()
    provenance = _collect_provenance(args, runtime, specs)
    signature = _run_signature(specs, args, provenance)
    source_session_sha256 = signature["source_session_sha256"]
    case_contracts = {
        spec.case_id: _case_contract_record(
            spec,
            source_session_sha256,
            args.prims_ts_interface,
            args.reference_backend,
        )
        for spec in specs
    }
    completed = (
        _resume_results(args.resume, signature, specs)
        if args.resume is not None
        else {}
    )
    print(json.dumps(provenance, indent=2))
    print(
        "\nCorrectness precedes and follows paired graph timing. "
        f"Selected rows: {len(specs)}; resumed rows: {len(completed)}; "
        f"PrimTS interface: {args.prims_ts_interface}; reference: "
        f"{args.reference_backend}; cache mode: {args.timing_cache_mode}."
    )

    results_by_id = dict(completed)
    completed_compile_reuse_groups = {
        spec.compile_reuse_group
        for spec in specs
        if spec.case_id in completed
        and spec.compile_reuse_group is not None
        and completed[spec.case_id].get("status") == "ok"
    }
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "status": "running",
        "source_session_sha256": source_session_sha256,
        "run_signature": signature,
        "provenance": provenance,
        "run_plan": [asdict(spec) for spec in specs],
        "case_contracts": case_contracts,
        "results": [],
        "summary": None,
        "final_attention_ts_cache": None,
    }

    def refresh_payload():
        payload["results"] = [
            results_by_id[spec.case_id]
            for spec in specs
            if spec.case_id in results_by_id
        ]
        payload["summary"] = _summary(payload["results"], args.gap_threshold)
        payload["final_attention_ts_cache"] = _cache_info_dict(runtime["cache_info"]())
        _write_outputs(args, payload)

    def verify_dependencies(phase: str, *, completion: bool = False) -> None:
        try:
            _verify_dependency_immutability(provenance, phase=phase)
            if completion:
                _verify_completion_source_hashes(provenance)
        except Exception as error:
            payload["status"] = "error"
            payload["failure"] = f"{type(error).__name__}: {error}"
            payload["dependency_drift_phase"] = phase
            refresh_payload()
            raise

    refresh_payload()
    for spec in specs:
        verify_dependencies(f"before case {spec.case_id}")
        if spec.case_id in completed:
            print(f"\n[{spec.case_id}] resumed")
            continue
        try:
            result = _run_case(spec, args, runtime, cold_l2_scrubber)
            _validate_compile_reuse(
                spec,
                result,
                completed_compile_reuse_groups,
            )
        except Exception as error:
            if not args.continue_on_error:
                payload["status"] = "error"
                payload["failure"] = f"{type(error).__name__}: {error}"
                refresh_payload()
                raise
            result = _error_result(
                spec,
                error,
                args.prims_ts_interface,
                args.gap_threshold,
                args.reference_backend,
            )
        result["source_session_sha256"] = source_session_sha256
        result["case_contract"] = case_contracts[spec.case_id]["contract"]
        result["case_contract_sha256"] = case_contracts[spec.case_id]["sha256"]
        results_by_id[spec.case_id] = result
        _print_result(result)
        refresh_payload()
        torch.cuda.empty_cache()

    verify_dependencies("at completion", completion=True)
    payload["status"] = "complete"
    refresh_payload()
    if args.json_output is not None:
        print(f"\nWrote {args.json_output}")
    if args.csv_output is not None:
        print(f"Wrote {args.csv_output}")
    if args.markdown_output is not None:
        print(f"Wrote {args.markdown_output}")
    summary = payload["summary"]
    if summary["error_row_count"] or summary["paired_metric_error_row_count"]:
        return 1
    if args.fail_on_regression and summary["gate_regression_row_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
