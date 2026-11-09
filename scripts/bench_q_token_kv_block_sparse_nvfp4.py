#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compare NVFP4-KV and FP8-KV Prims-TS QToken-KvBlock-Sparse-Attention.

Every row runs the public prepared wrapper three ways on matched values:
interleaved NVFP4 K/V with FP8 Q (FP8 MMA), the same cache with BF16 Q
(BF16 MMA), and plain FP8 Q/K/V holding the E4M3 rounding of the dequantized
NVFP4 values. Timing uses cold-L2 CUDA graphs.
"""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("nvfp4-fp8mma", "nvfp4-bf16mma", "fp8")
PRIMARY = "nvfp4-fp8mma"
# Relative L2 limits against the FP32 oracle over dequantized NVFP4 values.
# BF16 MMA is exact up to BF16 P/O rounding; the FP8 paths also round Q and P
# (and FP8 KV rounds the dequantized cache) to E4M3.
REL_L2_LIMITS = {"nvfp4-fp8mma": 0.06, "nvfp4-bf16mma": 0.01, "fp8": 0.06}
CHECK_TOKENS = 8


def load_cases(suite):
    """Expand a checked-in manifest into stable, human-readable case IDs."""
    spec = json.loads(Path(suite).read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported QSA NVFP4 manifest version")
    common = {
        k: spec[k]
        for k in (
            "context_tokens",
            "topk_blocks",
            "page_size",
            "overlap",
            "count_current_block",
            "seed",
        )
    }
    if common["page_size"] % 4 or common["context_tokens"] % common["page_size"]:
        raise ValueError("Page size must be a multiple of four dividing the context")
    if not 0 <= common["overlap"] < 1 or common["topk_blocks"] <= 0:
        raise ValueError("Overlap must be in [0, 1) and topk positive")
    rows = []
    if spec["phase"] == "decode":
        grid = itertools.product(spec["tp"], spec["batches"], spec["mtp"])
        for tp, batch, mtp in grid:
            rows.append(
                dict(
                    id=f"decode-tp{tp}-b{batch}-mtp{mtp}",
                    phase="decode",
                    tp=tp,
                    batch=batch,
                    query_tokens=mtp + 1,
                    group_size=mtp + 1,
                    mtp=mtp,
                )
            )
    elif spec["phase"] == "prefill":
        grid = itertools.product(spec["tp"], spec["query_tokens"], spec["group_sizes"])
        for tp, tokens, group in grid:
            if tokens % group:
                raise ValueError("Prefill query tokens must divide into groups")
            rows.append(
                dict(
                    id=f"prefill-tp{tp}-q{tokens}-g{group}",
                    phase="prefill",
                    tp=tp,
                    batch=1,
                    query_tokens=tokens,
                    group_size=group,
                )
            )
    else:
        raise ValueError("Phase must be decode or prefill")
    from attention_ts_qsa_nvfp4_fixtures import TP_DEGREES

    heads_q, heads_kv = spec["model_heads"]
    if heads_kv <= 0 or heads_q % heads_kv:
        raise ValueError("Model Q heads must be a positive multiple of KV heads")
    for row in rows:
        tp = row["tp"]
        if tp not in TP_DEGREES or heads_kv % tp or row["batch"] <= 0:
            raise ValueError("Unsupported TP or batch")
        if not 1 <= row["group_size"] <= 8:
            raise ValueError("Group size must be in [1, 8]")
        row["heads_q"], row["heads_kv"] = heads_q // tp, heads_kv // tp
        row.update(common)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Suite must contain nonempty, unique cases")
    return rows


def timing_summary(samples):
    """Reject invalid timings before calculating latency diagnostics."""
    if not samples or any(not math.isfinite(t) or t <= 0 for t in samples):
        raise ValueError("Timing samples must be positive, finite microseconds")
    ordered = sorted(samples)
    return dict(
        median_us=statistics.median(samples),
        p10_us=ordered[int(0.1 * (len(ordered) - 1))],
        p90_us=ordered[int(0.9 * (len(ordered) - 1))],
        samples_us=samples,
    )


def summarize(rows, expected_ids, *, min_speedup=None):
    """Require every requested row and backend; failures are never wins."""
    failures, below = [], []
    for row in rows:
        if row.get("status") != "ok" or any(
            row.get("backends", {}).get(n, {}).get("status") != "ok" for n in BACKENDS
        ):
            failures.append(row["id"])
            continue
        if min_speedup is not None and row["speedups"]["fp8"] < min_speedup:
            below.append(row["id"])
    complete = len(rows) == len(expected_ids) and {r["id"] for r in rows} == set(
        expected_ids
    )
    return dict(
        requested=len(expected_ids),
        completed=len(rows),
        failed=failures,
        cases_below_target=below,
        min_speedup=min_speedup,
        passed=bool(complete and expected_ids and not failures and not below),
    )


class Pools:
    """Matched NVFP4/FP8 caches, reused across rows with the same KV heads."""

    def __init__(self, device):
        self.device = device
        self.key = None
        self.data = None

    def get(self, heads_kv, pages, page_size, seed):
        from attention_ts_qsa_nvfp4_fixtures import make_pools

        key = (heads_kv, page_size, seed)
        if self.key != key or self.data["page_order"].numel() < pages:
            import torch

            self.data = None
            torch.cuda.empty_cache()
            self.data = make_pools(heads_kv, pages, page_size, seed, self.device)
            self.key = key
        return self.data


def make_backend(name, case, fx):
    """Prepare one public wrapper call; allocation and planning stay untimed."""
    import torch
    from flashinfer.attention.prims_ts import (
        QTokenKvBlockSparsePagedTSWrapper,
        get_q_token_kv_block_sparse_workspace_size,
    )

    storage = "fp8" if name == "fp8" else "nvfp4"
    q = fx.q_bf16 if name == "nvfp4-bf16mma" else fx.q_fp8
    k_cache, v_cache = fx.caches[storage]
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    workspace = torch.zeros(
        get_q_token_kv_block_sparse_workspace_size(
            q,
            k_cache,
            fx.block_table,
            block_topk=case["topk_blocks"],
            max_seq_len_kv=case["context_tokens"],
            out_dtype=torch.bfloat16,
            group_size=case["group_size"],
            kv_block_size=4,
            split_kv=case["phase"] == "decode",
            share_pattern_across_kv_heads=True,
            count_current_block_in_topk=case["count_current_block"],
        ),
        dtype=torch.uint8,
        device=q.device,
    )
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        fx.num_routes,
        case["heads_q"],
        case["heads_kv"],
        256,
        4,
        case["page_size"],
        case["topk_blocks"],
        case["context_tokens"],
        device=q.device,
        workspace_buffer=workspace,
        group_size=case["group_size"],
        packed_query=False,
        split_kv=case["phase"] == "decode",
        share_pattern_across_kv_heads=True,
        count_current_block_in_topk=case["count_current_block"],
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=torch.bfloat16,
    )

    def run():
        """Metadata, attention and any split reduction in one captured call."""
        wrapper.run(
            q,
            (k_cache, v_cache),
            fx.block_table,
            fx.ids,
            seq_lens=fx.seq_lens,
            sm_scale=256**-0.5,
            v_scale=1.0,
            out=out,
        )

    return SimpleNamespace(
        run=run, out=out, workspace=workspace, wrapper=wrapper, storage=storage
    )


def make_fixture(case, pools):
    """Build matched inputs for one row; every conversion stays untimed."""
    import torch
    from attention_ts_qsa_nvfp4_fixtures import build_indices

    device = pools.device
    indices = build_indices(case)
    pages_per_req = case["context_tokens"] // case["page_size"]
    data = pools.get(
        case["heads_kv"],
        case["batch"] * pages_per_req,
        case["page_size"],
        case["seed"],
    )
    block_table = data["page_order"][: case["batch"] * pages_per_req].view(
        case["batch"], pages_per_req
    )
    groups = case["query_tokens"] // case["group_size"]
    gen = torch.Generator(device=device).manual_seed(case["seed"] + 7)
    q_bf16 = torch.randn(
        (case["batch"], groups, case["group_size"], case["heads_q"], 256),
        device=device,
        generator=gen,
    ).to(torch.bfloat16)
    return SimpleNamespace(
        indices=indices,
        ids=torch.from_numpy(indices["ids"]).to(device),
        block_table=block_table,
        caches=dict(nvfp4=data["nvfp4"], fp8=data["fp8"]),
        q_bf16=q_bf16,
        q_fp8=q_bf16.to(torch.float8_e4m3fn),
        seq_lens=torch.full(
            (case["batch"],), case["context_tokens"], dtype=torch.int32, device=device
        ),
        num_routes=case["batch"] * groups,
    )


def benchmark_case(case, args, pools):
    """Check, capture cold graphs, alternate order and recheck after timing."""
    import torch
    from attention_ts_qsa_nvfp4_fixtures import accounting, check_output, reference
    from attention_ts_sparse_mla_timing import ColdL2GraphBenchmark

    fx = make_fixture(case, pools)
    total_tokens = case["batch"] * case["query_tokens"]
    check = sorted(
        {
            *range(min(CHECK_TOKENS // 2, total_tokens)),
            *range(max(0, total_tokens - CHECK_TOKENS // 2), total_tokens),
        }
    )
    refs = {}
    for q_name, q in (("bf16", fx.q_bf16), ("fp8", fx.q_fp8)):
        refs[q_name] = reference(
            case,
            fx.indices,
            q.reshape(total_tokens, case["heads_q"], 256).float(),
            *fx.caches["nvfp4"],
            fx.block_table,
            check,
        )
    backends = {name: make_backend(name, case, fx) for name in BACKENDS}

    def verify(name):
        backend = backends[name]
        out = backend.out.reshape(total_tokens, case["heads_q"], 256)[check]
        expected = refs["bf16" if name == "nvfp4-bf16mma" else "fp8"]
        return check_output(out, expected, rel_l2_limit=REL_L2_LIMITS[name])

    results, runners = {}, {}
    for name, backend in backends.items():
        backend.run()
        torch.cuda.synchronize()
        accuracy = {"eager": verify(name)}
        runner = ColdL2GraphBenchmark(
            backend.run,
            device=fx.ids.device,
            samples_per_replay=args.samples_per_replay,
        )
        backend.out.fill_(float("nan"))
        torch.cuda.synchronize()
        runner.sample()
        accuracy["graph"] = verify(name)
        runners[name] = runner
        results[name] = dict(status="ok", accuracy=accuracy, storage=backend.storage)
    samples = {name: [] for name in backends}
    names = list(backends)
    for replay in range(args.replays):
        # Rotate the first backend and reverse alternate cycles.
        offset = (replay // 2) % len(names)
        order = names[offset:] + names[:offset]
        if replay % 2:
            order.reverse()
        for name in order:
            samples[name].extend(runners[name].sample())
    work = accounting(case, fx.indices)
    for name in backends:
        results[name].update(timing_summary(samples[name]))
        results[name]["accuracy"]["after_timing"] = verify(name)
        seconds = results[name]["median_us"] * 1e-6
        q_dtype = "bf16" if name == "nvfp4-bf16mma" else "fp8"
        nbytes = work["bytes"][f"{backends[name].storage}-q{q_dtype}"]
        results[name].update(
            bytes=nbytes,
            achieved_tbps=nbytes / seconds / 1e12,
            achieved_tflops=work["flops"] / seconds / 1e12,
        )
    first = next(iter(runners.values()))
    row = dict(
        **case,
        status="ok",
        backends=results,
        flops=work["flops"],
        unique_blocks_per_request=work["unique_blocks_per_request"],
        mean_adjacent_overlap=work["mean_adjacent_overlap"],
        l2_bytes=first.l2_bytes,
        eviction_bytes=first.eviction_bytes,
        speedups={
            name: results[name]["median_us"] / results[PRIMARY]["median_us"]
            for name in backends
            if name != PRIMARY
        },
    )
    # Graph owners, plans and workspaces go out of scope with this frame.
    torch.cuda.synchronize()
    return row


def provenance(args):
    """Verify the import root and record path-redacted source identities."""
    sys.path.insert(0, str(args.source_root))
    import flashinfer
    import torch
    from cuda.bindings import runtime as cudart

    if Path(flashinfer.__file__).resolve().parent != args.source_root / "flashinfer":
        raise RuntimeError("FlashInfer did not import from --source-root")
    torch.cuda.set_device(args.device)

    def git(*cmd):
        return subprocess.check_output(
            ["git", "-C", str(args.source_root), *cmd], text=True
        ).strip()

    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    digest = hashlib.sha256()
    for path in sorted(
        (args.source_root / "flashinfer/attention/prims_ts").rglob("*.py")
    ):
        digest.update(str(path.relative_to(args.source_root)).encode())
        digest.update(path.read_bytes())
    properties = torch.cuda.get_device_properties(args.device)
    driver = cudart.cudaDriverGetVersion()
    scripts = Path(__file__).resolve().parent
    return dict(
        source_revision=git("rev-parse", "HEAD"),
        source_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        source_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
        prims_source_sha256=digest.hexdigest(),
        sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                "driver": Path(__file__),
                "fixtures": scripts / "attention_ts_qsa_nvfp4_fixtures.py",
                "timing": scripts / "attention_ts_sparse_mla_timing.py",
                "suite": args.suite,
            }.items()
        },
        gpu=properties.name,
        gpu_identity_sha256=hashlib.sha256(str(properties.uuid).encode()).hexdigest(),
        sm_count=properties.multi_processor_count,
        compute_capability=[properties.major, properties.minor],
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cuda_driver_version=driver[1]
        if driver[0] == cudart.cudaError_t.cudaSuccess
        else None,
        dsl=version("nvidia-cutlass-dsl"),
    )


def main(argv=None):
    """List CPU-only inventories or checkpoint a selected CUDA campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "suites/qsa_nvfp4/decode.json"
    )
    parser.add_argument(
        "--case", action="append", help="Repeat a stable ID from --list"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--samples-per-replay", type=int, default=4)
    parser.add_argument(
        "--quick", action="store_true", help="One sample per replay; functional only"
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        help="Optional FP8 / NVFP4-FP8-MMA median target, e.g. 1.3; no implicit gate",
    )
    args = parser.parse_args(argv)
    args.suite = args.suite.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        rows = load_cases(args.suite)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.case:
        unknown = set(args.case) - {r["id"] for r in rows}
        if unknown:
            parser.error(f"Unknown case IDs: {sorted(unknown)}")
        rows = [r for r in rows if r["id"] in args.case]
    if args.list:
        print(json.dumps(rows, indent=2))
        return 0
    if not args.source_root or not args.output:
        parser.error("--source-root and --output are required for execution")
    if args.output.exists():
        parser.error("Use a new --output file to preserve previous evidence")
    if args.replays <= 0 or args.replays % 6 or args.samples_per_replay <= 0:
        parser.error("--replays must be a positive multiple of 6; samples positive")
    if args.min_speedup is not None and (
        not math.isfinite(args.min_speedup) or args.min_speedup <= 0
    ):
        parser.error("--min-speedup must be finite and positive")
    args.source_root = args.source_root.resolve()
    if args.quick:
        args.samples_per_replay = 1
    report = dict(
        schema="qsa-nvfp4-v1",
        status="running",
        suite=args.suite.name,
        cases=[],
        requested=[r["id"] for r in rows],
        quick=args.quick,
        timing=dict(
            replays=args.replays,
            samples_per_replay=args.samples_per_replay,
            cache="cold-4xL2",
            cuda_graph=True,
            timed="prepared metadata, attention and required split reduction",
            excluded="index generation, planning, allocation, quantization, L2 eviction",
        ),
        numerical_contract=(
            "FP32 oracle over dequantized NVFP4 values on sampled tokens; "
            f"relative L2 limits {REL_L2_LIMITS}"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        """Atomically retain each completed row and any failing campaign."""
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    try:
        report["provenance"] = provenance(args)
        import torch

        pools = Pools(torch.device("cuda", args.device))
        # Group by KV heads so one pool, sized for the largest batch, serves them.
        for row in sorted(rows, key=lambda r: (r["heads_kv"], -r["batch"])):
            try:
                report["cases"].append(benchmark_case(row, args, pools))
            except Exception as error:
                report["cases"].append(
                    dict(
                        **row, status="failed", error=f"{type(error).__name__}: {error}"
                    )
                )
                raise
            save()
            result = report["cases"][-1]
            print(
                row["id"],
                {n: round(result["backends"][n]["median_us"], 2) for n in BACKENDS},
                {n: round(v, 3) for n, v in result["speedups"].items()},
                flush=True,
            )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["summary"] = summarize(
        report["cases"], report["requested"], min_speedup=args.min_speedup
    )
    report["status"] = "passed" if report["summary"]["passed"] else "failed"
    save()
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
