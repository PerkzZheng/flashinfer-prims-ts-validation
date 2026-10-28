#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compare NVFP4 and plain-FP8 Prims-TS MLA with FlashMLA, using cold-L2 graphs."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("d576", "d512-nvfp4-extra", "d512-fp8-extra")


def load_cases(suite):
    """Expand a checked-in workload manifest into stable, human-readable IDs."""
    spec = json.loads(suite.read_text())
    if spec["schema_version"] != 1:
        raise ValueError("Unsupported scaled MLA manifest version")
    rows = []
    for variant, b, q, h in itertools.product(
        spec["variants"], spec["batches"], spec["query_lengths"], spec["heads"]
    ):
        if variant not in VARIANTS or h not in (8, 16, 32, 64, 128):
            raise ValueError("Unsupported cache variant or head count")
        if min(b, q, spec["topk"], spec["swa_tokens"]) <= 0 or spec["page_size"] != 64:
            raise ValueError("Positive dimensions and page size 64 are required")
        if spec["context_tokens"] % 64 or spec["context_tokens"] < spec["topk"]:
            raise ValueError("Context must be page-aligned and at least topk")
        rows.append(
            dict(
                id=f"{variant}-b{b}-q{q}-h{h}",
                variant=variant,
                batch=b,
                queries=q,
                heads=h,
                **{
                    k: spec[k]
                    for k in (
                        "context_tokens",
                        "topk",
                        "page_size",
                        "swa_tokens",
                        "seed",
                    )
                },
            )
        )
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Suite must contain nonempty, unique cases")
    return rows


def timing_summary(samples):
    """Reject invalid timings before calculating absolute latency diagnostics."""
    if not samples or any(not math.isfinite(t) or t <= 0 for t in samples):
        raise ValueError("Timing samples must be positive, finite microseconds")
    ordered = sorted(samples)
    return dict(
        median_us=statistics.median(samples),
        p10_us=ordered[int(0.1 * (len(ordered) - 1))],
        p90_us=ordered[int(0.9 * (len(ordered) - 1))],
        samples_us=samples,
    )


def summarize(rows, expected_ids, *, without_flashmla, min_speedup=None):
    """Require every requested row/comparator; unavailable references are never wins."""
    failures, below = [], []
    for row in rows:
        expected = ["prims-nvfp4", "prims-fp8"]
        if not without_flashmla and row["variant"] != "d512-nvfp4-extra":
            expected.append("flashmla")
        if row.get("status") != "ok" or any(
            row.get("backends", {}).get(n, {}).get("status") != "ok" for n in expected
        ):
            failures.append(row["id"])
            continue
        if min_speedup is not None and any(
            row["speedups"].get(n, 0) < min_speedup for n in expected[1:]
        ):
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
        flashmla_omitted=without_flashmla,
        flashmla_unsupported=[
            r["id"] for r in rows if r["variant"] == "d512-nvfp4-extra"
        ],
        passed=bool(complete and expected_ids and not failures and not below),
    )


def make_prims(f, *, packed):
    """Prepare one public auto-dispatch call; conversions and allocation stay untimed."""
    import torch
    from attention_ts_scaled_mla_fixtures import expand_fp8
    from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

    primary, extra, scale = f.primary, f.extra, f.primary_scale
    fmt = (
        "nvfp4.fp8rope"
        if f.dim == 576
        else ("fp8" if primary.dtype == torch.float8_e4m3fn else "nvfp4")
    )
    extra_fmt = "nvfp4" if extra is not None else None
    if not packed:
        primary = expand_fp8(primary, f.dim, scale)
        extra = expand_fp8(extra, f.dim) if extra is not None else None
        fmt, extra_fmt, scale = "native", None, 1.0
    b, sq, h, _ = f.q.shape
    ids = f.indices.flatten(0, 1)
    extra_ids = None if f.extra_indices is None else f.extra_indices.flatten(0, 1)
    lengths = torch.full((b * sq,), ids.shape[-1], device=f.q.device, dtype=torch.int32)
    extra_lengths = (
        None if extra_ids is None else torch.full_like(lengths, extra_ids.shape[-1])
    )
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        device=f.q.device,
        batch_size=b,
        num_heads=h,
        max_seq_len_q=sq,
        max_topk=ids.shape[-1],
        max_extra_topk=0 if extra_ids is None else extra_ids.shape[-1],
        q_data_type=f.q.dtype,
        qk_rope_head_dim=f.dim - 512,
        kv_cache_format=fmt,
        extra_kv_cache_format=extra_fmt,
        assume_valid_prefix=True,
        return_lse=True,
    )
    out = torch.empty((b, sq, h, 512), device=f.q.device, dtype=torch.bfloat16)
    lse = torch.empty((b, sq, h), device=f.q.device, dtype=torch.float32)

    def run():
        """Include attention, source merging and split reduction in the captured call."""
        wrapper.run(
            f.q,
            primary,
            ids,
            lengths,
            extra_kv_cache=extra,
            extra_indices=extra_ids,
            extra_lengths=extra_lengths,
            kv_cache_scale=scale,
            softmax_scale=f.dim**-0.5,
            out=out,
            lse=lse,
            validate=False,
        )

    def read():
        """Return persistent outputs without an extraction kernel in the measured graph."""
        return out, lse

    return SimpleNamespace(
        run=run, read=read, fp8_mma=True, storage=fmt, physical_heads=h
    )


def make_flashmla(f):
    """Use native D576 or a labeled constant-scale D512 adapter, prepared once.

    FlashMLA consumes exact BF16 widenings of FP8 Q. H<64 is padded before
    capture; only native attention/reduction kernels are timed. Its D512 API
    cannot accept NVFP4 primary storage, so that recipe has no comparator.
    """
    import flash_mla
    import torch

    if f.case["variant"] == "d512-nvfp4-extra":
        raise NotImplementedError("FlashMLA D512 supports NVFP4 only in the extra pool")
    b, sq, h, d = f.q.shape
    physical_heads = max(h, 64)
    query = torch.zeros(
        (b, sq, physical_heads, d), device=f.q.device, dtype=torch.bfloat16
    )
    primary = f.primary
    storage = "nvfp4.fp8rope-352B"
    if d == 512:
        # A repeated 0.5 UE8M0 scale exactly describes the per-tensor E4M3
        # payload. This is a 528B adapter, not native 512B per-tensor support.
        scales = torch.full(
            (*primary.shape[:-1], 16), f.primary_scale, device=primary.device
        ).to(torch.float8_e8m0fnu)
        primary = torch.cat(
            (primary.view(torch.uint8), scales.view(torch.uint8)), -1
        ).contiguous()
        storage = "mxfp8-528B-constant-scale-adapter"
    out = torch.empty(
        (b, sq, physical_heads, 512), device=f.q.device, dtype=torch.bfloat16
    )
    scheduler, _ = flash_mla.get_mla_metadata()
    result = [None]

    def refresh_query():
        """Widen live FP8 values and retain padded zero heads outside capture."""
        query[:, :, :h].copy_(f.q)

    def run():
        """Keep FlashMLA's complete native attention and split-reduction call timed."""
        result[0] = flash_mla.flash_mla_with_kvcache(
            query,
            primary,
            None,
            None,
            512,
            scheduler,
            softmax_scale=d**-0.5,
            is_fp8_kvcache=True,
            indices=f.indices,
            extra_k_cache=f.extra,
            extra_indices_in_kvcache=f.extra_indices,
            out=out,
        )

    def read():
        """View logical heads and transpose FlashMLA's B,H,SQ LSE outside timing."""
        return out[:, :, :h], result[0][1][:, :h].transpose(1, 2)

    refresh_query()
    return SimpleNamespace(
        run=run,
        read=read,
        refresh_query=refresh_query,
        fp8_mma=False,
        storage=storage,
        physical_heads=physical_heads,
    )


def benchmark_case(case, args):
    """Check matched operands, capture cold graphs, alternate order and verify replay."""
    import torch
    from attention_ts_scaled_mla_fixtures import (
        check_output,
        make_fixture,
        reference,
        tensor_hash,
    )
    from attention_ts_sparse_mla_timing import ColdL2GraphBenchmark

    f = make_fixture(case, device=torch.device("cuda", args.device))
    fingerprints = {
        name: tensor_hash(getattr(f, name))
        for name in ("q", "primary", "extra", "indices", "extra_indices")
        if getattr(f, name) is not None
    }
    backends = {
        "prims-nvfp4": make_prims(f, packed=True),
        "prims-fp8": make_prims(f, packed=False),
    }
    results = {}
    if args.without_flashmla:
        results["flashmla"] = dict(
            status="omitted", reason="Explicit --without-flashmla diagnostic run"
        )
    elif case["variant"] == "d512-nvfp4-extra":
        results["flashmla"] = dict(
            status="unsupported",
            reason="D512 NVFP4 primary cache is not supported by FlashMLA",
        )
    else:
        backends["flashmla"] = make_flashmla(f)
    refs = {
        mma: reference(f, fp8_mma=mma) for mma in {b.fp8_mma for b in backends.values()}
    }
    runners = {}
    for name, backend in backends.items():
        backend.run()
        expected = refs[backend.fp8_mma]
        accuracy = {
            "eager": check_output(*backend.read(), expected, fp8_mma=backend.fp8_mma)
        }
        runner = ColdL2GraphBenchmark(
            backend.run, device=f.q.device, samples_per_replay=args.samples_per_replay
        )
        for output in backend.read():
            output.fill_(float("nan"))
        torch.cuda.synchronize()
        runner.sample()
        accuracy["graph"] = check_output(
            *backend.read(), expected, fp8_mma=backend.fp8_mma
        )
        runners[name] = runner
        results[name] = dict(
            status="ok",
            accuracy=accuracy,
            storage=backend.storage,
            physical_heads=backend.physical_heads,
        )
    samples = {name: [] for name in backends}
    names = list(backends)
    for replay in range(args.replays):
        # Rotate the first backend, and reverse alternate cycles. A multiple
        # of 12 replays balances order for both two- and three-backend rows.
        offset = (replay // 2) % len(names)
        order = names[offset:] + names[:offset]
        if replay % 2:
            order.reverse()
        for name in order:
            samples[name].extend(runners[name].sample())
    for name, backend in backends.items():
        results[name].update(timing_summary(samples[name]))
        results[name]["accuracy"]["after_timing"] = check_output(
            *backend.read(), refs[backend.fp8_mma], fp8_mma=backend.fp8_mma
        )
    # Exercise graph-live queries and addresses after scored timing, without
    # recapture or adding input updates to any measured attention invocation.
    f.q.copy_((f.q.float() * -0.5).to(f.q.dtype))
    f.indices.copy_(f.indices.roll(1, dims=0))
    if f.extra_indices is not None:
        f.extra_indices.copy_(f.extra_indices.roll(1, dims=0))
    for backend in backends.values():
        if hasattr(backend, "refresh_query"):
            backend.refresh_query()
        for output in backend.read():
            output.fill_(float("nan"))
    torch.cuda.synchronize()
    refs = {mma: reference(f, fp8_mma=mma) for mma in refs}
    for name, backend in backends.items():
        runners[name].sample()
        results[name]["accuracy"]["live_replay"] = check_output(
            *backend.read(), refs[backend.fp8_mma], fp8_mma=backend.fp8_mma
        )
    first = next(iter(runners.values()))
    return dict(
        **case,
        status="ok",
        fingerprints=fingerprints,
        backends=results,
        l2_bytes=first.l2_bytes,
        eviction_bytes=first.eviction_bytes,
        speedups={
            name: results[name]["median_us"] / results["prims-nvfp4"]["median_us"]
            for name in backends
            if name != "prims-nvfp4"
        },
    )


def setup_sources(args):
    """Verify actual import roots and record public, path-redacted source identities."""
    sys.path.insert(0, str(args.source_root))
    import flashinfer
    import torch
    from bench_attention_ts_sparse_mla import provenance

    if Path(flashinfer.__file__).resolve().parent != args.source_root / "flashinfer":
        raise RuntimeError("FlashInfer did not import from --source-root")
    torch.cuda.set_device(args.device)
    info = provenance(args.source_root)
    source_digest = hashlib.sha256()
    for path in sorted(
        (args.source_root / "flashinfer/attention/prims_ts").rglob("*.py")
    ):
        source_digest.update(str(path.relative_to(args.source_root)).encode())
        source_digest.update(path.read_bytes())
    info["prims_source_sha256"] = source_digest.hexdigest()
    info["sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in {
            "driver": Path(__file__),
            "fixtures": Path(__file__).with_name("attention_ts_scaled_mla_fixtures.py"),
            "timing": Path(__file__).with_name("attention_ts_sparse_mla_timing.py"),
            "suite": args.suite,
        }.items()
    }
    if not args.without_flashmla:
        if args.flashmla_root:
            sys.path.insert(0, str(args.flashmla_root))
        import flash_mla

        package = Path(flash_mla.__file__).resolve().parent
        if args.flashmla_root and package != args.flashmla_root / "flash_mla":
            raise RuntimeError("FlashMLA did not import from --flashmla-root")
        revision = subprocess.run(
            ["git", "-C", str(package.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(package.parent),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            capture_output=True,
            text=True,
        )
        info["flashmla"] = dict(
            revision=revision.stdout.strip() if revision.returncode == 0 else None,
            source_dirty=bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            interface_sha256=hashlib.sha256(
                (package / "flash_mla_interface.py").read_bytes()
            ).hexdigest(),
            extension_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in package.glob("*.so")
            },
        )
    return info


def main(argv=None):
    """List CPU-only inventories or checkpoint a selected CUDA comparison campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "suites/scaled_mla/decode.json"
    )
    parser.add_argument(
        "--case", action="append", help="Repeat a stable ID from --list"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--flashmla-root", type=Path)
    parser.add_argument(
        "--without-flashmla",
        action="store_true",
        help="Explicit two-Prims-backend diagnostic run",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--samples-per-replay", type=int, default=4)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="One sample per replay; functional validation only",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        help="Optional reference/packed median target, e.g.1.2; no implicit performance gate",
    )
    args = parser.parse_args(argv)
    args.suite = args.suite.resolve()
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
    if args.replays <= 0 or args.replays % 12 or args.samples_per_replay <= 0:
        parser.error(
            "--replays must be a positive multiple of12; samples must be positive"
        )
    if args.min_speedup is not None and (
        not math.isfinite(args.min_speedup) or args.min_speedup <= 0
    ):
        parser.error("--min-speedup must be finite and positive")
    args.source_root = args.source_root.resolve()
    args.flashmla_root = args.flashmla_root.resolve() if args.flashmla_root else None
    if args.quick:
        args.samples_per_replay = 1
    report = dict(
        schema="scaled-mla-v1",
        status="running",
        cases=[],
        requested=[r["id"] for r in rows],
        quick=args.quick,
        without_flashmla=args.without_flashmla,
        timing=dict(
            replays=args.replays,
            samples_per_replay=args.samples_per_replay,
            cache="cold-4xL2",
            cuda_graph=True,
            timed="attention and required reductions",
            excluded="planning, allocation, conversion, adapters, head padding, output views, L2 eviction",
        ),
        numerical_contract="FP8 Q; FP8-MMA oracle for both Prims paths; exact packed-KV/BF16-MMA oracle for FlashMLA",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        """Atomically retain each completed row and any failing/incomplete campaign."""
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    try:
        report["provenance"] = setup_sources(args)
        for row in rows:
            try:
                report["cases"].append(benchmark_case(row, args))
            except Exception as error:
                report["cases"].append(
                    dict(
                        **row, status="failed", error=f"{type(error).__name__}: {error}"
                    )
                )
                raise
            save()
            print(row["id"], report["cases"][-1]["speedups"], flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["summary"] = summarize(
        report["cases"],
        report["requested"],
        without_flashmla=args.without_flashmla,
        min_speedup=args.min_speedup,
    )
    report["status"] = "passed" if report["summary"]["passed"] else "failed"
    save()
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
