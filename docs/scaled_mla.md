# Scaled-KV MLA performance suites

These standalone suites compare packed NVFP4 Prims-TS, plain-FP8 Prims-TS and FlashMLA. They use the public `BatchSparseMLADecodePagedTSWrapper` API from [FlashInfer PR #7410](https://github.com/flashinfer-ai/flashinfer/pull/7410), validated starting at `d2f937beb813c8c28d6cb1a2a32720f4200d0aa9`. They are separate from the existing dense and native sparse MLA campaigns; their rows do not change the historical 564/480-row totals.

| Suite | Cases | Batch | Query length | Heads |
|---|---:|---|---|---|
| `decode.json` | 48 | 4/16/32/64 | 4/8 | 64/128 |
| `small_heads.json` | 72 | 4/16/32/64 | 4/8 | 8/16/32 |

Both suites cross every shape with the following three recipes. Every row uses FP8 E4M3 Q, 64K context tokens per request, 64-token physical pages, top-k 2048 and BF16 output. D512 adds a 128-token SWA window per query. Query and sparse-list generation are deterministic, with shuffled physical pages and distinct, randomly ordered top-k entries. Sparse lists define visibility; there is no additional causal mask. The seed and complete input fingerprints are saved in each result. This is a synthetic workload, not model-quality qualification.

| Recipe | Primary pool | Extra pool | FlashMLA comparison |
|---|---|---|---|
| `d576` | 352B: NVFP4 NoPE + FP8 RoPE | None | Native 352B cache |
| `d512-nvfp4-extra` | 288B NVFP4, including rotary channels | 288B NVFP4 | N/A: its D512 API does not accept NVFP4 primary storage |
| `d512-fp8-extra` | 512B per-tensor FP8, scale 0.5 | 288B NVFP4 | Explicit 528B constant-scale MXFP8 adapter |

NVFP4 uses E2M1 values with E4M3 scales per 16 channels and no global scale. D576 stores 256 payload bytes, 64 unscaled E4M3 RoPE bytes, then 32 scales in FlashMLA V32 order. D512 stores 256 payload bytes followed by 32 linear scale bytes per token. The per-tensor FP8 adapter repeats the same 0.5 scale in sixteen UE8M0 bytes; its values exactly equal the primary FP8 values supplied to Prims-TS. It is labeled as an adapter, not claimed as FlashMLA support for the 512B storage format.

## Setup and execution

Use an SM100/SM103 environment compatible with the selected FlashInfer checkout, Torch, Triton and `cuda-python`. Build/install FlashMLA with both the 352B D576 and token-interleaved D512 528B/288B layouts; the original comparison used [vllm-project/FlashMLA](https://github.com/vllm-project/FlashMLA) at `e2afa5f`. This driver verifies import roots and records the actual commits, compiled FlashMLA extension hashes, GPU/software identity and benchmark hashes. It does not install packages, patch either implementation or allocate hardware.

```bash
export FLASHINFER_ROOT=/path/to/flashinfer
export FLASHMLA_ROOT=/path/to/FlashMLA
export VALIDATION_ROOT=/path/to/flashinfer-prims-ts-validation

# CPU-only inventory and accounting checks.
python "$VALIDATION_ROOT/scripts/bench_attention_ts_scaled_mla.py" --list
python -m pytest -q "$VALIDATION_ROOT/scripts/scaled_mla_cpu_checks.py"
# Hand-encoded nibble/scale and rounding checks: PyTorch required, no GPU.
python -m pytest -q "$VALIDATION_ROOT/scripts/scaled_mla_fixture_checks.py"

# Functional smoke with production geometry; --quick only shortens timing.
python "$VALIDATION_ROOT/scripts/bench_attention_ts_scaled_mla.py" \
  --source-root "$FLASHINFER_ROOT" --flashmla-root "$FLASHMLA_ROOT" \
  --case d576-b4-q4-h64 --case d512-fp8-extra-b4-q4-h128 \
  --quick --output results/scaled-mla-smoke.json

# Full H64/H128 matrix, then the small-head matrix.
python "$VALIDATION_ROOT/scripts/bench_attention_ts_scaled_mla.py" \
  --source-root "$FLASHINFER_ROOT" --flashmla-root "$FLASHMLA_ROOT" \
  --output results/scaled-mla-decode.json
python "$VALIDATION_ROOT/scripts/bench_attention_ts_scaled_mla.py" \
  --suite "$VALIDATION_ROOT/suites/scaled_mla/small_heads.json" \
  --source-root "$FLASHINFER_ROOT" --flashmla-root "$FLASHMLA_ROOT" \
  --output results/scaled-mla-small-heads.json

# Absolute latencies and speedups as a Markdown table.
python "$VALIDATION_ROOT/scripts/summarize_scaled_mla.py" \
  results/scaled-mla-decode.json --output results/scaled-mla-decode.md
```

Repeat `--case` for a subset. Use a fresh output filename; existing results are never overwritten. `--without-flashmla` explicitly runs only the two Prims-TS storage paths and labels the omitted comparison. Unsupported, omitted and failed comparators are distinct; none receives a speedup. A requested comparator failure or incomplete campaign exits nonzero.

## Accuracy and fair timing

Both Prims-TS paths receive the same FP8 Q and the same effective E4M3 KV operands: the plain-FP8 cache is expanded from the packed bytes outside timing. FlashMLA receives exact BF16 widenings of that FP8 Q and the same packed KV bytes (or the explicitly labeled primary adapter). Its BF16 MMA differs from the FP8 MMA of the Prims paths, so the oracle models each backend's operand rounding explicitly. This is a comparison of the two compute recipes, not a claim that their MMA arithmetic is identical.

An independent FP64 reference checks every output element and natural-log LSE. FP8 uses the componentwise E4M3-probability forward-error bound from the sparse MLA validation work, with LSE `atol=2e-4, rtol=1e-4`. FlashMLA uses its BF16 output/LSE bounds and cosine-difference limit `5e-6`. All backends are checked eagerly, after poisoned-output graph replay, after timing, and after changing live queries and sparse addresses without recapture. Tolerances are not adjusted per case.

The shared `ColdL2GraphBenchmark` captures `4×L2 nonconstant-buffer eviction → start event → complete attention call → stop event`. Explicit external event nodes exclude eviction from timing. The measured call includes attention, pool merging and required split reductions. Planning, compilation, allocation, FP8 expansion, Q widening, the constant-scale adapter, small-head padding and output extraction are outside timing for all backends. There is no eager or hot-cache fallback. The graph has one complete attention invocation per measured sample, with a separate eviction immediately before each sample.

FlashMLA runs H8/H16/H32 workloads padded to physical H64. Padding is untimed; the native kernel's work on the padded heads remains measured and `physical_heads` reports it. D576 H128 may invoke the legacy H64 implementation twice inside FlashMLA's public call; both launches and required finishing are included. The harness does not replace public automatic dispatch with a tuned shape-specific policy.

Defaults are 12 replays × 4 samples per backend. Backend order rotates and reverses; a multiple of 12 replays balances both two- and three-backend rows. Results retain all microsecond samples, median, p10/p90 and speedups defined as `reference median / packed-NVFP4 median`; values above one favor packed Prims-TS. There is no implicit performance pass threshold. `--min-speedup 1.2` optionally requires 1.2× against every available requested comparator, including plain FP8. `--quick` is functional evidence only. Recheck outliers with more samples on the same idle GPU, and retain both reports.

The eviction implementation is reused from the native sparse suite. Validate cache-counter behavior and a larger-buffer control when moving to another GPU/timer; allocating 4×L2 alone does not prove cache state. These event-timed measurements should not be merged directly with historical CUPTI timings.

## Comparing revisions

Run the same selected cases on baseline and candidate checkouts with identical hardware, software, manifests and timing controls. Verify fingerprints and benchmark/reference hashes before comparing their absolute latencies. Historical speedups are not embedded as expected results. Generated JSON belongs under `results/`; inspect failure text for local paths before publishing artifacts. The dense `compare_runs.py` schema does not apply to this standalone report.

## Packaging validation

The added driver passed 18 selected production-geometry cases on GB300 with PyTorch 26.08, CUDA 13.4 and CuTe DSL 4.8.1: every recipe at B4/SQ4/H8/16/32/64/128, plus every recipe at B64/SQ8/H128. This covers 36 Prims-TS storage-path runs and 12 supported FlashMLA runs, each with 48 measured samples and all four accuracy phases passing. The two suite selections ran on separate GPUs; these checks qualify the harness and do not establish full 120-row performance signoff or cross-device latency comparisons.

Nine CPU inventory/failure-accounting checks and three CPU tensor packing/rounding checks passed. Both packed layouts additionally matched the pinned FlashMLA quantizer byte-for-byte, and decoded to identical values, on random and midpoint-tie inputs. The existing unified preflight also passed. A full performance decision requires a fresh complete run on the target environment; no historical speedup is an expected-result assertion.
