# Native sparse MLA benchmark suite

Compare Prims-TS sparse MLA with FlashInfer's TRTLLM-Gen DSV4 backend on the same native BF16 or E4M3 tensors. Both prefill and decode use the public `BatchSparseMLADecodePagedTSWrapper.plan()` / `run()` API, Dqk = Dv = 512, and BF16 output. The suite supports two KV pools (SWA plus selected compressed KV), HCA, and a single selected pool without SWA. It is separate from the repository's dense MLA and QToken-KvBlock-Sparse-Attention suites.

## Setup

Use a FlashInfer source checkout containing [PR #5434](https://github.com/flashinfer-ai/flashinfer/pull/5434). The implementation used for the published results is `8ac751ee6a09d8a349a136192dc1ca48d848ac6d` in [PerkzZheng/flashinfer](https://github.com/PerkzZheng/flashinfer/tree/feat/prims-ts-sparse-mla). Install/build that checkout following FlashInfer's instructions, including its matching TRTLLM-Gen artifacts. An older wheel without this API is insufficient.

The qualification environment is GB300 (SM103, 152 SMs), CUDA 13.0, PyTorch 2.8.0a0 from the 25.08 NVIDIA PyTorch image, and `nvidia-cutlass-dsl==4.7.0`. The runner accepts SM100/SM103; the published sparse results qualify SM103 only. It also needs Triton, pytest, and `cuda-python` with `cuda.bindings.runtime`. It does not install dependencies or allocate GPUs.

```bash
export FLASHINFER_ROOT=/path/to/flashinfer
export VALIDATION_ROOT=/path/to/flashinfer-prims-ts-validation

# Inventory and CPU preflight: no CUDA imports or GPU required.
python "$VALIDATION_ROOT/scripts/bench_attention_ts_sparse_mla.py" --list
python -m pytest -q "$VALIDATION_ROOT/scripts/sparse_mla_cpu_checks.py"

# Small prefill/decode selection in both dtypes.
python "$VALIDATION_ROOT/scripts/bench_attention_ts_sparse_mla.py" \
  --source-root "$FLASHINFER_ROOT" --indices 0,1,30,31,238,239 \
  --output results/sparse-mla-selected.json

# Full 480-case matrix.
python "$VALIDATION_ROOT/scripts/bench_attention_ts_sparse_mla.py" \
  --source-root "$FLASHINFER_ROOT" --output results/sparse-mla-full.json

# High compression: selected count may be shorter than top-k capacity.
python "$VALIDATION_ROOT/scripts/bench_attention_ts_sparse_mla.py" \
  --source-root "$FLASHINFER_ROOT" --compression-ratio 128 \
  --indices 242,243 --output results/sparse-mla-hca.json

# One selected KV pool, as needed by GLM-style sparse attention.
python "$VALIDATION_ROOT/scripts/bench_attention_ts_sparse_mla.py" \
  --source-root "$FLASHINFER_ROOT" --no-swa \
  --indices 242,243 --output results/sparse-mla-no-swa.json
```

The runner imports the preparation and FP64 reference helpers directly from `tests/attention/test_prims_ts_sparse_mla.py` in `--source-root`. This preserves the reviewed preparation and accuracy contract without a second copy. It verifies the FlashInfer import location and records the helper hash. A future incompatible helper/API change requires updating this driver.

## Cases and inputs

| Phase | Cases | Batch | Query tokens | Query heads | Selected capacity | Dtype |
|---|---:|---|---|---|---|---|
| Prefill | 30 | 1 | 8192 | 8/16/32/64/128 | 512/1024/2048 | BF16/E4M3 |
| Decode | 450 | 1/4/16/64/256 | 1/4/8 | 8/16/32/64/128 | 512/1024/2048 | BF16/E4M3 |

Raw context defaults to 32,768. Stable IDs 0–29 are prefill and 30–479 decode; within each phase the nesting is batch, query length, selected capacity, heads, dtype. `--indices` keeps those IDs so subsets map back to prior campaign results. This is a synthetic model-shape matrix, not an end-to-end model benchmark.

Top-k generation follows the random-score selection approach in [FlashMLA's sparse decoding test](https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_sparse_decoding.py):

1. Generate uniform random scores over candidate compressed entries for each request/query, in chunks of 128 queries to bound memory.
2. For query `j`, mask candidates beyond `floor((raw_context - SQ + j + 1) / compression_ratio)` with negative infinity.
3. Select `topk(..., sorted=True)`, then map logical IDs through a random physical-page permutation. Sorting is by score, **not physical address**; compressed-KV rows remain random.
4. Where fewer candidates exist than the requested capacity, fill unused slots with `-1` and retain the actual length. Every row has a valid prefix, so Prims-TS can use `assume_valid_prefix=True`.

Compressed KV has page size one. SWA selects 128 consecutive logical tokens per query, stored in randomly permuted 256-token pages. Thus SWA has real within-page locality; the fixture does not manufacture contiguous runs in random compressed KV. `--no-swa` makes Prims-TS use one pool; TRT's fixed 128-slot SWA segment is invalid and points at a poisoned dummy pool. Any remaining cost of that TRT segment is part of the measured public operator.

The default seed is 2026. Q is normal-distributed and clipped to [-1, 1]; KV is scaled by 0.1 before clipping and native conversion. Q/KV descales are one, and QK softmax scale is `512**-0.55`. Sinks include finite values and ±infinity. Unselected cache rows are poisoned with NaNs to expose accidental reads. These choices preserve the existing campaign fixture; the suite does not claim to reproduce trained-model activation distributions. Packed FP8 is not supported or compared.

## Accuracy and timing

Both backends use the same actual Q, KV, indices, lengths, sinks and scale. Complete tensor SHA-256 fingerprints are computed outside timing. A chunked FP64 reference checks all output elements before capture, after initial replay, and after timing. BF16 uses `abs(reference) * 0.02 + 8e-4`; FP8 uses the existing reference's probability/output rounding bound. No tolerance is changed for the comparator.

Metadata preparation, planning, compilation and allocation are outside timing for both backends. Their public attention calls, split reductions, finishing and required counter resets remain inside. Preallocated buffers and callable owners stay alive throughout capture and replay.

Each captured sample is `evict -> start event -> attention -> stop event`. Eviction reads a separate nonconstant int32 buffer of four times device L2 via a reduction, before the timing event. Explicit external CUDA event nodes exclude eviction from the measured span. There is no eager or warm-cache timing fallback. Requalify eviction with cache counters and a larger-buffer control when moving to another GPU/timing implementation; buffer size alone is not proof of cache behavior.

Default timing collects 12 replays × 4 samples per backend, alternating backend order between replays. JSON includes raw microsecond samples, median, p10 and p90. Speedup is `TRT median / Prims-TS median`, so values above one favor Prims-TS. The gate is `(Prims-TS median / TRT median - 1) * 100 <= 5`. This is the sparse campaign's ratio-of-medians statistic; the older dense suite uses its separately documented sum-of-samples statistic.

A failed TRT accuracy/runtime check is N/A, not a Prims-TS win. A missing/failed comparison or a gap above `--max-slowdown-pct` (default 5) gives a nonzero exit status. Prims-TS failures stop execution after saving the failing case. Reports checkpoint after every case; incomplete runs cannot pass. Rerun a noisy outlier with a larger even `--replays` value, preserve both reports, and distinguish variance from a repeatable gap.

## Results and provenance

Each report records the actual FlashInfer commit and dirty state, benchmark/helper hashes, GPU model/SM count and hashed device identity, CUDA runtime/driver, PyTorch, Triton, DSL and cubin package versions, inputs, timing controls and gate summary. Normal provenance excludes hostnames, job IDs and absolute paths. Inspect exception text before publishing failed reports because third-party errors may contain paths.

To compare FlashInfer revisions, run the same command on the same idle GPU with different `--source-root` checkouts and output files. Match case sets, fingerprints, hardware identity, software/artifact versions and timing parameters before interpreting a difference. The repository's dense `compare_runs.py` schema is not used by this standalone suite. Keep generated campaign reports under `results/`; no old benchmark outputs or debugging/policy-override tools are bundled here.

The PR's latest incremental refresh covered 227 cases, including 211 paired standalone-reducer cases. Those 211 had a 1.34× geometric-mean speedup over TRT and a worst +1.56% latency gap; all 227 were within 5%. That checkpoint predates this packaging and is not a fresh full 480-case run of the published driver.

Packaging validation on GB300 passed seven CPU checks and Prims-TS accuracy/graph checks for ten configurations: mixed-pool IDs 0/1/30/31/238/239, HCA IDs 242/243, and no-SWA IDs 242/243. All six mixed-pool input fingerprints matched the archived campaign. Nine TRT comparisons passed accuracy and the 5% gate. BF16 no-SWA case 242 failed TRT's accuracy check again on a separate invocation; its speedup remains N/A and that selection exits nonzero. The functional run used four replays per backend and is not full performance signoff.
