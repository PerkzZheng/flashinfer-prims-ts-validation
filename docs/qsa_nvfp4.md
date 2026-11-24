# QSA NVFP4-KV performance suites

These standalone suites compare NVFP4-KV with FP8-KV or BF16-KV Prims-TS QToken-KvBlock-Sparse-Attention (QSA) through the public `QTokenKvBlockSparsePagedTSWrapper`. They target D256 at a 128K context with two head geometries, 64 query heads over 16 or 4 KV heads, and are separate from the existing dense, MLA and vLLM-Triton sparse-attention campaigns.

| Suite | Cases | Workload |
|---|---:|---|
| `decode.json` | 54 | TP1/2/4 × BS 1/4/8/16/32/64 × MTP 3/4/7 (G = MTP + 1 query tokens per request) |
| `prefill.json` | 12 | TP1/2/4 × one request with its last 8K/16K of 128K tokens as queries × G 4/8, fixed-Q layout |
| `decode_hkv4.json`, `prefill_hkv4.json` | 54, 12 | The same grids with 64/4 heads |

Each manifest names its model heads (`model_heads`); TP is a single-GPU split of both Q and KV heads: Hq/Hkv = 64/16, 32/8 and 16/4, or 64/4, 32/2 and 16/1. No collective is measured.

By default (`--comparison fp8`), every row runs three backends on matched values:

| Backend | Q | K/V storage | MMA |
|---|---|---|---|
| `nvfp4-fp8mma` (primary) | E4M3 | interleaved NVFP4 rows | FP8 |
| `nvfp4-bf16mma` | BF16 | the same NVFP4 rows | BF16 |
| `fp8` | E4M3 | plain E4M3 | FP8 |

An interleaved NVFP4 row is `uint8 [D/2 + D/16]`: 128 payload bytes (E2M1, even element in the low nibble) followed by the token's sixteen E4M3 scales, one per 16 channels, with no global scale. The FP8 cache holds the E4M3 rounding of exactly the dequantized NVFP4 values, and FP8 Q is the E4M3 rounding of the BF16 Q, so all three backends see the same logical inputs. Prims-TS selects its kernel recipe itself; the harness does not force a route.

## BF16-KV comparison

`--comparison bf16` runs the same suites and workload with BF16 Q against a plain BF16-KV baseline instead. The default `--comparison fp8` is the matrix above and is unchanged.

| Backend | Q | K/V storage | MMA |
|---|---|---|---|
| `nvfp4-bf16mma` (primary) | BF16 | interleaved NVFP4 rows | BF16 |
| `nvfp4-bf16mma-separate` | BF16 | the same values as separate payloads plus `kv_scale_factors` | BF16 |
| `bf16` | BF16 | plain BF16 | BF16 |

The separate-scales layout is the TRT-LLM-style NVFP4 cache: K and V payloads are `uint8 [..., D/2]` (the rows' first 128 bytes) and the wrapper takes the block scales as `kv_scale_factors=(K scales, V scales)`, E4M3 `[..., D/16]` each. K scales stay in natural token order. V scales use TRT-LLM's four-token interleave, as FlashInfer's own tests build it: with `n = D/64` scales per quarter row, the scale of token `4g + t`, block `b` sits at token `4g + b // n`, slot `(b % n) * 4 + t`. The BF16 cache holds the dequantized NVFP4 values; every E2M1 × E4M3 product is representable in BF16, so that conversion is exact and all three backends see identical K/V values and the same BF16 Q.

Speedups are `backend median / nvfp4-bf16mma median`: `bf16` above one favors NVFP4 rows over BF16 K/V, and `nvfp4-bf16mma-separate` above one favors the rows layout over separate scales. `--min-speedup` applies to the BF16-KV ratio. All three backends use the 0.01 relative L2 limit against the BF16-Q oracle. Byte accounting counts BF16 K/V as four bytes per channel (K and V), and both NVFP4 layouts as payload plus scales. Reports record the comparison, and `summarize_qsa_nvfp4.py` renders either one but refuses to mix them.

The comparison keeps the NVFP4 rows, the separate copy and the BF16 cache resident and drops the FP8 pool. The BS64 TP1 decode rows at 64/16 allocate about 215 GB of caches (BF16 K/V alone is about 137 GB), which fit on one GB300; the 64/4 rows need a quarter of that.

## Workload

Each request gets 512 selected blocks of four tokens per query token plus the exact causal tail of its current block, over 16-token physical pages with a shuffled block table. Query token 0 draws distinct uniform blocks from its causal range, as FlashMLA's random top-k tests do. Each following token keeps a random 70% of its predecessor's blocks and draws the rest fresh, so overlap decays geometrically with distance. Indices come from a per-request NumPy generator seeded by `(seed, phase, request)`. The cache repeats one quantized random tile, which keeps allocation fast; the shuffled table still scatters every request. This is a synthetic workload, not model-quality qualification.

## Setup and execution

The suites need a FlashInfer checkout with NVFP4-KV support in QSA, which upstream main has since [#9375](https://github.com/flashinfer-ai/flashinfer/pull/9375). The recorded validation predates that merge: it used commit `39b53eb0d9` for the FP8-KV comparison and `98cd63fac` for the BF16-KV comparison (see the validation section below), earlier revisions of that PR's branch `qsa-nvfp4-kv-qtoken` of PerkzZheng/flashinfer. The branch has since been rebased, so fetch them by full hash: `git fetch https://github.com/PerkzZheng/flashinfer.git 39b53eb0d9aeea1cb6d89157a70b21e40aee8f35 98cd63facf29a30dc1a00f51ef66b94963616358`. A `--quick` run of `decode-tp1-b32-mtp4` in both comparisons also passed against main `a7ed35743` on GB300; that is functional evidence only. The harness records the actual revision, branch, dirty state and a hash of the Prims-TS sources for every run. It does not install packages or allocate hardware.

```bash
export FLASHINFER_ROOT=/path/to/flashinfer
export VALIDATION_ROOT=/path/to/flashinfer-prims-ts-validation

# CPU-only inventory, workload and accounting checks (Torch optional).
python "$VALIDATION_ROOT/scripts/bench_q_token_kv_block_sparse_nvfp4.py" --list
python -m pytest -q "$VALIDATION_ROOT/scripts/qsa_nvfp4_cpu_checks.py"

# Functional smoke; --quick only shortens timing.
python "$VALIDATION_ROOT/scripts/bench_q_token_kv_block_sparse_nvfp4.py" \
  --source-root "$FLASHINFER_ROOT" --case decode-tp1-b32-mtp4 \
  --quick --output results/qsa-nvfp4-smoke.json

# Full decode and prefill matrices.
python "$VALIDATION_ROOT/scripts/bench_q_token_kv_block_sparse_nvfp4.py" \
  --source-root "$FLASHINFER_ROOT" --output results/qsa-nvfp4-decode.json
python "$VALIDATION_ROOT/scripts/bench_q_token_kv_block_sparse_nvfp4.py" \
  --suite "$VALIDATION_ROOT/suites/qsa_nvfp4/prefill.json" \
  --source-root "$FLASHINFER_ROOT" --output results/qsa-nvfp4-prefill.json

# The 64/4 geometry.
python "$VALIDATION_ROOT/scripts/bench_q_token_kv_block_sparse_nvfp4.py" \
  --suite "$VALIDATION_ROOT/suites/qsa_nvfp4/decode_hkv4.json" \
  --source-root "$FLASHINFER_ROOT" --output results/qsa-nvfp4-decode-hkv4.json

# Latency, throughput and speedup table; accepts several (partial) reports.
python "$VALIDATION_ROOT/scripts/summarize_qsa_nvfp4.py" \
  results/qsa-nvfp4-decode.json results/qsa-nvfp4-prefill.json \
  --output results/qsa-nvfp4.md
```

Repeat `--case` for a subset; the campaign can be split across several output files and summarized together. Use a fresh output filename; existing results are never overwritten. A failed backend or incomplete campaign exits nonzero. The BS64 TP1 decode rows allocate about 110 GB of matched NVFP4 and FP8 caches.

The BF16-KV comparison takes the same suites and options. With `--nvtx` it also marks each backend's timed replays for Nsight Systems; `qsa_nvfp4_nsys_spans.py` needs `nsys` on `PATH` (or `--nsys`):

```bash
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o results/qsa-bf16-decode \
  python "$VALIDATION_ROOT/scripts/bench_q_token_kv_block_sparse_nvfp4.py" \
  --source-root "$FLASHINFER_ROOT" --comparison bf16 --nvtx \
  --output results/qsa-bf16-decode.json

# CUDA-event medians, as for the FP8-KV comparison.
python "$VALIDATION_ROOT/scripts/summarize_qsa_nvfp4.py" \
  results/qsa-bf16-decode.json --output results/qsa-bf16-decode.md

# Per-call GPU spans; several traces (e.g. per-TP files) are pooled.
python "$VALIDATION_ROOT/scripts/qsa_nvfp4_nsys_spans.py" \
  results/qsa-bf16-decode.nsys-rep --output results/qsa-bf16-decode-spans.json
```

## Accuracy and fair timing

An FP32 oracle attends the dequantized NVFP4 values on eight sampled query tokens (the first and last four) of every row. BF16-MMA is checked against the BF16 Q, both FP8 paths against the E4M3 Q. Relative L2 error must stay below 0.01 for BF16-MMA and 0.06 for the FP8 paths, which also round P (and, for FP8 KV, the cache) to E4M3. Each backend is checked eagerly, after a poisoned-output graph replay and after timing. Tolerances are not adjusted per case.

Timing reuses `ColdL2GraphBenchmark` from the native sparse MLA suite: each sample is `4×L2 nonconstant-buffer eviction → start event → complete wrapper call → stop event`, with external event nodes keeping eviction outside the measured span. The call includes the prepared QSA metadata kernel, attention and any split-KV reduction. Index generation, planning, allocation, quantization and conversion are outside timing for all backends.

Defaults are 12 replays × 4 samples per backend; backend order rotates and reverses (replays must be a multiple of 6). Results retain all microsecond samples, median, p10/p90, achieved TB/s and TFLOP/s, and speedups defined as `reference median / nvfp4-fp8mma median`; values above one favor NVFP4. There is no implicit performance gate. `--min-speedup 1.3` optionally requires that ratio against FP8 KV on every row. `--quick` is functional evidence only.

Throughput uses one accounting for all backends. Bytes are the unique needed K/V once per request (NVFP4: payload plus scales; FP8: E4M3 K/V; BF16: BF16 K/V), plus Q and BF16 O. Grouping overfetch is excluded. FLOPs are `4 × D × Hq × attended tokens`. These are workload metrics, not measured HBM transactions.

CUDA event timestamps on the qualified GB300 advance in steps of about 2 µs, so latencies of the smallest rows cluster on that grid. Compare small-batch rows with that resolution in mind, or use more samples.

For finer comparisons, `--nvtx` wraps each backend's timed replays in an NVTX range `QSA|<case>|<backend>`; it does not change what runs inside the graphs. Under `nsys profile -t cuda,nvtx --cuda-graph-trace=node`, every graph node appears as its own kernel with nanosecond timestamps. `qsa_nvfp4_nsys_spans.py` exports the kernels and projected NVTX ranges with `nsys stats`, and takes each call's span from the start of the QSA metadata kernel through the end of the last kernel before the next eviction kernel (or metadata kernel): metadata, attention and any split-KV reduction, with eviction outside, as in the event timing. It reports mean and median span per case and backend, the metadata kernel's mean time, call counts, speedups and their geomean. Speedups use mean spans, because small-batch spans can be bimodal and a median then jumps between the modes. Spans leave out the gaps around the event nodes; in the run below they sat 2–6 µs below the event medians of the same replays, so compare spans with spans.

## Validation

### FP8-KV comparison

All 132 rows (54 decode and 12 prefill per geometry) passed on one GB300 with PyTorch 26.08 (2.14.0a0), CUDA 13.4 and CuTe DSL 4.8.1, against FlashInfer `39b53eb0d9` (branch `qsa-nvfp4-kv-qtoken`, clean tree, based on upstream main `381960f63`), with default timing controls. Each decode matrix ran as three per-TP output files and was summarized together. The worst relative L2 error over every row and check point was 0.040 for both FP8 paths and 0.0035 for BF16-MMA.

Speedups over FP8 KV in that run, `nvfp4-fp8mma` / `nvfp4-bf16mma`, 64/16:

| Decode | TP1 | TP2 | TP4 |
|---|---|---|---|
| MTP3, BS 16–64 | 1.23–1.39× / 0.79–0.83× | 1.19–1.26× / 0.80–0.84× | 1.14–1.22× / 0.83–0.87× |
| MTP4/7, BS 16–64 | 1.20–1.33× / 0.63–0.67× | 1.14–1.24× / 0.62–0.69× | 1.16–1.24× / 0.62–0.84× |
| MTP3, BS 1–8 | 1.00–1.17× / 0.82–0.89× | 0.90–1.11× / 0.85–0.89× | 0.94–1.13× / 0.89–0.94× |
| MTP4/7, BS 1–8 | 1.10–1.17× / 0.67–0.91× | 1.06–1.21× / 0.81–0.96× | 1.06–1.23× / 0.87–0.94× |

64/4:

| Decode | TP1 | TP2 | TP4 |
|---|---|---|---|
| MTP3, BS 16–64 | 1.15–1.19× / 0.88–1.00× | 1.12–1.15× / 0.88–1.00× | 1.08–1.18× / 1.00–1.05× |
| MTP4/7, BS 16–64 | 0.86–0.96× / 0.67–0.80× | 0.85–0.95× / 0.71–0.86× | 0.89–1.06× / 0.80–0.99× |
| MTP3, BS 1–8 | 1.03–1.06× / 1.00–1.05× | 1.04–1.06× / 0.94–1.04× | 1.00× / 0.94–1.00× |
| MTP4/7, BS 1–8 | 0.92–1.00× / 0.87–1.00× | 0.95–1.06× / 0.92–0.96× | 0.95–1.00× / 0.94–1.00× |

At 64/16 MTP4/7 (G ≥ 5), FP8-Q Prims-TS runs the FP8 Keeps recipe fed by an NVFP4-to-FP8 transform, persistent, in one wave or split along the KV sequence with a separate reduction; MTP3 (G4) and every BF16-Q row there use the TMEM-transformed recipe. TP1 BS64 measured 1.39×, 1.30× and 1.33× at MTP3/4/7. At 64/4 every group exceeds a 32-row Swaps Q tile, so all rows run the Keeps recipe of Q's dtype (Q64 at G4, Q128 at G5..G8) fed by a transform to E4M3 or exact BF16 tiles. There FP8 KV is only partly memory-bound, the transform is not fully hidden, and the FP8-KV Keeps recipe reads per-query membership planes that the NVFP4-fed recipe does not use yet: FP8-MMA MTP4/7 is 0.85–1.06× (TP1 BS32 MTP7: 104.4 µs against 90.1 µs). Absolute latencies of one revision differ by a few percent between GB300 nodes.

Prefill rows were 0.84–1.06× (BF16-MMA 0.38–0.56×) at 64/16 and 0.75–0.89× (BF16-MMA 0.57–0.71×) at 64/4. These figures describe one run of one revision; they are not expected-result assertions.

### BF16-KV comparison

All 108 decode rows (18 per TP and geometry) passed with `--comparison bf16 --nvtx` under `nsys profile -t cuda,nvtx --cuda-graph-trace=node`, on GB300 with PyTorch 26.08 (2.14.0a0), CUDA 13.4 and CuTe DSL 4.8.1, against FlashInfer `98cd63fac` (branch `qsa-nvfp4-kv-qtoken` of PerkzZheng/flashinfer, clean tree, three commits on upstream main `381960f63`), with default timing controls. Each TP of each geometry ran as its own process and output file, four at a time on the four GPUs of one node; each geometry's three traces were pooled by `qsa_nvfp4_nsys_spans.py` (48 calls per case and backend). The worst relative L2 error over every row and check point was 0.0032 at 64/16 and 0.0035 at 64/4, the same for all three backends. Prefill was not run in this comparison.

Span-based speedups in that run, NVFP4 rows over BF16 KV / separate scales over rows (above one favors rows), 64/16:

| Decode | TP1 | TP2 | TP4 |
|---|---|---|---|
| MTP3, BS 16–64 | 1.55–1.77× / 1.04–1.05× | 1.51–1.63× / 1.03–1.05× | 1.35–1.57× / 1.03–1.05× |
| MTP4/7, BS 16–64 | 1.80–1.93× / 1.05× | 1.57–1.87× / 1.03–1.05× | 1.54–1.79× / 1.03–1.06× |
| MTP3, BS 1–8 | 1.14–1.50× / 1.00–1.03× | 1.03–1.35× / 1.00–1.03× | 1.02–1.24× / 1.00–1.02× |
| MTP4/7, BS 1–8 | 1.47–1.60× / 1.00–1.05× | 1.35–1.56× / 1.00–1.05× | 1.31–1.50× / 0.99–1.04× |

64/4:

| Decode | TP1 | TP2 | TP4 |
|---|---|---|---|
| MTP3, BS 16–64 | 1.43–1.58× / 1.03–1.04× | 1.32–1.52× / 1.02–1.03× | 1.18–1.43× / 1.00–1.04× |
| MTP4/7, BS 16–64 | 1.10–1.13× / 1.09–1.18× | 1.10–1.13× / 1.07–1.16× | 1.11–1.14× / 1.06–1.11× |
| MTP3, BS 1–8 | 1.06–1.31× / 0.99–1.02× | 1.02–1.17× / 1.00× | 1.01–1.07× / 0.98–1.00× |
| MTP4/7, BS 1–8 | 1.02–1.13× / 1.00–1.07× | 0.98–1.13× / 0.98–1.05× | 0.99–1.07× / 0.99–1.03× |

Geomeans over the 54 rows of each geometry were 1.52× over BF16 KV and 1.03× for separate scales at 64/16 (TP1/2/4: 1.62×, 1.52×, 1.44× over BF16), and 1.14× and 1.05× at 64/4 (1.17×, 1.13×, 1.10×). The CUDA-event medians of the same replays give 1.48× / 1.03× and 1.12× / 1.05×, and differ from the span speedups by at most 0.1 on any row.

At 64/16, TP1 BS64 spans were 330.9, 381.2 and 519.9 µs for NVFP4 rows against 586.8, 731.5 and 1003.0 µs for BF16 KV at MTP3/4/7 (1.77×, 1.92×, 1.93×). The smallest rows gain least: one-request MTP3 was 1.02–1.15× (TP4: 23.0 µs against 23.5 µs), where the 8.5–11.6 µs metadata kernel is a large part of every span. At 64/4 with MTP4/7, the NVFP4 layouts read about a third of BF16 KV's workload bytes yet gain only 1.10–1.14× at BS 16–64: at TP1 BS32/64, BF16 KV moves 5.6–6.3 TB/s of workload bytes against 1.8–2.1 TB/s for NVFP4 rows, so the NVFP4 paths there are bound by their own transform and MMA work rather than by K/V traffic. The separate-scales layout is 6–18% slower than rows in that band and falls behind plain BF16 KV in its six largest rows (TP1 BS32/64, TP2 BS64; TP1 BS64 MTP4: 171.9 µs rows, 202.2 µs separate, 193.7 µs BF16). The only rows below parity were three one-request 64/4 rows (TP2 MTP4, TP4 MTP4/7) at 0.98–0.99× (TP2 MTP4: 25.8 µs against 25.3 µs). These figures also describe one run of one revision and are not expected-result assertions.

The eighteen CPU checks (inventory of both geometries, case selection, invalid timings, failure accounting, index generation, byte accounting, summary rendering, the NVFP4 round trip, and for the BF16-KV comparison its backend specs, BF16 byte accounting, summary gating and rendering, span extraction and the separate-scale layout) pass, as do Ruff lint/format checks.

## Comparing revisions

Run the same selected cases on baseline and candidate checkouts with identical hardware, software, manifests and timing controls. Compare the recorded Prims-TS source hash and benchmark hashes before comparing absolute latencies. Historical speedups are not expected-result assertions. Generated JSON belongs under `results/`; inspect failure text for local paths before publishing artifacts. The dense `compare_runs.py` and the vLLM-Triton sparse summaries use different schemas.
