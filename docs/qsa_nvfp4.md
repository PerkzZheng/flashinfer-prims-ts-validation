# QSA NVFP4-KV performance suites

These standalone suites compare NVFP4-KV and FP8-KV Prims-TS QToken-KvBlock-Sparse-Attention (QSA) through the public `QTokenKvBlockSparsePagedTSWrapper`. They target D256 at a 128K context with two head geometries, 64 query heads over 16 or 4 KV heads, and are separate from the existing dense, MLA and vLLM-Triton sparse-attention campaigns.

| Suite | Cases | Workload |
|---|---:|---|
| `decode.json` | 54 | TP1/2/4 × BS 1/4/8/16/32/64 × MTP 3/4/7 (G = MTP + 1 query tokens per request) |
| `prefill.json` | 12 | TP1/2/4 × one request with its last 8K/16K of 128K tokens as queries × G 4/8, fixed-Q layout |
| `decode_hkv4.json`, `prefill_hkv4.json` | 54, 12 | The same grids with 64/4 heads |

Each manifest names its model heads (`model_heads`); TP is a single-GPU split of both Q and KV heads: Hq/Hkv = 64/16, 32/8 and 16/4, or 64/4, 32/2 and 16/1. No collective is measured.

Every row runs three backends on matched values:

| Backend | Q | K/V storage | MMA |
|---|---|---|---|
| `nvfp4-fp8mma` (primary) | E4M3 | interleaved NVFP4 rows | FP8 |
| `nvfp4-bf16mma` | BF16 | the same NVFP4 rows | BF16 |
| `fp8` | E4M3 | plain E4M3 | FP8 |

Above 32 Q rows per KV head (G × Hq/Hkv), FlashInfer serves NVFP4 K/V only with FP8 Q, so `nvfp4-bf16mma` is reported as `unsupported` on every 64/4 row and the row requires the other two backends.

An interleaved NVFP4 row is `uint8 [D/2 + D/16]`: 128 payload bytes (E2M1, even element in the low nibble) followed by the token's sixteen E4M3 scales, one per 16 channels, with no global scale. The FP8 cache holds the E4M3 rounding of exactly the dequantized NVFP4 values, and FP8 Q is the E4M3 rounding of the BF16 Q, so all three backends see the same logical inputs. Prims-TS selects its kernel recipe itself; the harness does not force a route.

## Workload

Each request gets 512 selected blocks of four tokens per query token plus the exact causal tail of its current block, over 16-token physical pages with a shuffled block table. Query token 0 draws distinct uniform blocks from its causal range, as FlashMLA's random top-k tests do. Each following token keeps a random 70% of its predecessor's blocks and draws the rest fresh, so overlap decays geometrically with distance. Indices come from a per-request NumPy generator seeded by `(seed, phase, request)`. The cache repeats one quantized random tile, which keeps allocation fast; the shuffled table still scatters every request. This is a synthetic workload, not model-quality qualification.

## Setup and execution

The suites need a FlashInfer checkout with NVFP4-KV support in QSA. The recorded validation used branch `qsa-nvfp4-kv-public` of [PerkzZheng/flashinfer](https://github.com/PerkzZheng/flashinfer/tree/qsa-nvfp4-kv-public) at commit `9006b820db` (see the validation section below); the harness records the actual revision, branch, dirty state and a hash of the Prims-TS sources for every run. It does not install packages or allocate hardware.

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

## Accuracy and fair timing

An FP32 oracle attends the dequantized NVFP4 values on eight sampled query tokens (the first and last four) of every row. BF16-MMA is checked against the BF16 Q, both FP8 paths against the E4M3 Q. Relative L2 error must stay below 0.01 for BF16-MMA and 0.06 for the FP8 paths, which also round P (and, for FP8 KV, the cache) to E4M3. Each backend is checked eagerly, after a poisoned-output graph replay and after timing. Tolerances are not adjusted per case.

Timing reuses `ColdL2GraphBenchmark` from the native sparse MLA suite: each sample is `4×L2 nonconstant-buffer eviction → start event → complete wrapper call → stop event`, with external event nodes keeping eviction outside the measured span. The call includes the prepared QSA metadata kernel, attention and any split-KV reduction. Index generation, planning, allocation, quantization and conversion are outside timing for all backends.

Defaults are 12 replays × 4 samples per backend; backend order rotates and reverses (replays must be a multiple of 6). Results retain all microsecond samples, median, p10/p90, achieved TB/s and TFLOP/s, and speedups defined as `reference median / nvfp4-fp8mma median`; values above one favor NVFP4. There is no implicit performance gate. `--min-speedup 1.3` optionally requires that ratio against FP8 KV on every row. `--quick` is functional evidence only.

Throughput uses one accounting for all backends. Bytes are the unique needed K/V once per request (NVFP4: payload plus scales; FP8: E4M3 K/V), plus Q and BF16 O. Grouping overfetch is excluded. FLOPs are `4 × D × Hq × attended tokens`. These are workload metrics, not measured HBM transactions.

CUDA event timestamps on the qualified GB300 advance in steps of about 2 µs, so latencies of the smallest rows cluster on that grid. Compare small-batch rows with that resolution in mind, or use more samples.

## Validation

All 132 rows (54 decode and 12 prefill per geometry) passed on one GB300 with PyTorch 26.08 (2.14.0a0), CUDA 13.4 and CuTe DSL 4.8.1, against FlashInfer `9006b820db` (branch `qsa-nvfp4-kv-public`, clean tree), with default timing controls. Each decode matrix ran as three per-TP output files and was summarized together. The worst relative L2 error over every row and check point was 0.040 for both FP8 paths and 0.0031 for BF16-MMA.

Speedups of `nvfp4-fp8mma` over FP8 KV in that run, 64/16:

| Decode | TP1 | TP2 | TP4 |
|---|---|---|---|
| MTP3, BS 16–64 | 1.23–1.38× | 1.21–1.28× | 1.13–1.22× |
| MTP4/7, BS 16–64 | 1.20–1.33× | 1.14–1.26× | 1.13–1.22× |
| MTP3, BS 1–8 | 0.94–1.17× | 0.93–1.11× | 0.90–1.11× |
| MTP4/7, BS 1–8 | 1.10–1.22× | 1.12–1.25× | 1.07–1.19× |

64/4:

| Decode | TP1 | TP2 | TP4 |
|---|---|---|---|
| MTP3, BS 16–64 | 1.14–1.19× | 1.17–1.23× | 1.07–1.18× |
| MTP4/7, BS 16–64 | 0.90–0.96× | 0.92–1.03× | 0.95–1.03× |
| MTP3, BS 1–8 | 1.00–1.12× | 1.04–1.07× | 1.00–1.07× |
| MTP4/7, BS 1–8 | 0.89–1.00× | 0.99–1.07× | 0.96–1.05× |

At 64/16 MTP4/7 (G ≥ 5), Prims-TS runs the FP8 Keeps recipe fed by an NVFP4-to-FP8 transform, persistent, in one wave or split along the KV sequence with a separate reduction; MTP3 (G4) uses the TMEM-transformed recipe. TP1 BS64 measured 1.38×, 1.31× and 1.33× at MTP3/4/7. At 64/4 every group exceeds a 32-row Swaps Q tile, so all rows run the Keeps recipe (Q64 at G4, Q128 at G5..G8). There FP8 KV is only partly memory-bound and the transform is not fully hidden: MTP4/7 is roughly at parity and up to 11% slower than FP8 KV (TP1 BS32 MTP7: 106.5 µs against 96.3 µs). Absolute latencies of one revision differ by a few percent between GB300 nodes.

Prefill rows were 0.95–1.07× at 64/16 and 0.85–1.00× at 64/4 (G8 0.85–0.86×). BF16-MMA NVFP4 was slower than FP8 KV on every 64/16 decode row. These figures describe one run of one revision; they are not expected-result assertions. The fourteen CPU checks (inventory of both geometries, case selection, invalid timings, failure and unsupported-backend accounting, index generation, byte accounting, summary rendering and the NVFP4 round trip) pass, as do Ruff lint/format checks.

## Comparing revisions

Run the same selected cases on baseline and candidate checkouts with identical hardware, software, manifests and timing controls. Compare the recorded Prims-TS source hash and benchmark hashes before comparing absolute latencies. Historical speedups are not expected-result assertions. Generated JSON belongs under `results/`; inspect failure text for local paths before publishing artifacts. The dense `compare_runs.py` and the vLLM-Triton sparse summaries use different schemas.
