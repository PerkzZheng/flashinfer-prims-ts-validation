# QToken-KvBlock-Sparse-Attention benchmark suites

Two reusable TP2 suites compare FlashInfer PrimTS with vLLM Triton. Both measure the complete sparse-attention pipeline, not just the FMHA kernel. They are separate from this repository's 564-row dense/MLA campaign and require recorded top-k routes plus a compatible vLLM source checkout.

## Coverage

| Suite | Cases | Workload | Query grouping |
| --- | --- | --- | --- |
| `q5` | 2 | BF16, 8K KV, 16/32 route groups, five Q tokens per group | Fixed G5; Q5 means MTP4, not five draft tokens |
| `tp2` prefill | 6 | BS1, SQ=KV=8K/16K/32K, BF16/FP8 | Packed Q; suggested G, normally G5 on the qualified GB300 |
| `tp2` decode | 48 | BS1/8/64/256, SQ1/SQ4 (MTP0/MTP3), KV8K/16K/32K, BF16/FP8 | Fixed `[B, num_query_groups, G, Hq, D]`; suggested G, normally G1 for BS1 |
| Optional `prefill-g4` control | 6 | Same prefill tensors and routes | Packed Q, fixed G4 |

The two default suites contain 56 cases; adding G4 controls gives 62. All use TP2-local Hq/Hkv=12/1, D256, 512 selected complete blocks of four tokens, and an exact causal tail of at most three tokens. The metadata bound is `model_len=131072` per request, not aggregate physical KV-cache capacity. FP8 uses E4M3 Q/K/V with BF16 output; BF16 also uses BF16 output. These standalone TP-shaped workloads run on one GPU: TP2 describes local head geometry, not an all-reduce measurement.

The Q5 suite deliberately reproduces the historical grouped-route proxy: 16 or 32 consecutive five-token groups from one real 8K request, starting at logical position 4107, share one physical cache. Its “batch size” is a route-group count, not independent serving requests. The 54-case matrix instead gives every decode request a disjoint physical cache and dense block-table row. Requests repeat the same recorded logical selection pattern and deterministic Q/K/V values. Physical cache residency differs; do not interchange these two interpretations.

Auto grouping is resolved once from the selected GPU's SM count, outside timing. The JSON records the actual G, TileQ, KV tile and both backends' split counts. Original case IDs retain their manifest G suffix for stable seeds; the resolved `query_group_size` field and appended `auto-gN` suffix are authoritative for automatic runs.

## Dependencies and source selection

Use a CUDA-enabled environment with matching PyTorch, Triton, FlashInfer and vLLM native dependencies. The sparse runner was ported from vLLM `b10972e13c51d0ba61906e64e9b01b87bd6a0549`. Its inference code is identical to tested source `a841edb6`; the later amend changes documentation only. The qualified FlashInfer feature head is `c3f42cdc34524e643800abcff5748be807be71ad` on main `fa2f4d0b`. Its annotation-only amend is executable-equivalent to tested head `0ac94841`.

```bash
git clone --recursive --branch qsa-packed-query-official-pr \
  https://github.com/PerkzZheng/flashinfer.git
git -C flashinfer checkout c3f42cdc34524e643800abcff5748be807be71ad

git clone --branch q-token-kv-block-sparse-ts-20260909 \
  https://github.com/PerkzZheng/vllm.git
git -C vllm checkout b10972e13c51d0ba61906e64e9b01b87bd6a0549
```

Follow each project's build/install instructions for the selected CUDA environment. Sparse qualification used CUTLASS DSL 4.7.1, CUDA 13.0 and GB300/SM103; this does not extend the dense campaign's B200/DSL4.7.0 qualification. Do not silently replace an incompatible vLLM native extension with one from another source revision. Installing the Python harness alone does not supply either project's kernels or native dependencies.

```bash
python -m pip install 'nvidia-cutlass-dsl[cu13]==4.7.1' pytest
export FLASHINFER_ROOT=/path/to/flashinfer
export VLLM_ROOT=/path/to/vllm
export VALIDATION_ROOT=/path/to/flashinfer-prims-ts-validation
export TRACE_ROOT=/path/to/recorded-route-archive
export RESULTS_ROOT=/path/to/results
```

The launcher prepends both selected source roots to `PYTHONPATH`; the CUDA driver checks where metadata and Triton operators actually imported from. Both Git identities, tracked-diff hashes, driver/helper hashes and actual Triton operator hashes are recorded. The complete PrimTS path uses the public prepared wrapper. Isolated component timings and launch-configuration reporting also use version-sensitive private helpers, so arbitrary released wheels are not promised compatible.

## Recorded route inputs

The two manifests under `suites/sparse_attention/` contain the exact trace paths, formats and SHA256 hashes. `--trace-root` is their common parent, independent of this repository's location. Existing archives need no file moves or source edits. All twelve unique files are checked before their suite runs. No model weights, prompts, credentials or tensor dumps are uploaded with this repository; obtain the original route archive separately. There is no synthetic fallback that silently changes the benchmark.

The Q5 artifact is `expanded_tokens_v1`, shape `[8192,2051]`. The loader reconstructs logical block-4 IDs and verifies the expanded live prefix and causal tail. The matrix consumes eleven `compact_blocks_v1` chunks: two at 8K, three at 16K and six at 32K. Chunks must cover positions zero through KV−1 exactly once. Prefill uses all rows; decode uses the final SQ rows. Replay retains logical IDs and physical-page adjacency/gap structure while remapping the physical allocation into a compact local cache.

The required common fields are `token_topk`, `compress_ratio`, `main_storage_page_size`, `token_to_req`, `logical_positions`, and `main_block_table`. Compact captures also provide `dump_format`, `selected_block_indices`, and `selected_tail_token_indices`; expanded captures provide `selected_token_indices`. Only tensor/primitive payloads readable by `torch.load(..., weights_only=True)` are accepted. Captured physical page sizes are 1600 for Q5 and 784 for the matrix; do not silently repack them to page four.

The matrix routes were captured at TP4 and replayed at TP2 for both dtypes. Selections are tokenwise, not headwise; using the same logical route pattern controls this part of the comparison. This is not evidence that independently sampled FP8 and BF16 indexers choose identical routes.

## Commands

Inventory requires neither a GPU, PyTorch, nor trace files:

```bash
python "$VALIDATION_ROOT/scripts/run_sparse_attention_suite.py" list
python "$VALIDATION_ROOT/scripts/run_sparse_attention_suite.py" list --suite prefill-g4
```

Verify the recorded file checksums without loading CUDA:

```bash
python "$VALIDATION_ROOT/scripts/run_sparse_attention_suite.py" validate \
  --trace-root "$TRACE_ROOT"
```

Run both canonical suites with 20 warmups and 300 samples per measured path:

```bash
python "$VALIDATION_ROOT/scripts/run_sparse_attention_suite.py" run \
  --source-root "$FLASHINFER_ROOT" --vllm-root "$VLLM_ROOT" \
  --trace-root "$TRACE_ROOT" --output-dir "$RESULTS_ROOT/sparse" --device 0
```

Add `--dry-run` to print commands without execution. Repeat `--suite q5 --suite tp2 --suite prefill-g4` to include the matched G4 comparison. Use a new output directory for each source/environment; existing files are not overwritten. `--resume` accepts only an exact match of manifest, selected cases, source identities, GPU/software, timing controls and seed. A failure stops the run and remains recorded. Shortened `--iterations` / `--warmup-iterations` runs are functional smoke tests, not performance evidence.

Inspect full component tables, including absolute BS1 latencies:

```bash
python "$VALIDATION_ROOT/scripts/summarize_sparse_attention.py" \
  "$RESULTS_ROOT/sparse/q5.json" "$RESULTS_ROOT/sparse/tp2.json"
```

The existing dense `compare_runs.py` expects a different schema; do not feed sparse result files to it.

## Timing and correctness contract

1. Allocate, prepare plans, initialize counters, compile and capture outside timing. Use native HND cache storage for PrimTS and native NHD for Triton with identical logical values. Exclude any setup layout copies from both sides.
2. Capture six separate CUDA graphs: metadata, PrimTS core, PrimTS combined, Triton expansion, Triton core and Triton combined. Core includes the backend's reduction when needed. PrimTS combined preserves the prepared metadata → attention → reduction PDL chain; Triton combined includes `expand_qsa_block_indices` and its attention/reduction.
3. Before every measured graph, update a separately seeded random-byte buffer of at least 256 MiB and at least twice reported L2 size on the same stream. Capture that scrub separately and place CUDA timing events after it. Do not scrub between metadata and attention inside a combined graph.
4. Rotate and periodically reverse path order; retain all 300 CUDA-event samples. The primary speedup is `mean(Triton combined) / mean(PrimTS combined)`; values above one favor PrimTS. Raw samples, mean, median, P95 and coefficient of variation remain in JSON.
5. Verify causal selections, exact Triton expansion, live metadata locators/membership bytes, and output agreement before and after graph timing. Compare sampled rows to an FP32 oracle when either backend splits KV. BF16 tolerances are atol/rtol 0.02; FP8 uses 0.05. Failures are not converted into performance passes.

The separately timed metadata graph is its isolated latency. `combined − core` is only an incremental pipeline difference and must not be called metadata kernel duration: launch envelopes, overlap/PDL and cache reuse can differ. Likewise, do not sum isolated components to estimate end-to-end latency. Selected-token counts and union-selected logical K/V bytes are recorded; these are not measured HBM transaction counts.

No Triton-speedup threshold changes kernel dispatch or hides slow rows. BS1 and some BS8 rows were known outliers. Record and investigate them using general policies rather than per-shape tuning.

## Port validation and historical results

The source runner previously passed all 62 cases (including G4 controls) on GB300/SM103 after the rebase. The standalone public API example also passed packed prefill G5/partial groups and G1, fixed MTP3 decode G4/G1, FP32 sampled outputs and poisoned graph replay. The example's Python calls do not need a rebase-specific change; its old README install pin should be advanced to the qualified head above.

The former sparse runner initialized its eviction buffer with uniform bytes. Blackwell can compress that pattern. This port uses random bytes and records a distinct timing protocol; earlier sparse timings must not be presented as fresh non-compressible-cold-L2 signoff. The previous same-node rebase comparison remains historical evidence, and no new GPU performance numbers are supplied by this packaging change. Run the published suites before making a new performance claim.

CPU preflight checks cover inventory, fixed/automatic command selection, trace checksums, incomplete-run rejection and summary ratio semantics:

Port checks pass: all 56 case contracts and twelve trace hashes match the original suites; 35 existing classes/functions retain their executable AST apart from trace-path redaction. The 17 new CPU checks and eight existing source-independent CPU checks pass, as do Ruff lint/format checks. GPU timings were not rerun for this publication. The wider dense/MLA CPU preflight still uses its existing staged package layout and PyTorch environment; it is not claimed rerun by these sparse checks.

```bash
python -m pytest -q "$VALIDATION_ROOT/scripts/sparse_attention_cpu_checks.py"
```

Normal result provenance omits hostnames and absolute source/trace paths. Failed exception messages can still contain paths; inspect failed artifacts before publishing them. Keep raw JSON and the exact source/trace manifests with every performance decision.
