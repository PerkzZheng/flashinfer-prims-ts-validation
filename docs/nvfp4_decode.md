# Dense NVFP4 decode benchmark suites

These suites reproduce the dense decode workloads used for
[FlashInfer #9315](https://github.com/flashinfer-ai/flashinfer/pull/9315),
[#9311](https://github.com/flashinfer-ai/flashinfer/issues/9311), and the
[D512 follow-up in #6979](https://github.com/flashinfer-ai/flashinfer/issues/6979#issuecomment-6356261073).
They are separate from the QSA sparse NVFP4 suite and the historical 564-row
campaign. They exercise public automatic dispatch; no kernel overrides or
experimental tuning patches are installed.

## Workloads

| Manifest under `suites/nvfp4_decode/` | Cases | Requested backend executions | Coverage |
| --- | ---: | ---: | --- |
| `issue9311.json` | 16 | 64 | B1/2/4/8/16/32/64/128, Q1/Q6, Hq16/Hkv2, D256, KV65536, page128; Prims NVFP4/FP8, TRT FP8/NVFP4 |
| `issue6979.json` | 64 | 128 | B8/B32, Q1/4/5/6, Hq16/Hkv2 and Hq32/Hkv4, D512, KV8192/32768, page32/64; Prims NVFP4 and TRT FP8 |
| `q9.json` | 1 | 2 | B32, Q9, Hq16/Hkv2, D512, KV32768, page64; Prims NVFP4 and TRT FP8 |
| `fp8_controls.json` | 6 | 6 | FP16/BF16 output for the three homogeneous-FP8 SQ1/full-grid shapes from the dispatch-scope review |

A case is one input geometry. A backend execution is one timing/reference row.
All use FP8 Q, HND paged caches, shuffled page IDs, complete storage pages,
and softmax scale `1/sqrt(D)`. The manifests retain exact seeds and geometry.

The input and timing contracts differ deliberately:

- **#9311:** independently generated FP8 and NVFP4 caches, as in the original
  reproduction. Prims outputs BF16, TRT FP8 outputs BF16, and TRT NVFP4 outputs
  FP8. Q1 is dense-equivalent and Q6 is bottom-right causal. KV lengths are
  supplied at execution time. Ten CUDA-graph replays of 20 calls, without an L2
  flush; the historical comparison statistic is the best per-call latency.
  Medians and every sample are retained. These are throughput diagnostics,
  not matched-logical-input or equal-output-precision comparisons.
- **#6979/Q9:** TRT FP8 K/V hold the exact FP8 dequantization of the NVFP4
  payloads and scales. Both backends output FP16, use causal queries, and plan
  uniform KV lengths. Median of 50 `bench_gpu_time` samples, 10 warmups,
  CUDA graphs, `enable_cupti=True`, `cold_l2_cache=True`. The FlashInfer timing
  API owns the scrub and any CUPTI fallback behavior. The fixtures preserve the
  historical synthetic signed scale distribution and RNG draw order.
- **FP8 controls:** seeded normal FP8 Q, normal K/V scaled by 0.5, causal
  queries, planned uniform lengths, FP16/BF16 outputs, and the same cold-CUPTI
  protocol. Run separately against the base and candidate to compare revisions.

All timing includes attention and any split-KV reduction. Allocation, input
conversion, planning, compilation and reference computation are excluded.
Backends run sequentially in manifest order. This preserves the session's
protocol; it is not the order-balanced protocol of the older dense suite.
Run revisions on the same idle GPU and preserve repetitions when studying
small differences.

## Environment and source requirements

Use a clean FlashInfer Git checkout containing the Prims-TS decode API and
`tests/attention/test_attention_ts_decode.py`. The publication checkpoint is
`e559d416bae79f0a203bdd94bf0103542defdab8`; the review comparator is main
`cfa0318a1316e8838bdee32c2a2014e22001dabe`.

This session used CUDA 13.4, PyTorch 2.14 development builds, and CUTLASS DSL
4.8.1 on Rubin, GB300 and B200. This suite's dependency version differs from
the historical 4.7.0 default in the repository's root requirements. In the
matching CUDA/PyTorch environment, install:

```bash
python -m pip install 'nvidia-cutlass-dsl[cu13]==4.8.1' cupti-python pytest
python -m pip install --no-deps --no-build-isolation -e "$FLASHINFER_ROOT"
```

TRT comparisons additionally need matching FlashInfer TRTLLM-Gen artifacts
and architecture support. Missing/unsupported backends are failures, not
silently omitted rows. Use an explicit `--backend` selection when intentionally
running a subset; the selected list is recorded in the output.

The runner verifies the imported FlashInfer and oracle paths, records the
source commit and oracle/harness hashes, and rejects tracked dirty source.
It uses a source-specific JIT workspace by default and disables DSL file
caching unless explicitly configured by the caller. Source changes during a
run invalidate it. It does not record hostnames, GPU UUIDs, private paths or
package URLs in normal JSON metadata. Failed-call exception details go to
stderr, which should be reviewed before publishing logs.

## Running and reporting

Set paths as in the root README. List a workload without CUDA:

```bash
python "$VALIDATION_ROOT/scripts/bench_attention_ts_nvfp4_decode.py" \
  --suite "$VALIDATION_ROOT/suites/nvfp4_decode/issue9311.json" --list
python -m pytest -q "$VALIDATION_ROOT/scripts/nvfp4_decode_cpu_checks.py"
```

Run the complete #9311 matrix on each target GPU:

```bash
python "$VALIDATION_ROOT/scripts/bench_attention_ts_nvfp4_decode.py" \
  --source-root "$FLASHINFER_ROOT" \
  --suite "$VALIDATION_ROOT/suites/nvfp4_decode/issue9311.json" \
  --output "$RESULTS_ROOT/issue9311.json"
python "$VALIDATION_ROOT/scripts/summarize_nvfp4_decode.py" \
  "$RESULTS_ROOT/issue9311.json" --output "$RESULTS_ROOT/issue9311.md"
```

Replace the manifest with `issue6979.json`, `q9.json`, or `fp8_controls.json`
for the other cohorts. Use `--case ID` and `--backend NAME` repeatedly to
select exact cases/backends. `--repeats 3` is useful for functional smoke runs,
not performance qualification. Existing output files are never overwritten.
Each completed case is saved atomically, including failed backends; any failure
or incomplete run yields a nonzero exit status. The summary prints failed rows
with no qualified latency or speedup.

Run the same command against a base checkout and a candidate checkout, with
different output paths, then compare:

```bash
python "$VALIDATION_ROOT/scripts/summarize_nvfp4_decode.py" \
  "$RESULTS_ROOT/candidate.json" --baseline "$RESULTS_ROOT/base.json" \
  --output "$RESULTS_ROOT/comparison.md"
```

The comparison requires matching suite, selected cases/backends, timing
controls, sample counts, software/GPU model and capacity, harness hashes,
input fingerprints, masks and output dtypes. Source revisions and their
numerical oracle hashes may differ by design. Because public metadata omits
GPU UUIDs, the operator must ensure the physical GPU is the same. A valid
comparison records ratios; it imposes no arbitrary performance gate.

## Numerical qualification

Before timing and after poisoning/replaying the output, every output must be
finite. Prims additionally uses the selected source revision's existing
`_fp8_decode_reference` and unchanged `_assert_case_correct` tolerances on the
first `min(B, 2)` requests. The reference models FP8-P rounding using the actual
split count, KV MMA instances and Q grouping. NVFP4 V scales are converted from
the four-token storage order to the logical order for independent dequantization.

TRT rows are explicitly **finiteness-only**, preserving the session's scope;
the suite does not claim a strict TRT numerical oracle or cross-backend equality.
Unlike the old scratch runner's post-timing finiteness-only check, the packaged
runner repeats the sampled Prims numerical check after graph replay.

The #6979 FP16 cohort has known strict-reference failures at both base and PR:
Hq16/Hkv2 B8/KV32768/page64/Q6; Hq32/Hkv4 B32/KV8192/page32/Q6;
and Hq32/Hkv4 B32/KV8192/page64/Q5. They remain in the suite, fail the command,
and are excluded from qualified timing claims. No tolerance is relaxed. Other
source versions can have different failures, so inspect every row.

Q9 remains a separate optimization follow-up. Q64 at GQA8 covers eight query
tokens; the recorded Q9 control uses two Q CTAs. Q CTAs and KV splits are
reported separately; no test pins a particular heuristic choice.

## Optional dispatch audit

The policy audit records 1,824 FP8 probes and 192 NVFP4 controls without
compiling/running kernels. It covers fixed/packed queries, attention sinks,
SQ1/multi-token queries, full grids, and explicit Keeps eligibility. Unsupported
configurations retain their rejection reason. This is a diagnostic comparison,
not a maintained golden dispatch snapshot or a numerical test.

```bash
python "$VALIDATION_ROOT/scripts/audit_attention_ts_decode_dispatch.py" \
  --source-root "$BASE_FLASHINFER_ROOT" --kind fp8 \
  --output "$RESULTS_ROOT/base-policy.json"
python "$VALIDATION_ROOT/scripts/audit_attention_ts_decode_dispatch.py" \
  --source-root "$FLASHINFER_ROOT" --kind fp8 \
  --baseline "$RESULTS_ROOT/base-policy.json" \
  --output "$RESULTS_ROOT/candidate-policy.json"
```

`--baseline` returns nonzero for any changed choice or rejection; an intended
policy change therefore needs interpretation, not a new expected snapshot.
For the NVFP4 preservation check use `--kind nvfp4` and compare published parent
`7cf20a41aabfec6521b2b6176e6f9bd14e1055f1` with `e559d416b`.

## Maintained correctness selection

Run the existing source tests without copying them into this repository:

```bash
cd "$FLASHINFER_ROOT"
python -m pytest tests/attention/test_attention_ts_decode.py -q --tb=short \
  --junitxml="$RESULTS_ROOT/decode.xml" \
  -k '(mixed_precision and nvfp4 and d256) or large_tile_size_q or nvfp4_resource_overrides or final_statistics or nvfp4_padded_scale_page_stride or mixed_precision_more_loaders_than_pages or large_grouped_nvfp4_grid or split_workspace_is_plan_invariant or plans_share_one_zeroed_workspace or replanned_sizes or packed_fp8_q_mask_product or fp8_speculative_runtime_reduction or static_fp8_d128_odd_kv_tail_is_finite or wide_fp8_gqa_accuracy'
```

At `e559d416b`, all 225 selected cases passed on B200. Counts can change with
the selected source version. This is a focused selection, not a full repository
or Rubin/GB300 requalification.

## Recorded session evidence

The [review reply](https://github.com/flashinfer-ai/flashinfer/pull/9315#issuecomment-6911335783)
records the original scripts' B200 review validation: 1,806 accepted FP8
configurations and 18 matching rejections equal the base, while 192 NVFP4
controls match the preceding PR head. The six FP8 timings below used 50 cold
CUPTI graph samples; both source revisions passed their numerical checks.
These historical timings are provenance, not fresh measurements of the
packaged runner.

| B | Q | D | Hq/Hkv | KV | Page | Output | Base µs | PR µs |
| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 32 | 1 | 128 | 64/1 | 8192 | 32 | FP16 | 23.38 | 23.42 |
| 32 | 1 | 128 | 64/1 | 8192 | 32 | BF16 | 23.44 | 23.42 |
| 128 | 4 | 128 | 64/1 | 8192 | 64 | FP16 | 108.21 | 108.13 |
| 128 | 4 | 128 | 64/1 | 8192 | 64 | BF16 | 108.16 | 108.27 |
| 16 | 8 | 64 | 64/4 | 16384 | 64 | FP16 | 114.61 | 115.14 |
| 16 | 8 | 64 | 64/4 | 16384 | 64 | BF16 | 114.56 | 115.22 |

The PR description retains the separate Rubin/GB300 NVFP4 performance evidence
and its precise source revisions. Compiler experiments, cluster launch scripts,
profiler installations and raw internal traces are not dependencies of this suite.

## Packaged runner validation

The published runner was exercised on one B200 with CUDA 13.4, Torch
2.14.0a0+4fdf77b940.nv26.08 and DSL 4.8.1, using source `e559d416b`:

- All 13 CPU contract checks pass.
- All 20 initial GPU backend executions pass their stated checks: six FP8
  controls; B1/Q1 and B1/Q6 with all four #9311 backends; D512 B8/Q5 and
  B8/Q6, Hq16/Hkv2, KV8192/page64 with Prims NVFP4 and TRT FP8; and the
  Q9 control with both backends. The NVFP4 smoke cases use three timing
  repetitions and are not new performance qualification.
- The final comparison path passes all six FP8 controls on both base
  `cfa0318a13` and candidate `e559d416b`, with 50 samples per row, and
  generates the complete matched-revision table.
- The packaged audit reproduces zero differences across 1,824 FP8 probes
  versus the base and 192 NVFP4 controls versus parent `7cf20a41aa`.

This validates packaging and the selected paths. The entire 200-execution
workload has not been rerun with this packaged runner, nor have Rubin/GB300
been requalified as part of publication. Historical evidence remains labeled
separately above.
