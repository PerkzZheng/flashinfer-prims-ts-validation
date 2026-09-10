# FlashInfer PrimTS validation suite

This repository contains the accuracy and performance-regression workflow used
to qualify FlashInfer PrimTS FMHA context, FMHA decode, and MLA decode kernels.
It exercises FlashInfer's public Python interfaces; it does not contain the
kernels themselves.

The two [QToken-KvBlock-Sparse-Attention suites](docs/sparse_attention.md) add a separate PrimTS-versus-vLLM-Triton workflow: TP2 BF16 Q5 at 16/32 route groups, and a 54-case BF16/FP8 prefill/decode matrix at 8K/16K/32K. Optional G4 prefill controls allow comparison with suggested G5. These sparse suites require externally supplied recorded top-k routes and a compatible vLLM checkout; they are not included in the 564-row dense/MLA total below.

The full performance campaign executes 564 rows:

- 350 paired PrimTS-versus-TRTLLM-Gen decode rows;
- 22 paired PrimTS-auto-versus-CuTe-DSL MLA feature rows;
- 128 causal-context correctness and standalone performance rows; and
- 64 paired FP8 causal-context rows, reusing the FP8 half of the context
  matrix.

The 414 TRTLLM-Gen comparison rows fail when PrimTS is more than 5% slower
under the order-balanced cold-L2 protocol described below. The 22 focused CuTe
DSL rows use the feature campaign's hot-graph protocol and record a 6% per-row
diagnostic; accepted outliers remain visible instead of becoming case-specific
policy exceptions. Numerical accuracy is checked before and after timing.

## Qualification environment

- Full performance gate: NVIDIA B200 (SM100) with CUDA 13.0.
- Accuracy gate: CUDA 12.9 and CUDA 13.0, run independently.
- Compiler package: public `nvidia-cutlass-dsl==4.7.0`.
- Source: a FlashInfer Git checkout containing `flashinfer.attention.prims_ts`.
- Reference backends: TRTLLM-Gen artifacts supplied through the matching
  FlashInfer/`flashinfer-cubin` installation, plus monolithic CuTe DSL for the
  focused MLA feature suite.

Other devices may work, but the recorded checkpoint was collected on B200.
Performance results from different GPUs, drivers, toolkits, power states, or
benchmark revisions are not directly comparable.

The focused grouped-tokens/heads-Q suite was additionally qualified on B200
with CUDA 13.4, PyTorch 2.14 nightly, and imported CUTLASS DSL 4.7.0.

## Installation

Create the CUDA/PyTorch environment appropriate for the target FlashInfer
checkout, then install the public CUTLASS DSL release and test dependency:

```bash
python -m pip install nvidia-cutlass-dsl==4.7.0 pytest
```

Build or install FlashInfer by following the
[upstream installation guide](https://docs.flashinfer.ai/installation.html).
Keep the source checkout available; all commands below point to it explicitly:

```bash
export FLASHINFER_ROOT=/path/to/flashinfer
export VALIDATION_ROOT=/path/to/flashinfer-prims-ts-validation
export RESULTS_ROOT=/path/to/results
```

## Accuracy gate

Run the four numerical PrimTS test files once in the CUDA 12.9 environment and
again in the CUDA 13.0 environment:

```bash
python "$VALIDATION_ROOT/scripts/run_accuracy.py" \
  --source-root "$FLASHINFER_ROOT" \
  --output-dir "$RESULTS_ROOT/accuracy-cu129"

python "$VALIDATION_ROOT/scripts/run_accuracy.py" \
  --source-root "$FLASHINFER_ROOT" \
  --output-dir "$RESULTS_ROOT/accuracy-cu130"
```

The runner verifies that FlashInfer imports from the selected checkout and
that CUTLASS DSL 4.7.0 is installed. It writes `accuracy.log`, `accuracy.xml`,
and a path-redacted `manifest.json` while preserving pytest's exit status.

At the recorded release snapshot, each environment collected 360 tests:

| Test file | Cases |
| --- | ---: |
| FMHA context | 134 |
| FMHA decode | 125 |
| Mask handling | 9 |
| MLA decode | 92 |
| Total per environment | 360 |

Counts can grow with FlashInfer. The release gate requires at least 360 tests,
zero failures/errors, and zero skips. Structural trace-template checks are
separate from numerical accuracy and can be added with `--include-trace`.
`--allow-partial` exists only for diagnostic selections and is not signoff.
Because these files come from the selected FlashInfer checkout, this gate also
collects newly added fixed/packed query, empty-request, dynamic-batch, and
non-power-of-two MLA tests without copying them into this repository.

## Benchmark coverage

| Suite | Rows | Coverage |
| --- | ---: | --- |
| FMHA decode, SQ=1 | 120 | B 1/4/16/64/128 × Hq 32/64 × Hkv 4 × KV 2K/8K × D 64/128/256 × FP8/BF16; dense-equivalent SQ1, paged KV, page 16 |
| FMHA speculative decode | 30 | causal SQ 2/4/8; Qwen-style Hq/Hkv/D 96/8/128 and GPT-OSS-style 64/8/64; B 8/16/32/40/64; KV 16K; FP8; page 32 |
| MLA decode, SQ=1 | 100 | Hq 8/16/32/64/128 × B 1/4/8/64/128 × KV 2K/8K × FP8/BF16; latent/nope/RoPE 512/512/64; BF16 output; page 32 |
| MLA decode, SQ=4 | 100 | The same 100-row product with grouped bottom-right-causal queries |
| MLA grouped tokens/heads-Q | 22 | PrimTS auto vs monolithic CuTe DSL; B4/K512 equivalent 48-row and 96-row Hq 12/24/48/96 factorizations; BF16/FP8; both 1CTA/2CTA and long-K split reducers; paired B3/B4 compile reuse |
| FMHA causal context | 128 | FP8 and BF16: D 128/256 × Hq 32 × Hkv 32/4 × B 1/4 × (SQ, SKV) 1K/1K, 4K/4K, 16K/16K, or 256/4K × packed-ragged separate QKV/paged KV; page 32 |
| FP8 context comparison | 64 | FP8 half of the same causal-context matrix, with common BF16 output for PrimTS and TRTLLM-Gen |

The unified runner owns all 564 executions. The 128-row context suite records
PrimTS correctness and latency but has no paired reference measurement. The
other rows contain 414 PrimTS/TRTLLM-Gen comparisons and 22 PrimTS/CuTe-DSL
comparisons through public FlashInfer APIs.

## Running the benchmarks

Inspect the suite inventory and complete context matrix without running CUDA:

```bash
python "$VALIDATION_ROOT/scripts/run_flashinfer_ts_suite.py" list \
  --source-root "$FLASHINFER_ROOT"
```

Run the CPU-only bundle and metric preflight:

```bash
python "$VALIDATION_ROOT/scripts/run_flashinfer_ts_suite.py" validate \
  --source-root "$FLASHINFER_ROOT" \
  --output-dir "$RESULTS_ROOT/preflight"
```

Print the exact signoff commands without GPU execution:

```bash
python "$VALIDATION_ROOT/scripts/run_flashinfer_ts_suite.py" run \
  --source-root "$FLASHINFER_ROOT" \
  --output-dir "$RESULTS_ROOT/dry-run" \
  --dry-run
```

Run the complete 564-row CUDA 13.0 qualification campaign:

```bash
python "$VALIDATION_ROOT/scripts/run_flashinfer_ts_suite.py" run \
  --source-root "$FLASHINFER_ROOT" \
  --output-dir "$RESULTS_ROOT/performance-cu130" \
  --device 0
```

Select suites by repeating `--suite`:

```bash
python "$VALIDATION_ROOT/scripts/run_flashinfer_ts_suite.py" run \
  --suite mla-groups-tokens-heads-q \
  --source-root "$FLASHINFER_ROOT" \
  --output-dir "$RESULTS_ROOT/mla-groups-tokens-heads-q"
```

`--quick` shortens timing for functional smoke tests. It is not performance
evidence. `--continue-on-error` collects later failures but never converts a
failure into a pass.

The runner creates a temporary decode-driver staging tree from Git-tracked
FlashInfer files only, removes it at exit, and stores redacted commands in the
manifest. Local untracked files are never copied into benchmark artifacts.

## Performance gate

For paired comparisons, both backends receive the same logical tensors, page
IDs, runtime lengths, scales, output dtype, and masking semantics. The timing
contract is:

1. Plan, compile, allocate, capture, and warm up outside the timed interval.
2. Capture one public backend call per CUDA graph.
3. For the 414 TRTLLM-Gen rows, immediately before every measured replay,
   update a non-compressible buffer sized to twice the GPU L2 cache on the same
   stream and exclude that scrub from CUDA-event timing.
4. For the 22 focused CuTe-DSL rows, use hot one-call graph replays to match the
   feature's public-backend performance campaign.
5. Alternate backend order. A complete cycle contains one PrimTS-first and one
   reference-first replay.
6. Check results before timing, poison/replay outputs where applicable, and
   check results again after timing.

The decision statistic for each paired row is:

```text
(sum(PrimTS sample duration) / sum(reference sample duration) - 1) * 100
```

A value greater than `+5.0%` fails a TRTLLM-Gen row. The focused CuTe-DSL suite
classifies values above `+6.0%` as recorded outliers but does not fail the run,
because its purpose is to expose the complete structural distribution without
encouraging head-, batch-, or sequence-specific policy exceptions. A negative
value means PrimTS is faster. Independent medians, ratio-of-medians, per-cycle
percentiles, p95, minimum, and maximum are diagnostics only.

For a row above its diagnostic threshold, rerun that exact row at a larger
sample count in the same session. Preserve both artifacts and identify timing
variance versus a repeatable kernel gap before considering a general policy or
kernel change.

## Comparing candidate and prior-source runs

The same-run TRTLLM-Gen comparison is the primary gate. To detect drift from a
previous FlashInfer revision as well, run the identical suite from baseline and
candidate worktrees on the same idle GPU, then compare corresponding artifacts:

```bash
python "$VALIDATION_ROOT/scripts/compare_runs.py" \
  "$RESULTS_ROOT/baseline/fmha-decode/sq1-full120.json" \
  "$RESULTS_ROOT/candidate/fmha-decode/sq1-full120.json" \
  --output "$RESULTS_ROOT/comparison/fmha-sq1.json" \
  --markdown-output "$RESULTS_ROOT/comparison/fmha-sq1.md" \
  --threshold 5
```

`compare_runs.py` requires identical case sets, row contracts, timing controls,
sample counts, benchmark hashes, GPU identity, software versions, and reference
artifact identity. Source commits and PrimTS source hashes may differ by design.
Environment mismatches are rejected unless explicitly overridden; overridden
comparisons should be treated as diagnostic rather than signoff evidence.

## Accuracy methodology

- Context uses deterministic sampled exact FP32 bottom-right-causal attention,
  checks finiteness across the full output, and covers every matrix row.
- Paged context uses shuffled nonidentity physical page IDs and poisoned guard
  pages.
- Decode drivers compare both backends with a backend-neutral reference and
  validate outputs again after CUDA-graph timing.
- TRTLLM-Gen workspaces and multi-CTA counter buffers are allocated before
  capture; counter reset state is checked after direct calls and graph replays.
- The shifted `(SQ, SKV)=(256,4096)` context rows specifically cover nonzero
  bottom-right causal offsets used by chunked prefill.

## Artifacts

Keep the following together for every release decision:

- `manifest.json` from the unified runner;
- per-suite JSON files with raw samples and accuracy fields;
- MLA CSV and Markdown summaries;
- accuracy log, JUnit XML, and accuracy manifest;
- candidate-versus-baseline comparison JSON/Markdown; and
- checksums for the archived directory.

The scripts intentionally avoid serializing hostnames, GPU UUIDs, private
package URLs, absolute source paths, and environment-variable contents in
normal provenance. Error logs and failed-row JSON tracebacks can still contain
paths or exception text from Python or CUDA, so inspect every failed artifact
before making it public.

## Recorded release checkpoint

An earlier CUDA 13.0 B200 campaign with public CUTLASS DSL 4.7.0 produced the
following results. Decode used the same non-compressible scrub contract as this
repository. The historical context driver used a zero-filled scrub buffer,
which Blackwell L2 may compress; its latency results are therefore not valid
signoff evidence for the corrected protocol and must be rerun.

| Suite | Completed | Reference / status |
| --- | ---: | --- |
| FMHA decode, SQ=1 | 120/120 | +1.4779% |
| Causal FMHA decode, SQ=2/4/8 | 30/30 | +2.5898% |
| MLA decode, SQ=1 | 100/100 | +3.4224% |
| MLA decode, SQ=4 | 100/100 | +3.2929% |
| MLA grouped tokens/heads-Q | 22/22 | CuTe DSL; 7 rows above the 6% diagnostic, worst +11.1996%; CuTe/PrimTS geometric mean 1.0216 |
| FP8 causal context comparison | 64/64 accuracy | Performance rerun required with the current scrub |
| FP8/BF16 causal context | 128/128 accuracy | Current standalone performance rerun required |

The MLA SQ=1 value is from an independent larger-sample confirmation of the
initial worst row. All 542 executions in the earlier campaign passed their
embedded accuracy checks, and the separate numerical suite passed 360/360
under CUDA 12.9 and 360/360 under CUDA 13.0. A fresh full run of the published
scripts is the release decision; this checkpoint is provenance, not a
substitute. The 22-row grouped-tokens/heads-Q result is a fresh CUDA 13.4 run
with 200 samples per backend, zero accuracy/runtime/metric failures, and two
expected compile cache hits from the B3→B4 topology-reuse pairs.

## Repository layout

```text
scripts/run_accuracy.py                 numerical accuracy gate
scripts/run_flashinfer_ts_suite.py      unified 564-row benchmark runner
scripts/compare_runs.py                 candidate-versus-baseline drift gate
scripts/bench_attention_ts_*.py         benchmark drivers
scripts/attention_ts_decode_*.py        fixtures, timing, and artifacts
scripts/*_cpu_checks.py                 CPU-only metric-contract tests
scripts/run_sparse_attention_suite.py   two TP2 sparse suites and optional G4 controls
scripts/summarize_sparse_attention.py   absolute pipeline/component tables
suites/sparse_attention/*.json          recorded sparse workload/trace manifests
docs/sparse_attention.md                sparse setup, methodology, and caveats
```

## License

Apache-2.0. The benchmark code retains the FlashInfer project copyright and
notice. FlashInfer, CUTLASS, NVIDIA, and TRTLLM-Gen names identify compatible
projects and comparison backends; this repository is an independent validation
harness, not an official FlashInfer release gate unless adopted upstream.
