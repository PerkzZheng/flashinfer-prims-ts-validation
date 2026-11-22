# Mask-tile dispatch A/B suite

This suite checks changes to how Prims-TS kernels decide which K tiles need a
score mask, such as
[FlashInfer #9846](https://github.com/flashinfer-ai/flashinfer/pull/9846). Such a
change should leave every output bit unchanged and must not slow down any masked
path. The suite runs a base and a candidate FlashInfer checkout alternately on
one GPU. For each case it reports the latency ratio, whether the outputs are
bitwise identical, and the result of an independent FP32 oracle.

It is separate from the historical 564-row campaign and from the other suites
in this repository.

## Workload

`suites/mask_tiles/mask_tiles.json` contains 33 cases. Each case names the
mask path it covers in its `covers` field.

| Family | Cases | Public API | Mask paths covered |
| --- | ---: | --- | --- |
| Context | 12 | `batch_prefill`, `batch_prefill_with_paged_kv_cache` | causal with S_q = S_kv (TAIL-only control); sliding windows 1024/4000; per-token variable window; packed causal and sliding window with S_kv > S_q; packed dense with an unaligned K length; paged shifted causal at D128 and D256 (the register-bound LDTM softmax), plus D256 S_q = S_kv |
| Decode | 8 | `batch_decode_with_paged_kv_cache` | SQ1 dense with unaligned lengths; grouped causal MTP (SQ 4/8); SQ1 and MTP sliding window 1024; FP8 grouped Q128 causal (Hq 128/256); D256 control |
| MLA | 13 | `batch_mla_decode_with_paged_kv_cache` | SQ1 causal and dense; fixed causal MTP; packed-Q causal (`qo_indptr`); two long split-KV requests; Hq 16/32 (1CTA path); context parallel 8×64 and 4×1 (SQ1, MTP, packed Q) |

The inputs are deterministic:

- Request lengths are either written out in the manifest or drawn from a
  seeded `random.Random` range, so they expand the same way on any host.
- Each case seeds a CUDA generator from the suite seed and the case index.
- Page tables are shuffled, every page is complete, and the KV layout is HND.
- Context uses BF16. Decode uses BF16 or FP8 Q/K/V with BF16 output.
- MLA uses a BF16 latent cache with 576-wide keys and 512-wide values.
- The softmax scale is `1/sqrt(D)`, or `1/sqrt(576)` for MLA.
- Context-parallel MLA cases store only the rank's block-cyclic share of the
  cache. They pass global `seq_lens_kv` together with `cp_world`,
  `cp_interleave_size` and `cp_rank`.

## Running

Set the paths as in the root README. Both checkouts must be clean Git checkouts
that build in the same environment.

List the workload and run the CPU checks without CUDA:

```bash
python "$VALIDATION_ROOT/scripts/bench_attention_ts_mask_tiles.py" --list
python -m pytest -q "$VALIDATION_ROOT/scripts/mask_tiles_cpu_checks.py"
```

Run the interleaved A/B (five rounds by default) on one idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python "$VALIDATION_ROOT/scripts/run_mask_tiles_ab.py" \
  --base-root "$BASE_ROOT" \
  --candidate-root "$FLASHINFER_ROOT" \
  --output-dir "$RESULTS_ROOT/mask-tiles"
```

- **Process order.** Round *k* runs the two checkouts back to back, each in a
  fresh process. The first checkout alternates between rounds (base first,
  then candidate first, and so on).
- **Output files.** Each process writes `base-rK.json` or `candidate-rK.json`
  and a log beside it.
- **JIT cache.** Kernels are compiled into a JIT workspace per source commit
  under the output directory, so later rounds reuse them.
- **Report.** At the end, the driver writes `comparison.json` and
  `comparison.md`, prints the Markdown report, and exits nonzero if any case
  fails.
- **Selecting cases.** Use `--case ID` (repeatable) to run part of the suite.
- **Other options.** `--samples N` changes the number of timing samples, and
  `--rounds N` the number of rounds.
- **Existing results.** Results already present in the output directory are
  never overwritten.

To compare runs that already exist, give the files in round order:

```bash
python "$VALIDATION_ROOT/scripts/summarize_mask_tiles.py" \
  --base base-r0.json base-r1.json --candidate candidate-r0.json candidate-r1.json \
  --output-prefix comparison
```

To run one checkout without a comparison, call
`bench_attention_ts_mask_tiles.py --source-root ... --output ...` directly.

## Timing

`torch.profiler` records the device time of every CUDA kernel a public API call
launches. This includes the attention kernels and any split-KV reduction. It
excludes PyTorch-native kernels (input handling) and memory copies or sets.
Host work is outside the measurement: planning, metadata reads, compilation and
launch overhead. This matters because the convenience APIs read request lengths
on the host and cannot be captured in a CUDA graph.

Each run:

1. makes one oracle call;
2. makes three warm-up calls;
3. takes seven samples of ten calls each;
4. records the median per-call time of those samples.

The comparison statistic for each case is the **median over rounds of the
paired ratio** between candidate and base. Because each round puts the two
revisions back to back, slow drift in GPU clock or temperature cancels out of
the ratio. A case fails if this median is above `1 + --max-regression`, which is
1% by default. The report also shows the smallest and largest per-round ratios
and the geometric mean across cases.

Comparisons across different GPUs or different times are not valid.

- The comparison refuses runs whose GPU model, SM count, compute capability,
  PyTorch, CUDA, CUTLASS DSL version or harness hashes differ.
- Normal metadata does not record GPU UUIDs, so the operator must make sure
  both checkouts ran on the same physical GPU.
- Cases faster than about 20 µs can vary by more than 1% between rounds. Run
  more rounds, or confirm a flagged case with `--case`, before treating it as a
  regression.

## Correctness

Each case runs these checks in each process:

- **Finite output.** Every output value must be finite.
- **FP32 oracle.** The first two requests are compared with an independent FP32
  oracle, computed in chunks of 512 query rows. It applies FlashInfer's
  bottom-right visible interval:
  - causal rows see keys up to `k_len - q_len + row`;
  - a sliding window keeps the last `window_left + 1` of those keys;
  - variable windows use inclusive per-token starts and ends;
  - context-parallel MLA rows see each owned key whose global position lies
    inside that interval.
- **Tolerances.** BF16 requires relative L2 ≤ 0.01 and max |Δ| ≤ 0.01. FP8
  requires relative L2 ≤ 0.05 and max |Δ| ≤ 0.02, because the kernel rounds P
  to FP8. On B300 the measured worst cases were 2.8e-3 / 0.001 (BF16) and
  2.5e-2 / 0.004 (FP8). The oracle catches tile-sized masking errors. One
  extra or missing key among thousands can stay inside these tolerances; the
  base/candidate byte comparison below catches it.
- **Determinism.** A second call after timing must produce byte-identical
  output.

Comparing a base and a candidate adds three requirements:

- each revision's runs have the same input fingerprint;
- each revision's output SHA-256 is the same in every round;
- the base and candidate outputs are bitwise identical.

The last requirement holds for a change that only moves the masking decision.
A change that legitimately changes numerics can pass `--allow-output-change`.
The oracle check still applies in that case.

## Recorded result

FlashInfer #9846 at `ba0560642`, against main `253986e2dc`. NVIDIA B300 SXM6
AC, CUDA 13.0, five interleaved rounds:

- **Verdict:** PASS. All 33 cases pass the oracle, and every output is bitwise
  identical to main.
- **Speedups:**
  - fixed sliding windows 1024 and 4000: −20.1% and −23.7%;
  - variable window 2048: −15.6%;
  - packed sliding window: −22.2%;
  - context-parallel packed-Q MLA: −2.1%;
  - context-parallel MLA MTP: −1.2%.
- **Paged shifted causal at D256:** −0.2% to +0.0%.
- **Worst case:** +0.9% (MLA H16 packed-Q causal).
- **Geomean:** −2.98%.

The suite also caught a regression during review: an intermediate commit that
rebuilt the D256 TAIL causal mask measured +1.6% to +1.9% on the D256 cases,
and was replaced before the recorded run.
