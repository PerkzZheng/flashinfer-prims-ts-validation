#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compare interleaved base/candidate mask-tile results.

Pass the base and candidate result files in round order. The report gates on
the median paired latency ratio per case (default limit +1%), bitwise output
equality and oracle status, and exits nonzero on any failure.
"""

import argparse
import json
from pathlib import Path

from attention_ts_decode_artifacts import atomic_write_json, atomic_write_text
from mask_tiles_suite import compare_runs, comparison_markdown


def summarize(
    base_files, candidate_files, prefix, *, max_regression, allow_output_change
):
    base = [json.loads(Path(p).read_text()) for p in base_files]
    cand = [json.loads(Path(p).read_text()) for p in candidate_files]
    result = compare_runs(
        base,
        cand,
        max_regression=max_regression,
        allow_output_change=allow_output_change,
    )
    report = comparison_markdown(result)
    if prefix is not None:
        prefix = Path(prefix)
        atomic_write_json(prefix.with_suffix(".json"), result)
        atomic_write_text(prefix.with_suffix(".md"), report)
    print(report, end="")
    return 0 if result["passed"] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, nargs="+", required=True)
    p.add_argument("--candidate", type=Path, nargs="+", required=True)
    p.add_argument("--output-prefix", type=Path, help="Write PREFIX.json and PREFIX.md")
    p.add_argument("--max-regression", type=float, default=0.01)
    p.add_argument("--allow-output-change", action="store_true")
    args = p.parse_args(argv)
    return summarize(
        args.base,
        args.candidate,
        args.output_prefix,
        max_regression=args.max_regression,
        allow_output_change=args.allow_output_change,
    )


if __name__ == "__main__":
    raise SystemExit(main())
