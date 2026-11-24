# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only checks for the numerical accuracy gate selection and disposition."""

from pathlib import Path

import pytest
from run_accuracy import (
    CONTEXT_TEST,
    DECODE_MIXED_PRECISION_TEST,
    DECODE_TEST,
    MASK_TEST,
    MASK_TEST_NODES,
    MLA_DECODE_TEST,
    _accuracy_gate_returncode,
    _accuracy_tests,
    main,
)

FEATURES_TEST = MASK_TEST_NODES[0].partition("::")[0]


def _checkout(root: Path, *files: str) -> Path:
    (root / ".git").mkdir(parents=True)
    for name in (CONTEXT_TEST, DECODE_TEST, MLA_DECODE_TEST, *files):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ({"tests": 360, "failures": 0, "errors": 0, "skipped": 0}, 0),
        ({"tests": 359, "failures": 0, "errors": 0, "skipped": 0}, 3),
        ({"tests": 360, "failures": 0, "errors": 0, "skipped": 1}, 3),
        (None, 3),
    ],
)
def test_accuracy_gate_requires_complete_unskipped_run(summary, expected: int) -> None:
    assert _accuracy_gate_returncode(0, summary, allow_partial=False) == expected


def test_accuracy_gate_preserves_pytest_failure() -> None:
    assert _accuracy_gate_returncode(1, None, allow_partial=True) == 1


def test_allow_partial_is_explicitly_diagnostic() -> None:
    summary = {"tests": 1, "failures": 0, "errors": 0, "skipped": 1}
    assert _accuracy_gate_returncode(0, summary, allow_partial=True) == 0


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        # Before FlashInfer #6529 moved the mask predicates.
        ((MASK_TEST,), [CONTEXT_TEST, DECODE_TEST, MASK_TEST, MLA_DECODE_TEST]),
        # From #6529: the two mask tests only, not the other feature tests.
        (
            (FEATURES_TEST,),
            [CONTEXT_TEST, DECODE_TEST, *MASK_TEST_NODES, MLA_DECODE_TEST],
        ),
        # From #9897: the mixed-precision decode grid has its own file.
        (
            (FEATURES_TEST, DECODE_MIXED_PRECISION_TEST),
            [
                CONTEXT_TEST,
                DECODE_TEST,
                DECODE_MIXED_PRECISION_TEST,
                *MASK_TEST_NODES,
                MLA_DECODE_TEST,
            ],
        ),
    ],
)
def test_accuracy_tests_follow_the_checkout_layout(
    tmp_path: Path, files: tuple[str, ...], expected: list[str]
) -> None:
    assert _accuracy_tests(_checkout(tmp_path, *files)) == expected


def test_list_prints_the_resolved_targets(tmp_path: Path, capsys) -> None:
    root = _checkout(tmp_path / "src", FEATURES_TEST, DECODE_MIXED_PRECISION_TEST)
    argv = ["--source-root", str(root), "--output-dir", str(tmp_path / "out")]
    assert main([*argv, "--list"]) == 0
    assert capsys.readouterr().out.split() == _accuracy_tests(root)


def test_missing_node_file_is_rejected(tmp_path: Path) -> None:
    root = _checkout(tmp_path / "src")
    argv = ["--source-root", str(root), "--output-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="missing test files"):
        main(argv)
