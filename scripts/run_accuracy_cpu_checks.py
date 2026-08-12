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

"""CPU-only checks for the numerical accuracy gate disposition."""

import pytest
from run_accuracy import _accuracy_gate_returncode


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
