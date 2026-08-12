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

"""Reusable infrastructure for Attention-TS comparison benchmarks.

The FMHA and MLA suites deliberately keep their case construction, public
backend calls, correctness rules, and result schemas separate.  This package
owns only mechanics that must be identical across those suites.
"""

from .artifacts import (
    atomic_write_json,
    atomic_write_text,
    cache_info_record,
    effective_command,
    git_value,
    sha256_file,
    sha256_tree,
)
from .timing import (
    ColdL2Scrubber,
    capture_repeated_graph,
    first_call,
    paired_backend_order,
    prepare_cold_l2_scrubber,
    summarize_times,
    time_backend,
    time_paired_cold_l2_cuda_graphs,
    time_paired_cuda_graphs,
)

__all__ = [
    "ColdL2Scrubber",
    "atomic_write_json",
    "atomic_write_text",
    "cache_info_record",
    "capture_repeated_graph",
    "effective_command",
    "first_call",
    "git_value",
    "paired_backend_order",
    "prepare_cold_l2_scrubber",
    "sha256_file",
    "sha256_tree",
    "summarize_times",
    "time_backend",
    "time_paired_cold_l2_cuda_graphs",
    "time_paired_cuda_graphs",
]
