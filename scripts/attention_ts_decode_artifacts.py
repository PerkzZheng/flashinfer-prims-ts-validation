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

"""Filesystem and source-control helpers for reproducible benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


def cache_info_record(info: Any) -> dict[str, int | None]:
    """Convert a ``functools`` cache record into stable JSON fields."""

    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
    }


def effective_command(script_path: Path | str, argv: Sequence[str] | None) -> str:
    """Render the actual script arguments, including programmatic ``main`` calls."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    return shlex.join([sys.executable, str(Path(script_path).resolve()), *arguments])


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` using a unique sibling temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a human-readable JSON object with atomic replacement."""

    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def sha256_file(path: Path | str) -> str | None:
    """Return the SHA-256 digest of a file, or ``None`` when absent."""

    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(
    root: Path | str,
    *,
    suffixes: Sequence[str] = (".py", ".toml", ".json"),
) -> str:
    """Hash relative paths and contents for selected files below ``root``."""

    root = Path(root)
    suffix_set = set(suffixes)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in suffix_set:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def git_value(repo_root: Path | str, *args: str) -> str | None:
    """Run a read-only Git query against an explicit checkout root."""

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()
