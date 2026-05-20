"""Prompts lock — freeze every prompt file's SHA256 so the eval harness
can refuse to score a test run against a mutated prompt.

Contract:
- Every .txt file directly under medreason/prompts/ is a tracked prompt.
- PROMPTS_LOCK.json maps `relative_path -> sha256_hex`.
- verify_lock() raises PromptsLockError if any tracked file drifts, if a
  new .txt file exists that isn't in the lock, or if a locked file has
  been deleted.
- write_lock() regenerates the lock file from the current directory. Use
  it during development when intentionally updating a prompt. Never call
  it programmatically from the eval harness.
- The env var MEDREASON_BYPASS_PROMPTS_LOCK=1 suppresses the enforcement.
  Tests assert that the bypass works ONLY when explicitly set.

SHA256 is computed over the file contents as bytes, after normalizing
Windows CRLF to LF so Git's autocrlf doesn't flip the lock on checkout.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
LOCK_PATH = PROMPTS_DIR / "PROMPTS_LOCK.json"
BYPASS_ENV = "MEDREASON_BYPASS_PROMPTS_LOCK"


class PromptsLockError(RuntimeError):
    """Raised when the frozen prompts lock does not match the directory."""


def _normalize_bytes(raw: bytes) -> bytes:
    """Normalize line endings so Windows CRLF doesn't flip the hash."""
    return raw.replace(b"\r\n", b"\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(_normalize_bytes(path.read_bytes()))
    return h.hexdigest()


def _iter_prompt_files() -> list[Path]:
    """Every tracked prompt file. Currently: *.txt directly under prompts/."""
    return sorted(p for p in PROMPTS_DIR.glob("*.txt") if p.is_file())


def compute_prompt_hashes() -> dict[str, str]:
    """Return the current `relative_name -> sha256` mapping for every prompt."""
    return {p.name: _sha256(p) for p in _iter_prompt_files()}


def _load_lock_mapping() -> dict[str, str]:
    if not LOCK_PATH.exists():
        raise PromptsLockError(
            f"PROMPTS_LOCK.json missing at {LOCK_PATH}. Run write_lock() to generate."
        )
    try:
        data = json.loads(LOCK_PATH.read_text())
    except json.JSONDecodeError as e:
        raise PromptsLockError(f"PROMPTS_LOCK.json is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise PromptsLockError(
            "PROMPTS_LOCK.json must be an object of {filename: sha256}"
        )
    return {str(k): str(v) for k, v in data.items()}


def verify_lock(*, allow_bypass: bool = True) -> dict[str, str]:
    """Verify the lock matches the current directory.

    Returns the current hash map on success. Raises PromptsLockError with
    a specific human-readable reason on mismatch. If `allow_bypass` is
    True and the env var MEDREASON_BYPASS_PROMPTS_LOCK is set to "1",
    the check is skipped and the current hashes are returned.

    The eval harness should call `verify_lock(allow_bypass=False)` before
    scoring any test-split run, so the bypass is literally unreachable at
    eval time.
    """
    current = compute_prompt_hashes()

    if allow_bypass and os.environ.get(BYPASS_ENV) == "1":
        return current

    locked = _load_lock_mapping()
    locked_names = set(locked.keys())
    current_names = set(current.keys())

    missing_on_disk = sorted(locked_names - current_names)
    if missing_on_disk:
        raise PromptsLockError(
            f"Locked prompt(s) missing on disk: {missing_on_disk}"
        )

    untracked = sorted(current_names - locked_names)
    if untracked:
        raise PromptsLockError(
            f"New prompt file(s) not in PROMPTS_LOCK.json: {untracked}. "
            "Run write_lock() if this is intentional."
        )

    drifted: list[tuple[str, str, str]] = []
    for name, expected in locked.items():
        actual = current[name]
        if actual != expected:
            drifted.append((name, expected, actual))
    if drifted:
        lines = [
            f"  {name}: locked={expected[:12]}…  actual={actual[:12]}…"
            for name, expected, actual in drifted
        ]
        raise PromptsLockError(
            "Prompt file(s) differ from PROMPTS_LOCK.json:\n" + "\n".join(lines)
            + "\nIf the change is intentional, run write_lock() and bump "
            "the leaderboard's prompts_lock_sha field."
        )

    return current


def write_lock() -> dict[str, str]:
    """Regenerate PROMPTS_LOCK.json from the current directory.

    Only call this during development when intentionally updating a prompt.
    The eval harness must NEVER call this — the whole point of the lock is
    that it cannot move silently.
    """
    current = compute_prompt_hashes()
    LOCK_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return current


def load_prompt(name: str) -> str:
    """Read a prompt file and return its text content.

    Does NOT verify the lock — call verify_lock() separately at the
    harness level before any scored run. This keeps prompt loading cheap
    for internal, non-scored operations.
    """
    path = PROMPTS_DIR / name
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text()
