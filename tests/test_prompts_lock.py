"""Tests for medreason.prompts.lock — Phase 2.

The lock is the tripwire that prevents silent prompt drift from invalidating
benchmark results. Every one of these tests is a benchmark-integrity guard.
"""

from __future__ import annotations

import json
import os

import pytest


# ── Happy path ────────────────────────────────────────────────────────────────


def test_lock_matches_current_directory():
    """The shipped PROMPTS_LOCK.json must match the files on disk. If this
    ever fails, either a prompt was edited without running write_lock(),
    or the lock file is stale."""
    from medreason.prompts.lock import verify_lock
    hashes = verify_lock(allow_bypass=False)
    assert isinstance(hashes, dict)
    assert "system_pa.txt" in hashes
    # sha256 hex is 64 chars
    assert len(hashes["system_pa.txt"]) == 64


def test_load_prompt_returns_non_empty_system_pa():
    from medreason.prompts.lock import load_prompt
    text = load_prompt("system_pa.txt")
    # Smoke content checks — not asserting the whole prompt so small
    # wording polishes don't require a test edit in addition to a lock bump.
    assert len(text) > 200
    assert "prior authorization" in text.lower()
    assert "json" in text.lower()


def test_load_prompt_missing_raises():
    from medreason.prompts.lock import load_prompt
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist.txt")


# ── Drift detection ───────────────────────────────────────────────────────────


def test_verify_lock_detects_drift(tmp_path, monkeypatch):
    """Modifying a prompt file without running write_lock() must trigger
    PromptsLockError with a message that names the offending file."""
    from medreason.prompts import lock as lock_mod

    # Build a throwaway prompts directory with one file + a lock
    fake_dir = tmp_path / "prompts"
    fake_dir.mkdir()
    f1 = fake_dir / "a.txt"
    f1.write_text("original content\n")
    lock_file = fake_dir / "PROMPTS_LOCK.json"

    monkeypatch.setattr(lock_mod, "PROMPTS_DIR", fake_dir)
    monkeypatch.setattr(lock_mod, "LOCK_PATH", lock_file)

    lock_mod.write_lock()
    lock_mod.verify_lock(allow_bypass=False)  # clean

    # Drift: rewrite the file
    f1.write_text("mutated content\n")
    with pytest.raises(lock_mod.PromptsLockError) as exc:
        lock_mod.verify_lock(allow_bypass=False)
    assert "a.txt" in str(exc.value)


def test_verify_lock_detects_new_file(tmp_path, monkeypatch):
    from medreason.prompts import lock as lock_mod

    fake_dir = tmp_path / "prompts"
    fake_dir.mkdir()
    (fake_dir / "a.txt").write_text("x\n")
    lock_file = fake_dir / "PROMPTS_LOCK.json"

    monkeypatch.setattr(lock_mod, "PROMPTS_DIR", fake_dir)
    monkeypatch.setattr(lock_mod, "LOCK_PATH", lock_file)

    lock_mod.write_lock()

    # Add an untracked file
    (fake_dir / "b.txt").write_text("y\n")
    with pytest.raises(lock_mod.PromptsLockError) as exc:
        lock_mod.verify_lock(allow_bypass=False)
    assert "b.txt" in str(exc.value)


def test_verify_lock_detects_deleted_file(tmp_path, monkeypatch):
    from medreason.prompts import lock as lock_mod

    fake_dir = tmp_path / "prompts"
    fake_dir.mkdir()
    (fake_dir / "a.txt").write_text("x\n")
    (fake_dir / "b.txt").write_text("y\n")
    lock_file = fake_dir / "PROMPTS_LOCK.json"

    monkeypatch.setattr(lock_mod, "PROMPTS_DIR", fake_dir)
    monkeypatch.setattr(lock_mod, "LOCK_PATH", lock_file)

    lock_mod.write_lock()

    # Remove a tracked file
    (fake_dir / "a.txt").unlink()
    with pytest.raises(lock_mod.PromptsLockError) as exc:
        lock_mod.verify_lock(allow_bypass=False)
    assert "a.txt" in str(exc.value)


def test_verify_lock_missing_file_raises(tmp_path, monkeypatch):
    from medreason.prompts import lock as lock_mod

    fake_dir = tmp_path / "prompts"
    fake_dir.mkdir()
    (fake_dir / "a.txt").write_text("x\n")
    missing_lock = fake_dir / "DOES_NOT_EXIST.json"

    monkeypatch.setattr(lock_mod, "PROMPTS_DIR", fake_dir)
    monkeypatch.setattr(lock_mod, "LOCK_PATH", missing_lock)

    with pytest.raises(lock_mod.PromptsLockError) as exc:
        lock_mod.verify_lock(allow_bypass=False)
    assert "missing" in str(exc.value).lower()


# ── Bypass semantics ──────────────────────────────────────────────────────────


def test_bypass_env_var_skips_check_when_allowed(tmp_path, monkeypatch):
    """MEDREASON_BYPASS_PROMPTS_LOCK=1 short-circuits verify_lock ONLY
    when allow_bypass=True."""
    from medreason.prompts import lock as lock_mod

    fake_dir = tmp_path / "prompts"
    fake_dir.mkdir()
    (fake_dir / "a.txt").write_text("x\n")
    lock_file = fake_dir / "PROMPTS_LOCK.json"

    monkeypatch.setattr(lock_mod, "PROMPTS_DIR", fake_dir)
    monkeypatch.setattr(lock_mod, "LOCK_PATH", lock_file)

    lock_mod.write_lock()
    (fake_dir / "a.txt").write_text("mutated\n")  # drift

    # With bypass env set + allow_bypass=True → passes
    monkeypatch.setenv("MEDREASON_BYPASS_PROMPTS_LOCK", "1")
    lock_mod.verify_lock(allow_bypass=True)

    # With bypass env set + allow_bypass=False → still raises
    # (eval harness calls with allow_bypass=False, so the bypass is
    # literally unreachable at eval time)
    with pytest.raises(lock_mod.PromptsLockError):
        lock_mod.verify_lock(allow_bypass=False)


def test_bypass_env_var_ignored_when_not_set(tmp_path, monkeypatch):
    from medreason.prompts import lock as lock_mod

    fake_dir = tmp_path / "prompts"
    fake_dir.mkdir()
    (fake_dir / "a.txt").write_text("x\n")
    lock_file = fake_dir / "PROMPTS_LOCK.json"

    monkeypatch.setattr(lock_mod, "PROMPTS_DIR", fake_dir)
    monkeypatch.setattr(lock_mod, "LOCK_PATH", lock_file)
    monkeypatch.delenv("MEDREASON_BYPASS_PROMPTS_LOCK", raising=False)

    lock_mod.write_lock()
    (fake_dir / "a.txt").write_text("mutated\n")

    with pytest.raises(lock_mod.PromptsLockError):
        lock_mod.verify_lock(allow_bypass=True)


# ── write_lock() round-trip ──────────────────────────────────────────────────


def test_write_lock_is_stable_under_re_run(tmp_path, monkeypatch):
    """write_lock() must produce the same output if called twice with no
    changes in between. Protects against sorting / formatting drift that
    would cause spurious git diffs."""
    from medreason.prompts import lock as lock_mod

    fake_dir = tmp_path / "prompts"
    fake_dir.mkdir()
    (fake_dir / "a.txt").write_text("one\n")
    (fake_dir / "b.txt").write_text("two\n")
    lock_file = fake_dir / "PROMPTS_LOCK.json"

    monkeypatch.setattr(lock_mod, "PROMPTS_DIR", fake_dir)
    monkeypatch.setattr(lock_mod, "LOCK_PATH", lock_file)

    lock_mod.write_lock()
    first = lock_file.read_text()
    lock_mod.write_lock()
    second = lock_file.read_text()
    assert first == second

    data = json.loads(first)
    assert set(data.keys()) == {"a.txt", "b.txt"}
    assert len(data["a.txt"]) == 64


def test_write_lock_normalizes_line_endings(tmp_path, monkeypatch):
    """A CRLF file and an LF file with identical text content must hash
    to the same value. Prevents Windows autocrlf from flipping the lock
    on checkout."""
    from medreason.prompts import lock as lock_mod

    fake_dir_lf = tmp_path / "lf"
    fake_dir_crlf = tmp_path / "crlf"
    fake_dir_lf.mkdir()
    fake_dir_crlf.mkdir()

    (fake_dir_lf / "a.txt").write_bytes(b"line1\nline2\n")
    (fake_dir_crlf / "a.txt").write_bytes(b"line1\r\nline2\r\n")

    monkeypatch.setattr(lock_mod, "PROMPTS_DIR", fake_dir_lf)
    monkeypatch.setattr(lock_mod, "LOCK_PATH", fake_dir_lf / "lock.json")
    lf_hashes = lock_mod.compute_prompt_hashes()

    monkeypatch.setattr(lock_mod, "PROMPTS_DIR", fake_dir_crlf)
    monkeypatch.setattr(lock_mod, "LOCK_PATH", fake_dir_crlf / "lock.json")
    crlf_hashes = lock_mod.compute_prompt_hashes()

    assert lf_hashes["a.txt"] == crlf_hashes["a.txt"]
