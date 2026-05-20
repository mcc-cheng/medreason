"""Frozen prompts for medreason.

Every .txt file in this directory is a frozen prompt whose SHA256 lives in
PROMPTS_LOCK.json. The eval harness refuses to score a test-split run if
the lock is stale — this is the tripwire that prevents silent prompt drift
from invalidating benchmark results.

Local iteration: set MEDREASON_BYPASS_PROMPTS_LOCK=1. Never commit benchmark
results produced with the bypass set.
"""

from .lock import (
    LOCK_PATH,
    PROMPTS_DIR,
    PromptsLockError,
    compute_prompt_hashes,
    load_prompt,
    verify_lock,
    write_lock,
)

__all__ = [
    "LOCK_PATH",
    "PROMPTS_DIR",
    "PromptsLockError",
    "compute_prompt_hashes",
    "load_prompt",
    "verify_lock",
    "write_lock",
]
