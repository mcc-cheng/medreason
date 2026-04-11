"""medreason.store — persistence layer for the memory pipeline.

New Phase 1 surface:
    - LeakGuard, TestSetLeakError, LikelyTestLeakError   (leak_guard.py)
    - RuleStore                                          (rules.py)
    - TraceStore                                         (traces.py)

Legacy (pre-rework) surface kept for backward compatibility:
    - PatternStore                                       (_legacy_pattern_store.py)

`PatternStore` is re-exported so pre-rework modules
(`medreason.agent`, `medreason.injector`, `medreason.benchmark`) keep
importing `from medreason.store import PatternStore` successfully until
Phase 5 replaces them.
"""

from ._legacy_pattern_store import PatternStore
from .leak_guard import (
    LeakGuard,
    LikelyTestLeakError,
    TestSetLeakError,
)
from .rules import RuleStore
from .traces import TraceStore

__all__ = [
    # New
    "LeakGuard",
    "TestSetLeakError",
    "LikelyTestLeakError",
    "RuleStore",
    "TraceStore",
    # Legacy
    "PatternStore",
]
