"""Manifest writer + verifier.

Emits, per version:
    <out_dir>/train.jsonl
    <out_dir>/dev.jsonl
    <out_dir>/test.jsonl
    <out_dir>/fingerprints.json

The fingerprints file is the thing LeakGuard loads. Its schema is
exactly what leak_guard.LeakGuard.from_fingerprint_file() expects:

    {
      "train": {"case_0001": "<sha256>", ...},
      "dev":   {"case_0011": "<sha256>", ...},
      "test":  {"case_0013": "<sha256>", ...}
    }

Canonical fingerprint: sha256 over `json.dumps(case.model_dump(mode="json"),
sort_keys=True, separators=(",",":"))`. This:
- Uses mode="json" so enums serialize to their string values, not names.
- Sorts keys so insertion order doesn't affect the hash.
- Uses compact separators so the bytes are stable.

The jsonl files themselves also use that canonical form per line, so a
byte-diff of the output directory is the fastest way to spot manifest drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from medreason.ontology import BenchmarkCase

from .stratify import MANIFEST_SPLITS


class ManifestError(RuntimeError):
    """Raised when a manifest cannot be written or verified."""


# ── Canonicalization ─────────────────────────────────────────────────────────


def _canonical_bytes(case: BenchmarkCase) -> bytes:
    data = case.model_dump(mode="json")
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_case_fingerprint(case: BenchmarkCase) -> str:
    """SHA256 hex of the case's canonical JSON form."""
    return hashlib.sha256(_canonical_bytes(case)).hexdigest()


# ── Writers ──────────────────────────────────────────────────────────────────


def write_manifest(
    splits: Mapping[str, list[BenchmarkCase]],
    out_dir: Path | str,
) -> dict[str, dict[str, str]]:
    """Write <split>.jsonl + fingerprints.json. Returns the fingerprint map.

    Overwrites any existing files in `out_dir`. Creates `out_dir` if it
    doesn't exist. Raises ManifestError if a split name is unknown or if
    case_ids are duplicated across splits (which is a fatal contamination
    bug, not a recoverable warning).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name in splits:
        if split_name not in MANIFEST_SPLITS:
            raise ManifestError(
                f"Unknown split {split_name!r}. Expected one of {MANIFEST_SPLITS}."
            )

    # Duplicate case_id check across splits — a leak in itself.
    seen: dict[str, str] = {}
    for split_name, cases in splits.items():
        for case in cases:
            if case.case_id in seen and seen[case.case_id] != split_name:
                raise ManifestError(
                    f"case_id {case.case_id!r} appears in both "
                    f"{seen[case.case_id]!r} and {split_name!r}"
                )
            seen[case.case_id] = split_name

    fingerprints: dict[str, dict[str, str]] = {s: {} for s in MANIFEST_SPLITS}

    for split_name in MANIFEST_SPLITS:
        cases = sorted(splits.get(split_name, []), key=lambda c: c.case_id)
        jsonl_path = out / f"{split_name}.jsonl"
        with jsonl_path.open("wb") as f:
            for case in cases:
                line = _canonical_bytes(case)
                f.write(line)
                f.write(b"\n")
                fingerprints[split_name][case.case_id] = hashlib.sha256(line).hexdigest()

    fp_path = out / "fingerprints.json"
    fp_path.write_text(
        json.dumps(fingerprints, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return fingerprints


def verify_manifest(out_dir: Path | str) -> dict[str, dict[str, str]]:
    """Re-hash every case in every split and assert the manifest matches.

    Returns the (verified) fingerprints map on success. Raises
    ManifestError with a specific reason on any drift.

    This is what the `medreason-bench splits verify` CLI invokes.
    """
    out = Path(out_dir)
    fp_path = out / "fingerprints.json"
    if not fp_path.exists():
        raise ManifestError(f"fingerprints.json missing in {out}")

    try:
        locked: dict[str, dict[str, str]] = json.loads(fp_path.read_text())
    except json.JSONDecodeError as e:
        raise ManifestError(f"fingerprints.json is not valid JSON: {e}") from e

    computed: dict[str, dict[str, str]] = {s: {} for s in MANIFEST_SPLITS}

    for split_name in MANIFEST_SPLITS:
        jsonl_path = out / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            if locked.get(split_name):
                raise ManifestError(
                    f"{split_name}.jsonl missing but fingerprints.json "
                    f"lists {len(locked[split_name])} case(s) for it"
                )
            continue

        with jsonl_path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                # Don't trust the order of keys on the line; re-hash the
                # raw bytes directly. The manifest writer guarantees the
                # jsonl file stores the canonical form, so this is a
                # tamper-detection check.
                line_bytes = raw.rstrip(b"\n")
                data = json.loads(line_bytes.decode("utf-8"))
                case_id = data.get("case_id")
                if not case_id:
                    raise ManifestError(
                        f"{split_name}.jsonl contains a line with no case_id"
                    )
                computed[split_name][case_id] = hashlib.sha256(line_bytes).hexdigest()

    # Compare
    for split_name in MANIFEST_SPLITS:
        locked_split = locked.get(split_name, {})
        computed_split = computed.get(split_name, {})

        missing = sorted(set(locked_split) - set(computed_split))
        if missing:
            raise ManifestError(
                f"{split_name}: case_id(s) in fingerprints.json but missing "
                f"from {split_name}.jsonl: {missing}"
            )
        extra = sorted(set(computed_split) - set(locked_split))
        if extra:
            raise ManifestError(
                f"{split_name}: case_id(s) in {split_name}.jsonl but not in "
                f"fingerprints.json: {extra}"
            )
        drifted = [
            cid for cid in locked_split
            if locked_split[cid] != computed_split[cid]
        ]
        if drifted:
            raise ManifestError(
                f"{split_name}: fingerprint drift for case_id(s) {drifted}"
            )

    return computed


def load_split(out_dir: Path | str, split: str) -> list[BenchmarkCase]:
    """Read a split's jsonl back into BenchmarkCase objects.

    Used by the eval harness and by test fixtures. Does NOT verify the
    manifest — call verify_manifest() separately if you need that.
    """
    if split not in MANIFEST_SPLITS:
        raise ManifestError(
            f"Unknown split {split!r}. Expected one of {MANIFEST_SPLITS}."
        )
    p = Path(out_dir) / f"{split}.jsonl"
    if not p.exists():
        raise ManifestError(f"Split file not found: {p}")
    cases: list[BenchmarkCase] = []
    with p.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            cases.append(BenchmarkCase.model_validate_json(raw))
    return cases
