"""OpenTimestamps wrapper using the Python API.

The `ots.exe` script bundled with opentimestamps-client fails on
Python 3.14 / Windows with a ctypes LoadLibrary error, so we use the
library directly instead. Submits to the three canonical public
calendars (alice/bob/finney) and writes a `.ots` receipt next to the
target file. The receipt is initially pending — bitcoin attestation
finalizes in ~1-6 hours.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from opentimestamps.calendar import RemoteCalendar
from opentimestamps.core.op import OpSHA256
from opentimestamps.core.serialize import StreamSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp


DEFAULT_CALENDARS = (
    "https://alice.btc.calendar.opentimestamps.org",
    "https://bob.btc.calendar.opentimestamps.org",
    "https://finney.calendar.eternitywall.com",
)


@dataclass
class OTSReceipt:
    file_path: Path
    receipt_path: Path
    sha256_hex: str
    calendars_ok: list[str]
    calendars_failed: list[str]

    @property
    def pending(self) -> bool:
        return bool(self.calendars_ok)


def stamp_file(
    file_path: str | Path,
    *,
    calendars: tuple[str, ...] = DEFAULT_CALENDARS,
) -> OTSReceipt:
    file_path = Path(file_path)
    data = file_path.read_bytes()
    file_hash = hashlib.sha256(data).digest()
    timestamp = Timestamp(file_hash)

    ok: list[str] = []
    failed: list[str] = []
    for url in calendars:
        try:
            cal = RemoteCalendar(url)
            result = cal.submit(file_hash)
            timestamp.merge(result)
            ok.append(url)
        except Exception:
            failed.append(url)

    if not ok:
        raise RuntimeError(
            f"All {len(calendars)} OTS calendars failed for {file_path.name}"
        )

    dtf = DetachedTimestampFile(OpSHA256(), timestamp)
    receipt_path = file_path.with_suffix(file_path.suffix + ".ots")
    with receipt_path.open("wb") as f:
        ctx = StreamSerializationContext(f)
        dtf.serialize(ctx)

    return OTSReceipt(
        file_path=file_path,
        receipt_path=receipt_path,
        sha256_hex=file_hash.hex(),
        calendars_ok=ok,
        calendars_failed=failed,
    )
