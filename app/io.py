"""Input/output helpers for candidate records."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_candidate_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield candidate records from JSON, JSONL, or JSONL.GZ files."""
    source = Path(path)
    if source.suffix == ".json" and not source.name.endswith(".jsonl"):
        with source.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"{source} must contain a JSON array of candidates")
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    with _open_text(source) as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSONL row") from exc
            if isinstance(item, dict):
                yield item


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL and return the number of rows written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

