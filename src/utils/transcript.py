from __future__ import annotations

import atexit
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _as_text(value: Any) -> str:
    """Convert arbitrary metadata values to compact text."""
    if isinstance(value, (dict, list, tuple)):
        return repr(value)
    return str(value)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class TranscriptWriter:
    """Create and append to transcript files for agent conversations."""

    def __init__(
        self,
        *,
        base_dir: Path | str = "transcripts",
        metadata: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._metadata = metadata
        self._started_at = timestamp or datetime.now()

        day_dir = self._base_dir / self._started_at.strftime("%Y-%m-%d")
        _ensure_directory(day_dir)

        basename = self._started_at.strftime("%H%M%S")
        candidate = day_dir / f"{basename}.txt"
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = day_dir / f"{basename}-{suffix}.txt"

        self._path = candidate
        self._file = self._path.open("a", encoding="utf-8")
        atexit.register(self._file.close)

        self._write_metadata()

    @property
    def path(self) -> Path:
        return self._path

    def log_user(self, text: str) -> None:
        self._write_block("USER", text)

    def log_agent(self, text: str) -> None:
        self._write_block("AGENT", text)

    def log_tool_call(self, name: str, arguments: str) -> None:
        snippet = arguments.strip()
        self._write_block("TOOL CALL", f"{name} {snippet}" if snippet else name)

    def log_tool_result(self, call_id: str, result: str) -> None:
        self._write_block("TOOL RESULT", f"{call_id} {result}")

    def _write_metadata(self) -> None:
        lines = [
            "# Meta",
            f"started_at: {self._started_at.isoformat(timespec='seconds')}",
        ]
        for key, value in sorted(self._metadata.items()):
            lines.append(f"{key}: {_as_text(value)}")
        lines.append("---")
        self._file.write("\n".join(lines) + "\n")
        self._file.flush()

    def _write_block(self, label: str, text: str) -> None:
        stripped = (text or "").strip()
        if not stripped:
            return

        lines = stripped.splitlines()
        first_line = lines[0]
        self._file.write(f"[{label}] {first_line}\n")
        for line in lines[1:]:
            self._file.write(f"    {line}\n")
        self._file.flush()
