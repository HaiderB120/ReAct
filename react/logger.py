from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SessionLogger:
    """Append-only JSONL logger for ReAct experiment events."""

    def __init__(self, log_dir: str | Path | None = None) -> None:
        directory = Path(log_dir) if log_dir else PROJECT_ROOT / "react_logs"
        directory.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = directory / f"react_session_{stamp}.jsonl"

    def log(self, event_type: str, **fields: Any) -> None:
        row = {
            "timestamp_unix": time.time(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **fields,
        }

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")
