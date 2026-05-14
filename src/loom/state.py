"""Checkpoint and resume for Loom sessions.

State is just the messages list. Serialize it to JSON after every turn
so a killed session can resume from the last safe point.
"""

import json
from datetime import datetime
from pathlib import Path

from google.genai import types

CHECKPOINT_DIR = Path(".loom")
CHECKPOINT_FILE = CHECKPOINT_DIR / "session.json"


def save(
    messages: list[types.Content],
    model: str,
    max_tokens: int,
    compact_threshold: float,
) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    data = {
        "version": 1,
        "model": model,
        "max_tokens": max_tokens,
        "compact_threshold": compact_threshold,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        # mode='json' converts enums and other non-serializable types to
        # plain Python values so json.dumps doesn't choke.
        "messages": [m.model_dump(mode="json") for m in messages],
    }
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))


def load() -> dict | None:
    """Return checkpoint data if a session file exists, else None."""
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        return json.loads(CHECKPOINT_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def restore(data: dict) -> list[types.Content]:
    """Reconstruct Content objects from checkpoint data."""
    return [types.Content.model_validate(m) for m in data["messages"]]


def clear() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
