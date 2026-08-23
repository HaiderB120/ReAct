from __future__ import annotations

from enum import Enum


class Command(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FORWARD = "FORWARD"
    BACK = "BACK"
    STOP = "STOP"


OPPOSITE_COMMAND = {
    Command.LEFT: Command.RIGHT,
    Command.RIGHT: Command.LEFT,
    Command.FORWARD: Command.BACK,
    Command.BACK: Command.FORWARD,
}


def command_from_label(label: str | None) -> Command | None:
    """Convert a trained label such as 'left' into a ReAct command."""
    if not label:
        return None

    try:
        return Command(label.strip().upper())
    except ValueError:
        return None
