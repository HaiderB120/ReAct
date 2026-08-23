from __future__ import annotations

from dataclasses import dataclass

from .commands import Command


@dataclass(frozen=True)
class RobotExecution:
    command: Command


class SimulatedRobot:
    """Development robot used before the Tello is available."""

    def __init__(self) -> None:
        self.last_command = Command.STOP

    def execute(self, command: Command) -> RobotExecution:
        self.last_command = command
        print(f"[SIMULATED ROBOT] {command.value}")
        return RobotExecution(command=command)
