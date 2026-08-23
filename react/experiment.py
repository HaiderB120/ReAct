from __future__ import annotations

import random
from dataclasses import dataclass

from .commands import Command, OPPOSITE_COMMAND
from .logger import SessionLogger
from .robot_simulator import SimulatedRobot


@dataclass(frozen=True)
class CommandDecision:
    intended: Command
    executed: Command
    error_injected: bool


class ExperimentController:
    """
    Route intended commands to the robot.

    During development, an optional controlled error rate can deliberately
    execute the opposite motion command. STOP is never corrupted.
    """

    def __init__(
        self,
        robot: SimulatedRobot,
        logger: SessionLogger,
        error_rate: float = 0.0,
        seed: int = 7,
    ) -> None:
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be between 0.0 and 1.0")

        self.robot = robot
        self.logger = logger
        self.error_rate = error_rate
        self.rng = random.Random(seed)

    def handle_command(
        self,
        intended: Command,
        *,
        gesture_label: str | None = None,
        gesture_score: float | None = None,
    ) -> CommandDecision:
        error_injected = (
            intended in OPPOSITE_COMMAND
            and self.rng.random() < self.error_rate
        )

        executed = (
            OPPOSITE_COMMAND[intended]
            if error_injected
            else intended
        )

        self.logger.log(
            "robot_command",
            gesture_label=gesture_label,
            gesture_score=gesture_score,
            intended_command=intended.value,
            executed_command=executed.value,
            error_injected=error_injected,
        )

        self.robot.execute(executed)

        print(
            "[ReAct] "
            f"intended={intended.value} "
            f"executed={executed.value} "
            f"error_injected={error_injected}"
        )

        return CommandDecision(
            intended=intended,
            executed=executed,
            error_injected=error_injected,
        )
