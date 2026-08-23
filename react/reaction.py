from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .experiment import CommandDecision
from .features import landmarks_to_normalized_xyz
from .logger import SessionLogger


@dataclass
class ReactionWindowState:
    trial_id: int
    decision: CommandDecision
    started_at: float
    frame_index: int = 0


class ReactionRecorder:
    """Record the human's visual response immediately after a robot action."""

    def __init__(
        self,
        logger: SessionLogger,
        duration_sec: float = 2.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if duration_sec <= 0.0:
            raise ValueError("duration_sec must be greater than zero")

        self.logger = logger
        self.duration_sec = duration_sec
        self.clock = clock
        self._state: ReactionWindowState | None = None

    @property
    def active(self) -> bool:
        return self._state is not None

    @property
    def remaining_sec(self) -> float:
        if self._state is None:
            return 0.0

        elapsed = self.clock() - self._state.started_at
        return max(0.0, self.duration_sec - elapsed)

    def start(self, trial_id: int, decision: CommandDecision) -> None:
        if self.active:
            raise RuntimeError("A reaction window is already active")

        started_at = self.clock()
        self._state = ReactionWindowState(
            trial_id=trial_id,
            decision=decision,
            started_at=started_at,
        )

        self.logger.log(
            "reaction_window_start",
            trial_id=trial_id,
            intended_command=decision.intended.value,
            executed_command=decision.executed.value,
            error_injected=decision.error_injected,
            duration_sec=self.duration_sec,
        )

    def record_frame(
        self,
        *,
        hand_landmarks=None,
        feature_vector: np.ndarray | None = None,
        accepted_label: str | None = None,
        best_label: str | None = None,
        gesture_score: float | None = None,
    ) -> bool:
        """
        Record one frame from the active reaction window.

        Returns True when the frame was recorded. If the observation duration has
        elapsed, the window is closed and False is returned.
        """
        if self._state is None:
            return False

        now = self.clock()
        elapsed = now - self._state.started_at

        if elapsed > self.duration_sec:
            self.end(reason="completed")
            return False

        hand_visible = hand_landmarks is not None
        landmarks_xyz = (
            landmarks_to_normalized_xyz(hand_landmarks)
            if hand_visible
            else None
        )
        feature_list = (
            [float(value) for value in feature_vector.tolist()]
            if feature_vector is not None
            else None
        )

        self.logger.log(
            "reaction_frame",
            trial_id=self._state.trial_id,
            frame_index=self._state.frame_index,
            elapsed_since_action_sec=elapsed,
            hand_visible=hand_visible,
            landmarks_xyz=landmarks_xyz,
            feature_vector=feature_list,
            accepted_gesture=accepted_label,
            best_gesture=best_label,
            gesture_score=gesture_score,
            intended_command=self._state.decision.intended.value,
            executed_command=self._state.decision.executed.value,
            error_injected=self._state.decision.error_injected,
        )
        self._state.frame_index += 1

        if elapsed >= self.duration_sec:
            self.end(reason="completed")

        return True

    def end(self, reason: str = "completed") -> None:
        if self._state is None:
            return

        elapsed = self.clock() - self._state.started_at

        self.logger.log(
            "reaction_window_end",
            trial_id=self._state.trial_id,
            frames_recorded=self._state.frame_index,
            elapsed_sec=elapsed,
            reason=reason,
        )

        self._state = None
