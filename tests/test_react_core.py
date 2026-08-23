import json

from react.commands import Command, command_from_label
from react.experiment import CommandDecision, ExperimentController
from react.logger import SessionLogger
from react.reaction import ReactionRecorder
from react.robot_simulator import SimulatedRobot


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_command_from_label():
    assert command_from_label("left") == Command.LEFT
    assert command_from_label(" STOP ") == Command.STOP
    assert command_from_label("peace") is None
    assert command_from_label(None) is None


def test_forced_error_uses_opposite_command(tmp_path):
    logger = SessionLogger(tmp_path)
    robot = SimulatedRobot()
    controller = ExperimentController(
        robot=robot,
        logger=logger,
        error_rate=1.0,
        seed=1,
    )

    decision = controller.handle_command(Command.LEFT, trial_id=1)

    assert decision.error_injected is True
    assert decision.executed == Command.RIGHT
    assert robot.last_command == Command.RIGHT


def test_stop_is_never_corrupted(tmp_path):
    logger = SessionLogger(tmp_path)
    robot = SimulatedRobot()
    controller = ExperimentController(
        robot=robot,
        logger=logger,
        error_rate=1.0,
        seed=1,
    )

    decision = controller.handle_command(Command.STOP, trial_id=1)

    assert decision.error_injected is False
    assert decision.executed == Command.STOP


def test_reaction_window_records_and_closes(tmp_path):
    clock = FakeClock()
    logger = SessionLogger(tmp_path)
    recorder = ReactionRecorder(
        logger=logger,
        duration_sec=1.0,
        clock=clock,
    )
    decision = CommandDecision(
        intended=Command.LEFT,
        executed=Command.RIGHT,
        error_injected=True,
    )

    recorder.start(trial_id=7, decision=decision)
    assert recorder.active is True

    assert recorder.record_frame(hand_landmarks=None) is True
    clock.advance(0.5)
    assert recorder.record_frame(hand_landmarks=None) is True

    clock.advance(0.6)
    assert recorder.record_frame(hand_landmarks=None) is False
    assert recorder.active is False

    rows = [
        json.loads(line)
        for line in logger.path.read_text(encoding="utf-8").splitlines()
    ]
    events = [row["event_type"] for row in rows]

    assert events == [
        "reaction_window_start",
        "reaction_frame",
        "reaction_frame",
        "reaction_window_end",
    ]
    assert rows[0]["trial_id"] == 7
    assert rows[-1]["frames_recorded"] == 2
