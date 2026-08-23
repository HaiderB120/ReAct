from react.commands import Command, command_from_label
from react.experiment import ExperimentController
from react.logger import SessionLogger
from react.robot_simulator import SimulatedRobot


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

    decision = controller.handle_command(Command.LEFT)

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

    decision = controller.handle_command(Command.STOP)

    assert decision.error_injected is False
    assert decision.executed == Command.STOP
