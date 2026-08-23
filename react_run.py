from __future__ import annotations

import argparse

import cv2
import mediapipe as mp

from react.commands import command_from_label
from react.experiment import ExperimentController
from react.logger import SessionLogger
from react.perception import TemplateGestureRecognizer
from react.robot_simulator import SimulatedRobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReAct with a simulated robot."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--confirm-frames", type=int, default=4)
    parser.add_argument("--release-frames", type=int, default=4)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    recognizer = TemplateGestureRecognizer(threshold=args.threshold)
    if not recognizer.labels:
        raise RuntimeError(
            "No ReAct command gestures are trained yet. "
            "Run: python react_train.py left"
        )

    logger = SessionLogger()
    robot = SimulatedRobot()
    controller = ExperimentController(
        robot=robot,
        logger=logger,
        error_rate=args.error_rate,
        seed=args.seed,
    )

    print("[ReAct] Loaded labels:", recognizer.labels)
    print("[ReAct] Log:", logger.path)
    print("[ReAct] Press Q to quit.")

    hands_module = mp.solutions.hands
    draw = mp.solutions.drawing_utils

    hands = hands_module.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    candidate_label: str | None = None
    candidate_frames = 0
    release_frames = args.release_frames
    armed = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            accepted_label: str | None = None
            score = 0.0
            best_label: str | None = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    hands_module.HAND_CONNECTIONS,
                )

                prediction = recognizer.predict(
                    hand_landmarks,
                    width,
                    height,
                )
                accepted_label = prediction.label
                score = prediction.score
                best_label = prediction.best_label

            command = command_from_label(accepted_label)

            if command is not None:
                release_frames = 0

                if accepted_label == candidate_label:
                    candidate_frames += 1
                else:
                    candidate_label = accepted_label
                    candidate_frames = 1

                if armed and candidate_frames >= args.confirm_frames:
                    controller.handle_command(
                        command,
                        gesture_label=accepted_label,
                        gesture_score=score,
                    )
                    armed = False
            else:
                candidate_label = None
                candidate_frames = 0
                release_frames += 1

                if release_frames >= args.release_frames:
                    armed = True

            cv2.putText(
                frame,
                "ReAct - simulated robot",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Best: {best_label or '-'} ({score:.2f})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"State: {'ARMED' if armed else 'WAITING FOR RELEASE'}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                f"Error injection: {args.error_rate:.0%}",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow("ReAct", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    logger.log("session_end")
    print("[ReAct] Session ended.")
    print("[ReAct] Log saved to:", logger.path)


if __name__ == "__main__":
    main()
