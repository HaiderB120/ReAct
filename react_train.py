from __future__ import annotations

import argparse

import cv2
import mediapipe as mp

from react.commands import Command
from react.features import to_feature_vec
from react.perception import (
    DEFAULT_GESTURE_DB,
    load_gesture_db,
    save_gesture_db,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect ReAct command-gesture samples."
    )
    parser.add_argument(
        "command",
        choices=[command.value.lower() for command in Command],
        help="Command gesture to train.",
    )
    parser.add_argument("--target", type=int, default=40)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append instead of replacing existing samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = args.command.lower()

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

    samples: list[list[float]] = []
    save_requested = False

    print(f"Training ReAct command: {label.upper()}")
    print("SPACE = capture sample")
    print("ENTER = save and exit")
    print("Q = abort without saving")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    hands_module.HAND_CONNECTIONS,
                )
                status = f"{label.upper()} samples: {len(samples)}/{args.target}"
            else:
                status = "No hand detected"

            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "SPACE capture | ENTER save | Q abort",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow("ReAct - Train Command Gesture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Training aborted. Nothing saved.")
                break

            if key == 13:
                if not samples:
                    print("No samples captured yet.")
                    continue
                save_requested = True
                break

            if key == 32:
                if not results.multi_hand_landmarks:
                    print("No hand detected; sample not captured.")
                    continue

                hand_landmarks = results.multi_hand_landmarks[0]
                feature = to_feature_vec(hand_landmarks, width, height)
                samples.append(feature.tolist())
                print(f"Captured sample {len(samples)}")

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    if not save_requested:
        return

    db = load_gesture_db(DEFAULT_GESTURE_DB)

    if args.append and label in db:
        samples = list(db[label].get("samples", [])) + samples

    db[label] = {"samples": samples}
    save_gesture_db(db, DEFAULT_GESTURE_DB)

    print(
        f"Saved {len(samples)} samples for {label.upper()} "
        f"to {DEFAULT_GESTURE_DB.name}"
    )


if __name__ == "__main__":
    main()
