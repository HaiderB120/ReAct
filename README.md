# ReAct

ReAct is a research prototype for closed-loop repair in gesture-based human-robot interaction.

## Research question

Can a robot detect that a person is naturally trying to correct a misunderstood gesture command, and use that signal to recover from the interaction error?

## Phase 1: command pipeline

Phase 1 establishes:
- MediaPipe hand landmark extraction
- simple command-gesture recognition
- abstract robot commands
- a simulated robot
- controlled command-error injection
- experiment logging

## Phase 2: post-action reaction capture

After each robot action, ReAct opens a configurable observation window (2 seconds by default). During this window, normal command execution is suspended and the system records the human response frame-by-frame.

Each reaction window is associated with a trial ID and records:
- intended command
- executed command
- whether an error was intentionally injected
- elapsed time since the robot action
- whether a hand is visible
- MediaPipe's 21 normalized x/y/z hand landmarks
- the 42-D normalized hand feature vector
- best/accepted command gesture and similarity score

At this stage the reaction is recorded but **not classified**. The resulting temporal data will be used to develop and evaluate the future correction-intent detector.

## Running the current prototype

Train command gestures, for example:

```bash
python react_train.py left
python react_train.py right
python react_train.py stop
```

Run without deliberate robot errors:

```bash
python react_run.py
```

Force eligible motion commands to execute in the opposite direction for engineering tests:

```bash
python react_run.py --error-rate 1
```

Change the post-action observation duration:

```bash
python react_run.py --reaction-window 2.5
```

Local gesture samples and session logs are intentionally excluded from Git.

The Tello integration, additional human-perception modalities, and correction-intent model are added in later phases.
