"""Shared constants and small helpers for HW3 rescue robot.

Students should not need to edit this file.
"""

from __future__ import annotations

from typing import Literal, Tuple

Action = Literal["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]
State = Tuple[int, int, bool, int, int]

ACTIONS: tuple[Action, ...] = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
MOVE_ACTIONS: tuple[Action, ...] = ("UP", "DOWN", "LEFT", "RIGHT")

ACTION_TO_DELTA: dict[Action, tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "WAIT": (0, 0),
}

LEFT_OF: dict[Action, Action] = {
    "UP": "LEFT",
    "DOWN": "RIGHT",
    "LEFT": "DOWN",
    "RIGHT": "UP",
    "WAIT": "WAIT",
}

RIGHT_OF: dict[Action, Action] = {
    "UP": "RIGHT",
    "DOWN": "LEFT",
    "LEFT": "UP",
    "RIGHT": "DOWN",
    "WAIT": "WAIT",
}

HAZARD_TRANSITIONS: dict[str, tuple[tuple[str, float], ...]] = {
    "normal": (("intended", 0.85), ("left", 0.05), ("right", 0.05), ("stay", 0.05)),
    "smoky": (("intended", 0.65), ("left", 0.10), ("right", 0.10), ("stay", 0.15)),
    "slippery": (("intended", 0.55), ("left", 0.20), ("right", 0.20), ("stay", 0.05)),
    "damaged": (("intended", 0.70), ("left", 0.05), ("right", 0.05), ("stay", 0.20)),
}

ARROWS: dict[Action, str] = {
    "UP": "^",
    "DOWN": "v",
    "LEFT": "<",
    "RIGHT": ">",
    "WAIT": ".",
}
