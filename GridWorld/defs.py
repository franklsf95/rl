#!/usr/bin/env python3

from enum import Enum
from typing import Tuple

State = Tuple[int, int]

Reward = float


class Action(Enum):
    Up = "U"
    Down = "D"
    Left = "L"
    Right = "R"

    def __repr__(self):
        return str(self.value)


EPSILON = 1e-4  # Convergence threshold
