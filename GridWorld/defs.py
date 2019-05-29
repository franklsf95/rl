#!/usr/bin/env python3

from enum import Enum
from typing import Dict, Tuple

State = Tuple[int, int]
Reward = float
Value = float
Prob = float
ValueFn = Dict[State, Value]


class Action(Enum):
    Up = "U"
    Down = "D"
    Left = "L"
    Right = "R"
    Unknown = "?"

    def __repr__(self):
        return str(self.value)


EPSILON = 1e-4  # Convergence threshold
