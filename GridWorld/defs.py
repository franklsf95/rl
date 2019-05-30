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


# Threshold for value function interation convergence.
VALUE_CONVERGENCE_EPSILON = 1e-3
# If true, then taking an action will only succeed 1/2 times, and end with a random action the other 1/2 times.
WINDY = True

