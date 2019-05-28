#!/usr/bin/env python3

from enum import Enum

State = (int, int)

Reward = float


class Action(Enum):
    Up = "U"
    Down = "D"
    Left = "L"
    Right = "R"

    def __repr__(self):
        return str(self.value)

