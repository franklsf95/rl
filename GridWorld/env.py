#!/usr/bin/env python3

import logging
import numpy as np
from typing import Dict, Iterable, List

from defs import *


class Env(object):
    def __init__(self, rows: int, cols: int, start: State):
        self.rows = rows
        self.cols = cols
        self.state = start
        self.rewards: Dict[State, Reward] = {}
        self.actions: Dict[State, List[Action]] = {}

    def available_actions(self) -> List[Action]:
        return self.actions.get(self.state, [])

    def move(self, action: Action) -> Reward:
        if action not in self.actions[self.state]:
            logging.warn(f"Invalid action at {self.state}.")
            return 0
        i, j = self.state
        if action == Action.Up:
            self.state = (i - 1, j)
        elif action == Action.Down:
            self.state = (i + 1, j)
        elif action == Action.Left:
            self.state = (i, j - 1)
        elif action == Action.Right:
            self.state = (i, j + 1)
        self.__assert_valid_state()
        return self.rewards.get(self.state, 0)

    def unmove(self, action: Action):
        i, j = self.state
        if action == Action.Up:
            self.state = (i + 1, j)
        elif action == Action.Down:
            self.state = (i - 1, j)
        elif action == Action.Left:
            self.state = (i, j + 1)
        elif action == Action.Right:
            self.state = (i, j - 1)
        self.__assert_valid_state()

    def __assert_valid_state(self):
        i, j = self.state
        assert i >= 0 and i < self.rows and j >= 0 and j < self.cols

    def is_terminal(self, s: State) -> bool:
        return s not in self.actions

    def is_game_over(self) -> bool:
        return self.is_terminal(self.state)

    def all_states(self) -> List[State]:
        return [(i, j) for j in range(self.cols) for i in range(self.rows)]

    def print(self):
        for i in range(self.rows):
            for j in range(self.cols):
                s = ""
                if self.state == (i, j):
                    # Self
                    s = "o"
                elif (i, j) in self.rewards:
                    # Reward
                    r = self.rewards[(i, j)]
                    if np.abs(r) > 0.5:
                        # Omit trivial penalty step rewards
                        s = f"{r:.0f}"
                    else:
                        s = "_"  # Underscore denotes step penalty
                elif len(self.actions.get((i, j), [])) == 0:
                    # Roadblock
                    s = "[]"
                else:
                    s = "."
                # Now print
                if len(s) == 1:
                    print(f" {s} ", end="")
                else:
                    print(f"{s} ", end="")
            print()

    # Factory methods

    @classmethod
    def standard(cls) -> "instanceof(cls)":
        rows = 3
        cols = 4

        def actions_for(i: int, j: int) -> Iterable[Action]:
            if i > 0:
                yield Action.Up
            if i < rows - 1:
                yield Action.Down
            if j > 0:
                yield Action.Left
            if j < cols - 1:
                yield Action.Right

        env = cls(rows, cols, (2, 0))
        env.rewards = {(0, 3): 1, (1, 3): -1}
        roadblocks = {(1, 1)}
        env.actions = {
            (i, j): list(actions_for(i, j))
            for j in range(cols)
            for i in range(rows)
            if (i, j) not in env.rewards and (i, j) not in roadblocks
        }
        return env

    @classmethod
    def negative(cls, step_reward: float = -0.1) -> "instanceof(cls)":
        env = cls.standard()
        for state in env.actions:
            env.rewards[state] = step_reward
        return env
