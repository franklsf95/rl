#!/usr/bin/env python3

import logging
import numpy as np
from typing import Iterable, Optional, Tuple

from defs import *
from env import Env
from player import Player


class Agent(Player):
    """
    A trainable reinforcement-learning agent.
    """

    def __init__(self, color: Color, alpha=0.5, epsilon=0.1):
        super().__init__(color)
        self.color = color  # color of the player, X or O
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # probability of choosing to explore (random action)
        self.history: List[State] = []  # history of states in the current game
        self.__initialize_value()

    def __str__(self):
        return f"Agent {COLORS[self.color]}"

    def record_state(self, state: State):
        self.history.append(state)

    def reset_state_history(self):
        self.history = []

    def update_value(self):
        # Learn.
        pass

    # def act(self, env: Env):
    #     pass

    def __initialize_value(self):
        logging.info(f"{self} initializing value function.")
        self.value = np.zeros(N_STATES + 1)
        env = Env()
        for state, game_over, winner in _fill(env, 0, 0):
            self.value[state] = self.__get_initial_value(game_over, winner)

    def __get_initial_value(self, game_over: bool, winner: Optional[Color]) -> float:
        if not game_over:
            return 0.5
        return 1 if winner == self.color else 0


def _fill(env: Env, i: int, j: int) -> Iterable[Tuple[State, bool, Optional[Color]]]:
    if j == D:
        yield from _fill(env, i + 1, 0)
        return
    if i == D:
        # We're done filling the board
        game_over = env.is_game_over(force=True)
        winner = env.winner
        yield env.get_state(), game_over, winner
        return
    for c in [0, X, O]:
        env.place(c, i, j)
        yield from _fill(env, i, j + 1)
