#!/usr/bin/env python3

import numpy as np

from defs import *
from env import Env


class Player(object):
    """
    By default, a player plays random moves on the chess board.
    """

    def __init__(self, color: Color):
        self.color = color  # color of the player, X or O

    def __str__(self):
        return f"Player {COLORS[self.color]}"

    def act(self, env: Env):
        # Place a piece randomly on the board
        for _ in range(D * D):
            if self.__act_once(env):
                return

    def __act_once(self, env: Env) -> bool:
        i = np.random.randint(0, D)
        j = np.random.randint(0, D)
        return env.place(self.color, i, j)

