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
        locations = []
        for i in range(D):
            for j in range(D):
                if env.can_place(i, j):
                    locations.append((i, j))
        if len(locations) == 0:
            return
        idx = np.random.choice(len(locations))
        i, j = locations[idx]
        env.place(self.color, i, j)
