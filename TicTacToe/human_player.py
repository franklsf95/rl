#!/usr/bin/env python3

import numpy as np

from defs import *
from env import Env
from player import Player


class HumanPlayer(Player):
    """
    A human player enters coordinates for each move.
    """

    def act(self, env: Env):
        # Place a piece randomly on the board
        inp = input("> Input coordinates (e.g. '1 2'): ")
        try:
            coords = list(map(int, inp.split()))
            i = coords[0]
            j = coords[1]
        except EOFError:
            print("Bye!")
            exit(0)
        except (IndexError, ValueError) as ex:
            print(f"Invalid input: {ex}.")
            return self.act(env)
        if env.can_place(i, j):
            env.place(self.color, i, j)
        else:
            print("Error: Cannot place at this coordinate.")
            return self.act(env)
