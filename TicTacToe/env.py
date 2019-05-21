#!/usr/bin/env python3

import numpy as np

from defs import *


class Env(object):
    def __init__(self):
        self.__board = np.zeros((D, D), dtype=np.int)  # the chess board
        self.winner: Optional[Color] = None  # winner of this game
        self.game_over = False  # whether game is ended
        # Note that self.winner == None implies either the game is ongoing or the game ended in a draw.

    def print(self):
        board = self.__board
        colors = [".", "X", "O"]
        for i in range(D):
            for j in range(D):
                print(f"{colors[board[i, j]]} ", end="")
            print()

    def print_big(self):
        board = self.__board
        sep_line = "-" * ((D + 1) * D + 1)
        print(sep_line)
        for i in range(D):
            print("|", end="")
            for j in range(D):
                c = COLORS[board[i, j]]
                print(f" {c} |", end="")
            print()
            print(sep_line)

    def place(self, color: Color, i: int, j: int) -> bool:
        """
        Returns false if the specified location is occupied.
        """
        if self.__board[i, j] != 0:
            return False
        self.__board[i, j] = color
        return True

    def reward(self, color: Color) -> float:
        """
        The environment gives a reward for a player at a given time.
        """
        return 1 if self.winner == color else 0

    def is_game_over(self, force=False) -> bool:
        """
        If force is set, will recalculate the whole board even if game is over.
        """
        if not force and self.game_over:
            return True

        board = self.__board

        def check_win(arr: np.ndarray) -> bool:
            for color in [X, O]:
                if np.all(arr == color):
                    self.winner = color
                    self.game_over = True
                    return True
            return False

        # Check rows
        for i in range(D):
            if check_win(board[i]):
                return True

        # Check columns
        for j in range(D):
            if check_win(board[:, j]):
                return True

        # Check main diagonal
        if check_win(board.diagonal()):
            return True

        # Check antidiagonal
        if check_win(np.fliplr(board).diagonal()):
            return True

        # Otherwise, there is no winner.
        self.winner = None

        # Check draw
        if np.all(board != 0):
            self.game_over = True
            return True

        # Otherwise, the game is still ongoing.
        return False

    def get_result(self) -> str:
        if self.winner is None:
            return "The game ended in a draw."
        else:
            return f"Player {COLORS[self.winner]} is the winner."
