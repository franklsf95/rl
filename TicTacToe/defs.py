#!/usr/bin/env python3

State = int
Color = int  # can be X (1) or O (2)

# Game Configurations
D = 3  # board size
X = 1  # Player 1's piece
O = 2  # Player 2's piece
N_STATES = 3 ** (D * D)  # number of all possible states

COLORS = [" ", "X", "O"]
