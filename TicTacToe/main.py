#!/usr/bin/env python3

from typing import Optional

from defs import *
from env import Env
from player import Player
from human_player import HumanPlayer


def main():
    env = Env()
    playerX = HumanPlayer(X)
    playerO = Player(O)
    current_player = None

    def print_env():
        if current_player:
            print(f"{current_player}'s move.")
        else:
            print("Game start.")
        env.print()

    while not env.is_game_over():
        print_env()
        current_player = playerO if current_player == playerX else playerX
        current_player.act(env)
    print_env()
    print("Game Over!", env.get_result())


if __name__ == "__main__":
    main()
