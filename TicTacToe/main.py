#!/usr/bin/env python3

from defs import *
from env import Env
from player import Player


def main():
    env = Env()
    playerX = Player(X)
    playerO = Player(O)
    current_player = None

    while not env.is_game_over():
        current_player = playerO if current_player == playerX else playerX
        current_player.act(env)
        print(f"{current_player}'s move.")
        env.print()
        print()
    print("Game Over!")
    print(env.get_result())


if __name__ == "__main__":
    main()
