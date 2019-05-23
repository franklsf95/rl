#!/usr/bin/env python3

from collections import Counter
import logging
import numpy as np
import time
from typing import Optional

from defs import *
from env import Env
from agent import Agent
from human_player import HumanPlayer
from player import Player

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG
)

N_TRAINING_EPISODES = 10000
N_VALIDATION_EPISODES = 1000
N_TESTING_EPISODES = 1000


def play_game(playerX: Player, playerO: Player, train=False, print_game=False) -> Env:
    """
    Plays the game and returns the environment.
    """
    env = Env()
    current_player = None

    def print_env():
        if not print_game:
            return
        if current_player:
            print(f"{current_player}'s move.")
        else:
            print("Game start.")
        env.print()

    while not env.is_game_over():
        print_env()
        current_player = playerO if current_player == playerX else playerX
        current_player.act(env)

        if train:
            state = env.get_state()
            playerX.record_state(state)
            playerO.record_state(state)

    print_env()
    if train:
        playerX.update_value(env)
        playerO.update_value(env)
    return env


def main():
    logging.info("Hello, world!")
    agentX = Agent(X)
    agentO = Agent(O)

    logging.disable(logging.DEBUG)
    logging.info("Begin training episodes.")
    for t in range(N_TRAINING_EPISODES):
        play_game(agentX, agentO, train=True)
        n_iterations = t + 1
        if n_iterations % 1000 == 0:
            logging.info(f"Completed {n_iterations} simulations.")

    logging.info("Begin validation episodes.")
    winners = Counter()
    for t in range(N_VALIDATION_EPISODES):
        env = play_game(agentX, agentO)
        winners.update({env.winner: 1})
    logging.info("Expect both players to be equally strong.")
    logging.info(f"Validation result: {winners}")

    logging.info("Turning off random exploration for testing")
    agentX.epsilon = 0
    logging.info("Begin testing episodes")
    winners = Counter()
    random_player = Player(O)
    for t in range(N_TESTING_EPISODES):
        env = play_game(agentX, random_player)
        winners.update({env.winner: 1})
    logging.info("Expect Agent X (1) to be stronger than the random player O (2).")
    logging.info(f"Testing result: {winners}")
    logging.disable(logging.NOTSET)

    logging.info("Training complete. Game is ready to play.")
    while True:
        env = play_game(HumanPlayer(X), agentO, print_game=True)
        print("Game Over!", env.get_result())
        print()


if __name__ == "__main__":
    main()
