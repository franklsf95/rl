#!/usr/bin/env python3

from collections import Counter
import logging
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

N_ITERATIONS = 10000
TRAINING_PCT = 0.8
VALIDATION_PCT = 0.1
TESTING_PCT = 0.1


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
        for agent in [playerX, playerO]:
            agent.update_value()
            agent.reset_state_history()
    return env


def main():
    logging.info("Hello, world!")
    agentX = Agent(X)
    agentO = Agent(O)

    logging.info("Begin training episodes.")
    n_training_episodes = round(N_ITERATIONS * 0.1)
    for t in range(n_training_episodes):
        play_game(agentX, agentO, train=True)
        n_iterations = t + 1
        if n_iterations % 1000 == 0:
            logging.info(f"Completed {n_iterations} simulations.")

    logging.info("Begin validation episodes.")
    winners = Counter()
    n_validation_episodes = round(N_ITERATIONS * VALIDATION_PCT)
    for t in range(n_validation_episodes):
        env = play_game(agentX, agentO)
        winners.update({env.winner: 1})
    logging.info("Expect both players to be equally strong.")
    logging.info(f"Validation result: {winners}")

    logging.info("Begin testing episodes")
    winners = Counter()
    n_testing_episodes = round(N_ITERATIONS * TESTING_PCT)
    random_player = Player(O)
    for t in range(n_testing_episodes):
        env = play_game(agentX, random_player)
        winners.update({env.winner: 1})
    logging.info("Expect Agent X (1) to be stronger than the random player O (2).")
    logging.info(f"Testing result: {winners}")

    logging.info("Training complete. Game is ready to play.")
    while True:
        env = play_game(HumanPlayer(X), agentO, print_game=True)
        print("Game Over!", env.get_result())
        print()


if __name__ == "__main__":
    main()
