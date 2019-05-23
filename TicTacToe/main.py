#!/usr/bin/env python3

from collections import Counter
import logging
import numpy as np
import time
from typing import Iterable, List, Optional

from defs import *
from env import Env
from agent import Agent, InitialValueTriplet
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


def get_initial_value() -> List[InitialValueTriplet]:
    return list(_fill(Env(), 0, 0))


def _fill(env: Env, i: int, j: int) -> Iterable[InitialValueTriplet]:
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


def main():
    logging.info("Generating initial value triplets.")
    initial_value_triplets = get_initial_value()
    logging.info("Initializing agents.")
    agentX = Agent(X, initial_value_triplets)
    agentO = Agent(O, initial_value_triplets)

    logging.disable(logging.DEBUG)
    logging.info("Begin training episodes.")
    for t in range(N_TRAINING_EPISODES):
        play_game(agentX, agentO, train=True)
        n_iterations = t + 1
        if n_iterations % 1000 == 0:
            logging.info(f"Completed {n_iterations} simulations.")

    logging.info("Turning off random exploration for validation and testing.")
    agentX.epsilon = 0
    agentO.epsilon = 0
    logging.info("Begin validation episodes.")
    winners = Counter()
    for t in range(N_VALIDATION_EPISODES):
        if t % 2 == 0:
            env = play_game(agentX, agentO)
        else:
            env = play_game(agentO, agentX)
        winners.update({env.winner: 1})
    logging.info("Expect both players to be equally strong.")
    logging.info(f"Validation result: {winners}")

    logging.info("Begin testing episodes.")
    winners = Counter()
    random_player = Player(O)
    for t in range(N_TESTING_EPISODES):
        env = play_game(agentX, random_player)
        winners.update({env.winner: 1})
    logging.info("Expect Agent X (1) to never lose to the random player O.")
    logging.info(f"Testing result: {winners}")
    logging.disable(logging.NOTSET)

    logging.info("Training complete. Game is ready to play.")
    human_first = False
    while True:
        if human_first:
            env = play_game(HumanPlayer(X), agentO, print_game=True)
        else:
            env = play_game(agentX, HumanPlayer(O), print_game=True)
        print("Game Over!", env.get_result())
        print()
        human_first = not human_first


if __name__ == "__main__":
    main()
