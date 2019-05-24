#!/usr/bin/env python3

import logging
import numpy as np
from typing import List, Optional, Tuple

from defs import *
from env import Env
from player import Player

InitialValueTriplet = Tuple[State, bool, Optional[Color]]


class Agent(Player):
    """
    A trainable reinforcement-learning agent.
    """

    def __init__(
        self,
        color: Color,
        initial_value_triplets: List[InitialValueTriplet],
        alpha=0.5,
        epsilon=0.1,
    ):
        super().__init__(color)
        self.color = color  # color of the player, X or O
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # probability of choosing to explore (random action)
        self.state_history: List[State] = []  # history of states in the current game
        self.__initialize_value(initial_value_triplets)

    def __str__(self):
        return f"Agent {COLORS[self.color]}"

    def __initialize_value(self, triplets: List[InitialValueTriplet]):
        self.value = np.zeros(N_STATES)
        for state, game_over, winner in triplets:
            if game_over:
                v = 1 if winner == self.color else 0
            else:
                v = 0.5
            self.value[state] = v
        logging.info(f"{self} initialized.")

    def record_state(self, state: State):
        self.state_history.append(state)

    def learn(self, won: bool):
        """
        Update the value function from the outcome of this game.
        """
        reward = 1 if won else 0
        target = reward
        for s in reversed(self.state_history):
            v = self.value[s]
            v = v + self.alpha * (target - v)
            self.value[s] = v
            target = v
        # Reset state history.
        self.state_history = []

    def act(self, env: Env):
        coin_toss = np.random.random()
        if coin_toss < self.epsilon:
            # Act randomly
            super().act(env)
        else:
            # Act rationally
            self.act_best_value(env)

    def act_best_value(self, env: Env):
        best_value = 0
        best_loc = None
        loc_value_map = {}
        for i in range(D):
            for j in range(D):
                if not env.can_place(i, j):
                    continue
                env.place(self.color, i, j)
                state = env.get_state()
                value = self.value[state]
                env.place(0, i, j)
                loc_value_map[(i, j)] = value
                if value >= best_value:
                    best_value = value
                    best_loc = (i, j)
        logging.debug(f"{loc_value_map}")
        if best_loc:
            i, j = best_loc
            env.place(self.color, i, j)
