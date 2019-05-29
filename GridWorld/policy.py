#!/usr/bin/env python3

import abc
import numpy as np
from typing import Dict, Iterable, Optional

from defs import *
from env import Env


class Policy(abc.ABC):
    @abc.abstractmethod
    def actions(self, env: Env, s: State) -> Iterable[Tuple[Prob, Action]]:
        pass


class UniformlyRandomPolicy(Policy):
    def __str__(self):
        return "Uniformly Random Policy"

    def actions(self, env: Env, s: State) -> Iterable[Tuple[Prob, Action]]:
        actions = env.available_actions(s)
        if len(actions) == 0:
            return []
        p = 1 / len(actions)
        return [(p, a) for a in actions]


class FixedPolicy(Policy):
    def __init__(self, env: Env, policy: Optional[Dict[State, Action]] = None):
        self.env = env
        if policy:
            self.policy = policy
        else:
            # Initialize policy randomly.
            self.policy = {
                s: np.random.choice(actions) for s, actions in env.actions.items()
            }

    def __str__(self):
        return "Fixed Policy"

    def actions(self, env: Env, s: State) -> Iterable[Tuple[Prob, Action]]:
        return [(1, self.policy[s])] if s in self.policy else []

    def print(self):
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                v = self.policy.get((i, j), Action.Unknown)
                s = repr(v)
                print(f" {s} ", end="")
            print()
