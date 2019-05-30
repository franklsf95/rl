#!/usr/bin/env python3

import logging
import numpy as np
from typing import Dict, Iterable, Tuple

from defs import *
from env import Env
from policy import Policy, UniformlyRandomPolicy, FixedPolicy

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG
)


def print_value_fn(env: Env, value: ValueFn):
    for i in range(env.rows):
        for j in range(env.cols):
            v = value.get((i, j), 0)
            s = f"{v:.2f}"
            # Now print
            if len(s) <= 4:
                print(f" {s}  ", end="")
            else:
                print(f"{s}  ", end="")
        print()


def value_of_action(
    env: Env, state: State, action: Action, value_fn: ValueFn, gamma: float
) -> float:
    env.state = state
    r = env.move(action)
    v = r + gamma * value_fn[env.state]
    return v


def evaluate_policy(
    env: Env, policy: Policy, gamma: float = 0.9, print_value: bool = False
) -> ValueFn:
    """
    gamma: Discount factor.
    """
    # Initialize value function.
    states = env.all_states()
    value_fn = {s: 0 for s in states}

    # Iterate until convergence.
    logging.info(f"Start evaluating {policy}.")
    n_iter = 0
    while True:
        n_iter += 1
        if n_iter % 100 == 0:
            logging.debug(f"Iteration {n_iter}")
        max_change = 0
        for s in states:
            old_value = value_fn[s]
            # Compute new value
            new_value = 0
            for p, a in policy.actions(env, s):
                v = value_of_action(env, s, a, value_fn, gamma)
                new_value += p * v
            value_fn[s] = new_value
            value_change = np.abs(new_value - old_value)
            if max_change < value_change:
                max_change = value_change
        if max_change < VALUE_CONVERGENCE_EPSILON:
            break
    logging.info(f"Evaluation converged in {n_iter} iterations.")
    if print_value:
        print(f"Value function for {policy}:")
        print_value_fn(env, value_fn)
    return value_fn


def main():
    print("Standard environment:")
    env = Env.standard()
    env.print()

    evaluate_policy(env, UniformlyRandomPolicy(), gamma=1.0, print_value=True)
    evaluate_policy(
        env,
        FixedPolicy(
            env,
            {
                (0, 0): Action.Right,
                (0, 1): Action.Right,
                (0, 2): Action.Right,
                (1, 0): Action.Up,
                (1, 2): Action.Right,
                (2, 0): Action.Up,
                (2, 1): Action.Right,
                (2, 2): Action.Right,
                (2, 3): Action.Up,
            },
        ),
        gamma=0.9,
        print_value=True,
    )


if __name__ == "__main__":
    main()
