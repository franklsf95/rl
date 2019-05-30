#!/usr/bin/env python3

import logging
import numpy as np

from defs import *
from env import Env
from value_function import evaluate_policy, value_of_action
from policy import FixedPolicy

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG
)


def optimize_policy(env: Env, policy: FixedPolicy, gamma: float = 0.9):
    n_iter = 0
    while True:
        n_iter += 1
        print(f"Policy Iteration {n_iter}:")
        policy.print()
        value_fn = evaluate_policy(env, policy, print_value=True)
        # Optimize policy according to the current value function
        policy_changed = False
        states = env.all_states()
        for s in states:
            if s not in policy.policy:
                continue
            old_action = policy.policy[s]
            new_action = None
            best_value = -np.inf
            for a in env.available_actions(s):
                v = value_of_action(env, s, a, value_fn, gamma)
                if best_value < v:
                    new_action = a
                    best_value = v
            if new_action != old_action:
                policy_changed = True
                policy.policy[s] = new_action
        if not policy_changed:
            print("Policy converged.")
            env.print()
            policy.print()
            break


def main():
    print("Negative environment:")
    env = Env.negative()
    env.print()
    policy = FixedPolicy(env)

    optimize_policy(env, policy)


if __name__ == "__main__":
    main()
