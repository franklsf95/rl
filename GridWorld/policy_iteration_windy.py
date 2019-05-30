#!/usr/bin/env python3

import logging

from defs import *
from env import Env
from value_function import evaluate_policy, value_of_action
from policy import FixedPolicy
from policy_iteration import optimize_policy

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG
)


def main():
    print("Negative environment:")
    env = Env.negative(step_reward=-1)
    env.print()
    policy = FixedPolicy(env)

    global WINDY
    WINDY = True
    print("'Windy' environment set.")

    optimize_policy(env, policy)


if __name__ == "__main__":
    main()
