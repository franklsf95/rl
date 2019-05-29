#!/usr/bin/env python3

import abc
import logging
import numpy as np
from typing import Dict, Iterable, Tuple

from defs import *
from env import Env
from policy import FixedPolicy

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG
)


def main():
    print("Negative environment:")
    env = Env.negative()
    env.print()

    print("Initial policy:")
    policy = FixedPolicy(env)
    policy.print()


if __name__ == "__main__":
    main()
