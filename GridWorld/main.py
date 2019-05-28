#!/usr/bin/env python3

from collections import Counter
import logging

from defs import *
from env import Env

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.DEBUG
)


def main():
    print("standard")
    env = Env.standard()
    env.print()
    print("negative")
    env = Env.negative()
    env.print()


if __name__ == "__main__":
    main()
