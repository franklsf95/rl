#!/usr/bin/env python3

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from SlotMachine import SlotMachine


class Bandit(object):
    """
    A viewer's model of a Gaussian bandit.
    This object keeps track of our interactions with the slot machine,
    and our best estimate of the true mean.
    """

    def __init__(self, mu: float):
        self.mu = mu  # the true mean
        self.mu_hat = 0  # our current maximum likelihood estimate of mu
        self.n_pulls = 0  # how many times have we pulled the slot machine?

    def pull(self) -> float:
        return np.random.randn() + self.mu

    def update(self, x: int):
        """
        Update the MLE with a weighted average of the old MLE and the new observation.

        x: float, an observation of the bandit (the result of a pull)
        """
        self.n_pulls += 1
        self.mu_hat = (1 - 1.0 / self.n_pulls) * self.mu_hat + 1.0 / self.n_pulls * x

    def pull_and_update(self) -> float:
        x = self.pull()
        self.update(x)
        return x


def test_epsilon(mu: List[float], epsilon: float, n_trials: int):
    bandits = list(map(Bandit, mu))
    n_bandits = len(bandits)

    xs = np.empty(n_trials)  # our observations
    for t in range(n_trials):
        coin_toss = np.random.random()
        if coin_toss < epsilon:
            # Explore. Randomly choose a bandit.
            b = np.random.choice(n_bandits)
        else:
            # Exploit. Choose the best bandit.
            b = np.argmax([bandit.mu_hat for bandit in bandits])

        # Pull from the b-th bandit.
        x = bandits[b].pull_and_update()
        xs[t] = x

    # Calculate the average rewards up to every time tick (higher is better)
    cumulative_avg = np.cumsum(xs) / (np.arange(1, n_trials + 1))

    print(f"Experiment with epsilon = {epsilon}")
    for bandit in bandits:
        print(f"Bandit has mu {bandit.mu}; estimated to be {bandit.mu_hat}")
    print()

    return cumulative_avg


def main():
    mu = [1.0, 2.0, 3.0]
    n_trials = 100000
    for eps in [0.1, 0.05, 0.01]:
        cumulative_avg = test_epsilon(mu, eps, n_trials)
        plt.plot(cumulative_avg, label=f"e={eps}")
    plt.legend()
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    main()
