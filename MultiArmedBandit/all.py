#!/usr/bin/env python3

import abc
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np


class Bandit(object):
    """
    A viewer's model of a Gaussian bandit.
    This object keeps track of our interactions with the slot machine,
    and our best estimate of the true mean.
    """

    def __init__(self, mu: float, mu_initial: float = 0):
        self.mu = mu  # the true mean
        self.mu_hat = mu_initial  # our current maximum likelihood estimate of mu
        self.n_pulls = 1  # how many times have we pulled the slot machine?
        # n_pulls must initialize to 1 for optimistic initial values to work (like a Bayesian prior?)

    def get_mu_hat(self) -> float:
        return self.mu_hat

    def pull(self) -> float:
        return np.random.randn() + self.mu

    def update(self, x: int):
        """
        Update the MLE with a weighted average of the old MLE and the new observation.

        x: float, an observation of the bandit (the result of a pull)
        """
        self.n_pulls += 1
        # print(
        #     f"Bandit={self.mu}  n_pulls={self.n_pulls}  x={x:.4f}  mu_hat={self.mu_hat:.4f}",
        #     end="",
        # )
        self.mu_hat = (1 - 1 / self.n_pulls) * self.mu_hat + 1 / self.n_pulls * x
        # print(f" -> {self.mu_hat:.4f}")

    def pull_and_update(self) -> float:
        x = self.pull()
        self.update(x)
        return x


class BayesianBandit(Bandit):
    """
    A viewer's model of a Gaussian bandit.
    This object keeps track of our interactions with the slot machine,
    and our best estimate of the true mean.
    Since this is a Bayesian estimate, we do not keep our best estimate mu_hat,
    but rather the parameters of our posterior distribution of mu.
    Our model is: X ~ Normal(mu, 1), mu ~ Normal(mu_0, lambda_0).
    Note that we are assuming the variance of X to be fixed at 1.
    """

    def __init__(self, mu: float):
        self.mu = mu  # the true mean
        self.x_sum = 0  # Cumulative sum of all observations so far
        self.mu_0 = 0  # posterior mean
        self.lambda_0 = 1  # posterior variance

    def pull(self) -> float:
        return np.random.randn() + self.mu

    def get_mu_hat(self) -> float:
        """
        Returns an estimate of mu drawn from the posterior distribution.
        """
        return np.random.normal(self.mu_0, np.sqrt(self.lambda_0))

    def update(self, x: int):
        """
        Update the posterior parameters using Gaussian conjugate.

        x: float, an observation of the bandit (the result of a pull)
        """
        self.x_sum += x
        self.lambda_0 += 1
        self.mu_0 = self.x_sum / self.lambda_0


class Strategy(abc.ABC):
    def get_mu_initial(self) -> float:
        return 0.0

    @abc.abstractmethod
    def choose(self, bandits: List[Bandit]) -> Bandit:
        pass


class EpsilonGreedyStrategy(Strategy):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __str__(self):
        return f"Eps({self.epsilon})"

    def __repr__(self):
        return f"Epsilon Greedy (eps={self.epsilon})"

    def choose(self, bandits: List[Bandit]) -> Bandit:
        n_bandits = len(bandits)
        coin_toss = np.random.random()
        if coin_toss < self.epsilon:
            # Explore. Randomly choose a bandit.
            b = np.random.choice(n_bandits)
        else:
            # Exploit. Choose the best bandit.
            b = np.argmax([bandit.get_mu_hat() for bandit in bandits])
        return bandits[b]


class OptimisticInitialValues(Strategy):
    def __init__(self, mu_initial: float):
        self.mu_initial = mu_initial

    def __str__(self):
        return f"OptInit({self.mu_initial})"

    def __repr__(self):
        return f"Optimistic Initial Values ({self.mu_initial})"

    def get_mu_initial(self) -> float:
        return self.mu_initial

    def choose(self, bandits: List[Bandit]) -> Bandit:
        # Always exploit because we set optimistic initial values.
        b = np.argmax([bandit.get_mu_hat() for bandit in bandits])
        return bandits[b]


class UpperConfidenceBound(Strategy):
    def __init__(self, n_total_trials: int):
        self.n_total_trials = n_total_trials

    def __str__(self):
        return f"UCB"

    def __repr__(self):
        return f"Upper Confidence Bound"

    def choose(self, bandits: List[Bandit]) -> Bandit:
        def ucb(bandit: Bandit) -> float:
            return bandit.get_mu_hat() + np.sqrt(
                2 * np.log(self.n_total_trials) / bandit.n_pulls
            )

        b = np.argmax([ucb(bandit) for bandit in bandits])
        return bandits[b]


class DecayingEpsilonGreedyStrategy(Strategy):
    def __init__(self):
        # total number of pulls
        self.t = 0

    def __str__(self):
        return f"DecayingEps"

    def __repr__(self):
        return f"Decaying Epsilon Greedy"

    def choose(self, bandits: List[Bandit]) -> Bandit:
        self.t += 1
        n_bandits = len(bandits)
        coin_toss = np.random.random()
        epsilon = 1 / (self.t + 1)
        if coin_toss < epsilon:
            # Explore. Randomly choose a bandit.
            b = np.random.choice(n_bandits)
        else:
            # Exploit. Choose the best bandit.
            b = np.argmax([bandit.get_mu_hat() for bandit in bandits])
        return bandits[b]


def test_run(mu: List[float], n_trials: int, bandit_cls: Callable, strategy: Strategy):
    bandits = [Bandit(m, strategy.get_mu_initial()) for m in mu]
    n_bandits = len(bandits)

    xs = np.empty(n_trials)  # our observations
    for t in range(n_trials):
        bandit = strategy.choose(bandits)
        x = bandit.pull_and_update()
        xs[t] = x

    # Calculate the average rewards up to every time tick (higher is better)
    cumulative_avg = np.cumsum(xs) / (np.arange(1, n_trials + 1))

    print(f"Experiment with {repr(strategy)}")
    for bandit in bandits:
        print(
            f"Bandit has mu {bandit.mu}; estimated to be {bandit.mu_hat}; pulled {bandit.n_pulls} times"
        )
    print()

    return cumulative_avg


def main():
    mu = [1, 2, 3]
    n_trials = 100000
    for bandit_cls, strategy in [
        (Bandit, EpsilonGreedyStrategy(0.1)),
        (Bandit, DecayingEpsilonGreedyStrategy()),
        (Bandit, OptimisticInitialValues(10)),
        (Bandit, UpperConfidenceBound(n_trials)),
        (BayesianBandit, OptimisticInitialValues(10)),
    ]:
        cumulative_avg = test_run(mu, n_trials, bandit_cls, strategy)
        plt.plot(cumulative_avg, label=f"{bandit_cls.__name__}, {strategy}")
    plt.legend()
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    main()
