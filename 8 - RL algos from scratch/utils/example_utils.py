import random
from typing import Any
import numpy as np
from algos.algo import NArmedBanditAlgorithm
from environment.environment import NArmedBanditEnvironment


def normal_n_armed_bandit_example(algo: NArmedBanditAlgorithm, *args: Any, **kwargs: Any) -> NArmedBanditAlgorithm:
    n_arms = 3
    random.seed(42)  # For reproducibility
    np.random.seed(42)  # For reproducibility
    reward_distributions = [lambda: random.gauss(0, 2), lambda: random.gauss(5, 0.5), lambda: random.gauss(10, 8)]
    env = NArmedBanditEnvironment(actions_dimension=n_arms, reward_distributions=reward_distributions)
    agent = algo(env, *args, **kwargs)
    return agent