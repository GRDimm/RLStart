

from dataclasses import dataclass, field
import random
from typing import Callable, Dict, List

from numpy import log, sqrt

from algo import NArmedBanditAlgorithm
from utils.environment_utils import FiniteActionSpaceEnvironment, NArmedBanditEnvironment
from utils.example_utils import normal_n_armed_bandit_example


@dataclass
class UCB(NArmedBanditAlgorithm):
    # Function to calculate the upper confidence bound
    # This function should take the step, action counts, average reward for action, and variance of reward for action as inputs and return the bound value
    upper_bound: Callable[[int, int, float, float], float] = field(default_factory=lambda: lambda step, action_count, action_reward_average, action_reward_variance: sqrt(2 * log(step) / action_count) + 3 * log(step) / action_count)
    exploration_parameter: float = 2.0 # Exploration parameter

    def select_action(self) -> int:
        return max(
            range(self.environment.actions_dimension),
            key=lambda a: self.action_reward_average[a] + self.exploration_parameter*self.upper_bound(
                self.training_step,
                self.action_counts[a],
                self.action_reward_average[a],
                self.action_reward_variance[a],
            )
        )

        
if __name__ == "__main__":
    # Define the upper bound function
    # This is a simple implementation of the upper confidence bound formula
    def upper_bound(step: int, action_count: int, _action_reward_average: float, action_reward_variance: float) -> float:
        if action_count == 0:
            return float('inf')
        if step == 0:
            return 0.0
        return sqrt(2 * log(step) * action_reward_variance / action_count) + 3 * log(step) / action_count

    ucb_agent = normal_n_armed_bandit_example(UCB, upper_bound=upper_bound)
    ucb_agent.train(n_steps=100000, render=True)
    