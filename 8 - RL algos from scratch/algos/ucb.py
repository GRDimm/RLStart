from typing import Callable
from numpy import log, sqrt

from algos.algo import StatelessFiniteActionSpaceAlgorithm
from utils.example_utils import cartpole_example, normal_n_armed_bandit_example


class UCB(StatelessFiniteActionSpaceAlgorithm):
    def __init__(self, environment, upper_bound: Callable[[int, int, float, float], float] = None):
        super().__init__(environment)
        # Function to calculate the upper confidence bound
        # This function should take the step, action counts, average reward for action, and variance of reward for action as inputs and return the bound value
        self.upper_bound = upper_bound if upper_bound else self.default_upper_bound
        self.exploration_parameter = 2.0
    
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
    
    @staticmethod
    def default_upper_bound(step: int, action_count: int, _action_reward_average: float, action_reward_variance: float) -> float:
        if action_count == 0:
            return float('inf')
        if step == 0:
            return 0.0
        return sqrt(2 * log(step) * action_reward_variance / action_count) + 3 * log(step) / action_count

    
if __name__ == "__main__":
    ucb_agent = normal_n_armed_bandit_example(UCB)
    ucb_agent.train(n_episodes=100000, render=True)
