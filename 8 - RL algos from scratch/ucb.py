

from dataclasses import dataclass, field
import random
from typing import Callable, Dict, List

from numpy import log, sqrt

from utils.utils import FiniteActionSpaceEnvironment, NArmedBanditEnvironment


@dataclass
class UCB:
    environment: FiniteActionSpaceEnvironment
    # Function to calculate the upper confidence bound
    # This function should take the step, action counts, average reward for action, and variance of reward for action as inputs and return the bound value
    upper_bound: Callable[[int, int, float, float], float]
    step: int = 0
    action_reward_average: Dict[int, float] = field(default_factory=dict)
    action_reward_variance: Dict[int, float] = field(default_factory=dict)
    action_counts: Dict[int, int] = field(default_factory=dict)
    exploration_parameter: float = 2.0 # Exploration parameter

    def reset(self):
        """Reset the environment and the agent's state."""
        self.environment.reset()
        self.step = 0
        self.action_reward_average = {action: 0.0 for action in range(self.environment.actions_dimension)}
        self.action_reward_variance = {action: 0.0 for action in range(self.environment.actions_dimension)}
        self.action_counts = {action: 0 for action in range(self.environment.actions_dimension)}
        self.reward_history = {action: [] for action in range(self.environment.actions_dimension)}

    def render(self):
        print(f"Step: {self.step}, Average Rewards: {self.action_reward_average}, Variance: {self.action_reward_variance}, Action Counts: {self.action_counts}")

    def train(self, n_steps: int, render: bool = False):
        self.reset()

        if render:
            print("Training started...")
            print(self.environment.first_render())

        for _ in range(n_steps):
            # Select the action with the highest upper confidence bound
            action = max(
                range(self.environment.actions_dimension),
                key=lambda a: self.action_reward_average[a] + self.exploration_parameter*self.upper_bound(
                    self.step,
                    self.action_counts[a],
                    self.action_reward_average[a],
                    self.action_reward_variance[a],
                )
            )
            reward = self.environment.step(action)

            # Update the average reward and counts for the selected action
            self.action_counts[action] += 1
            self.reward_history[action].append(reward)
            self.action_reward_average[action] = (
                self.action_reward_average.get(action, 0) * (self.action_counts[action] - 1) + reward
            ) / self.action_counts[action]

            # Update the variance for the selected action
            if len(self.reward_history[action]) > 1:
                mean = self.action_reward_average[action]
                self.action_reward_variance[action] = sum(
                    (r - mean) ** 2 for r in self.reward_history[action]
                ) / (self.action_counts[action] - 1)
            
            if render:
                self.render()
            
            self.step += 1
        
if __name__ == "__main__":
    n_arms = 3
    random.seed(42)  # For reproducibility
    reward_distributions = [lambda: random.gauss(0, 2), lambda: random.gauss(5, 0.5), lambda: random.gauss(10, 8)]
    environment = NArmedBanditEnvironment(n_arms, reward_distributions)

    # Define the upper bound function
    # This is a simple implementation of the upper confidence bound formula
    def upper_bound(step: int, action_count: int, _action_reward_average: float, action_reward_variance: float) -> float:
        if action_count == 0:
            return float('inf')
        if step == 0:
            return 0.0
        return sqrt(2 * log(step) * action_reward_variance / action_count) + 3 * log(step) / action_count

    ucb_agent = UCB(
        environment=environment,
        upper_bound=upper_bound,
    )

    ucb_agent.train(n_steps=1000, render=True)
    print("Training completed.")
    