
from dataclasses import dataclass
import random
from typing import List, Tuple

from tqdm import tqdm

from utils.utils import BaseEnvironment


@dataclass
class NArmedBanditEnvironment(BaseEnvironment):
    # Probabilities of success for each arm
    reward_probabilities: List[float]
    # Each tuple contains (reward if successful, reward if unsuccessful)
    reward_values: List[Tuple[float, float]]

    def reset(self):
        pass

    def step(self, action: int):
        if action < 0 or action >= self.action_space:
            raise ValueError("Invalid action: action must be between 0 and n_arms - 1")
        
        # Simulate the reward based on the action taken
        reward = self.reward_values[action][0] if random.random() < self.reward_probabilities[action] else self.reward_values[action][1]
        return reward
    
    def first_render(self):
        print(f"Current action space: {self.action_space}")
        print(f"Reward probabilities: {self.reward_probabilities}")
        print(f"Reward values: {self.reward_values}")

@dataclass
class EpsilonGreedy:
    environment: BaseEnvironment
    epsilon: float  # Probability of exploring
    step: int = 0
    action_average_reward: List[float] = None


    def __post_init__(self):
        self.reset()

    def reset(self):
        """Reset the environment and the agent's state."""
        self.environment.reset()
        self.step = 0
        self.action_average_reward = [0.0] * self.environment.action_space

    def train(self, n_steps: int, render: bool = False):

        if render:
            print("Training started...")
            print(self.environment.first_render())

        for _ in tqdm(range(n_steps)):
            # Select the action with the highest average reward
            if random.random() < self.epsilon:
                # Explore: select a random action
                action = random.randint(0, self.environment.action_space - 1)
            else:
                # Exploit: select the action with the highest average reward
                action = self.action_average_reward.index(max(self.action_average_reward))
            
            reward = self.environment.step(action)
            
            # Update the average reward for the selected action
            self.action_average_reward[action] = (self.action_average_reward[action] * self.step + reward ) / (self.step + 1)
            self.step += 1

            if render:
                self.render()
    
    def render(self):
        print(f"Step: {self.step}, Average Rewards: {self.action_average_reward}")

if __name__ == "__main__":
    # Example usage
    n_arms = 3
    reward_probabilities = [random.uniform(0.1, 0.9) for _ in range(n_arms)]
    reward_values = [(random.uniform(1, 10), random.uniform(-1, -5)) for _ in range(n_arms)]

    env = NArmedBanditEnvironment(action_space=n_arms, reward_probabilities=reward_probabilities, reward_values=reward_values)
    agent = EpsilonGreedy(environment=env, epsilon=0.1)

    agent.train(n_steps=1000, render=True)
    print("Training completed.")
    print("Average rewards:", agent.action_average_reward)