
from dataclasses import dataclass, field
import random
from typing import Dict
from tqdm import tqdm
from utils.utils import FiniteActionSpaceEnvironment, NArmedBanditEnvironment


@dataclass
class Greedy:
    environment: FiniteActionSpaceEnvironment
    step: int = 0
    action_average_reward: Dict[int, float] = field(default_factory=dict)

    def reset(self):
        """Reset the environment and the agent's state."""
        self.environment.reset()
        self.step = 0
        self.action_average_reward = {action: 0.0 for action in range(self.environment.actions_dimension)}

    def train(self, n_steps: int, render: bool = False):
        self.reset()

        if render:
            print("Training started...")
            print(self.environment.first_render())

        for _ in tqdm(range(n_steps)):
            # Select the action with the highest average reward
            action = max(self.action_average_reward, key=self.action_average_reward.get) if self.action_average_reward else random.randint(0, self.environment.actions_dimension - 1)
            reward = self.environment.step(action)
            
            # Update the average reward for the selected action
            self.action_average_reward[action] = (self.action_average_reward.get(action, 0) * self.step + reward ) / (self.step + 1)
            self.step += 1

            if render:
                self.render()
    
    def render(self):
        print(f"Step: {self.step}, Average Rewards: {self.action_average_reward}")

if __name__ == "__main__":
    # Example usage
    n_arms = 3
    reward_distributions = []
    random.seed(42)  # For reproducibility

    for i in range(n_arms):
        # Create a random Gaussian distribution for each arm
        # Mean is between -5 and 5, standard deviation is between 0 and 5
        mean = random.uniform(-25, 25)
        stddev = random.uniform(3, 20)
        reward_distributions.append(lambda m=mean, s=stddev: random.gauss(m, s))

    env = NArmedBanditEnvironment(actions_dimension=n_arms, reward_distributions=reward_distributions)
    agent = Greedy(environment=env)

    agent.train(n_steps=100, render=True)
    print("Training completed.")
    print("Average rewards:", agent.action_average_reward)