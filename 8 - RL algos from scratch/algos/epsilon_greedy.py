
from dataclasses import dataclass
import random
from algo import NArmedBanditAlgorithm
from utils.example_utils import normal_n_armed_bandit_example


@dataclass
class EpsilonGreedy(NArmedBanditAlgorithm):
    epsilon: float = 0.1  # Probability of exploring

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            # Explore: select a random action
            return random.randint(0, self.environment.actions_dimension - 1)
        
        # Exploit: select the action with the highest average reward
        return max(self.action_reward_average, key=self.action_reward_average.get) if self.action_reward_average else random.randint(0, self.environment.actions_dimension - 1)
        
if __name__ == "__main__":
    agent = normal_n_armed_bandit_example(EpsilonGreedy)
    agent.train(n_steps=100000, render=True)