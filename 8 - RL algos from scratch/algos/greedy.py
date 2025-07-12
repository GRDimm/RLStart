
from dataclasses import dataclass
import random
from algos.algo import NArmedBanditAlgorithm
from utils.example_utils import normal_n_armed_bandit_example


@dataclass
class Greedy(NArmedBanditAlgorithm):

    def select_action(self) -> int:
        return max(self.action_reward_average, key=self.action_reward_average.get) if self.action_reward_average else random.randint(0, self.environment.actions_dimension - 1)  
    

if __name__ == "__main__":
    agent = normal_n_armed_bandit_example(Greedy)
    agent.train(n_steps=1000, render=True)