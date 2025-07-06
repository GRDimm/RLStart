from abc import ABC
from dataclasses import dataclass
from typing import Callable, List

from matplotlib import pyplot as plt

@dataclass(frozen=True)
class BaseEnvironment(ABC):
    def reset(self) -> None:
        """Reset the environment to its initial state."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def step(self, action: int) -> float:
        """Take an action in the environment and return the result."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def first_render(self) -> None:
        """Render the initial state of the environment."""
        raise NotImplementedError("This method should be overridden by subclasses.")


@dataclass(frozen=True)
class FiniteActionSpaceEnvironment(BaseEnvironment):
    actions_dimension: int


@dataclass(frozen=True)
class NArmedBanditEnvironment(FiniteActionSpaceEnvironment):
    reward_distributions: List[Callable[[], float]]
    
    def action_reward(self, action: int) -> float:
        return self.reward_distributions[action]()
    
    def step(self, action: int) -> float:
        if action < 0 or action >= self.actions_dimension:
            raise ValueError("Invalid action: action must be between 0 and n_actions - 1")
        
        return self.action_reward(action)
    
    def reset(self):
        pass
    
    def first_render(self):
        print(f"Current action space: {self.actions_dimension}")
        plt.figure(figsize=(14, 7))
        plt.suptitle("Reward Distributions for Each Action", fontsize=16)
        # Plot the distributions
        for i in range(self.actions_dimension):
            plt.hist([self.action_reward(i) for _ in range(10000)], bins=100, alpha=0.3, label=f"Action {i}")
        plt.xlabel("Reward value")
        plt.ylabel("Frequency")
        plt.legend()
        #plt.savefig("greedy_reward_distributions.png")
        plt.show()

