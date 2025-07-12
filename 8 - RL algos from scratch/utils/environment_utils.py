from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List
from matplotlib import pyplot as plt

@dataclass(frozen=True)
class BaseEnvironment(ABC):
    def reset(self) -> None:
        """Reset the environment to its initial state."""
        pass

    @abstractmethod
    def step(self, action: int) -> float:
        """Take an action in the environment and return the result."""
        pass
    
    def first_render(self) -> None:
        """Render the initial state of the environment."""
        pass

@dataclass(frozen=True)
class FiniteActionSpaceStatelessEnvironment(BaseEnvironment):
    actions_dimension: int  # Available actions in the environment [0, actions_dimension - 1]


@dataclass(frozen=True)
class NArmedBanditEnvironment(FiniteActionSpaceStatelessEnvironment):
    """Environment for N-armed bandit problems."""
    reward_distributions: List[Callable[[], float]]
    
    def action_reward(self, action: int) -> float:
        return self.reward_distributions[action]()
    
    def step(self, action: int) -> float:
        if action < 0 or action >= self.actions_dimension:
            raise ValueError("Invalid action: action must be between 0 and n_actions - 1")
        
        return self.action_reward(action)
    
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


@dataclass(frozen=True)
class FiniteActionSpaceEnvironment(BaseEnvironment):
    """Environment with finite action space and state space."""
    actions_dimension: int  # Available actions in the environment [0, actions_dimension - 1]
    state_dimension: int  # Environments states [0, state_dimension - 1]

