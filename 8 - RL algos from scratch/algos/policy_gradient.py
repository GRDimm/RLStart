from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Callable, List
import numpy as np

from algos.algo import NArmedBanditAlgorithm
from utils.example_utils import normal_n_armed_bandit_example

@dataclass
class BasePolicy(ABC):
    action_preferences: List[Callable[[List[float]], float]]  # Action preferences
    parameters: List[float]  # Parameters for the policy (theta)

    @abstractmethod
    def normalisation_function(self, action_preferences: List[float]) -> List[float]:
        """Normalize the action preferences to create a probability distribution."""
        pass

    def policy_density(self, parameters: List[float]) -> List[float]:
        """Normalize the action preferences to create a probability distribution."""
        return self.normalisation_function([action_preference(parameters) for action_preference in self.action_preferences])
    
    def sample_action(self) -> int:
        """Sample an action based on the normalized action preferences."""
        probabilities = self.policy_density(self.parameters)
        return random.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
    
    def grad_log_policy_density(self, sampled_action: int, eps: float = 1e-5) -> List[float]:
        """Compute the gradient of the log policy density."""
        grad = [0.0] * len(self.parameters)
        for i in range(len(self.parameters)):
            theta_up = self.parameters.copy()
            theta_down = self.parameters.copy()
            theta_up[i] += eps
            theta_down[i] -= eps

            logp_up = np.log(self.policy_density(theta_up)[sampled_action])
            logp_down = np.log(self.policy_density(theta_down)[sampled_action])

            grad[i] = (logp_up - logp_down) / (2 * eps)
        return grad
    
    def update_parameters(self, reward: float, learning_rate: float, baseline: float, gradient: List[float]) -> None:
        """Update the policy parameters using the gradient."""
        self.parameters = [param + learning_rate * (reward - baseline) * grad for param, grad in zip(self.parameters, gradient)]

    @abstractmethod
    def reset(self) -> None:
        """Reset the policy parameters to their initial state."""
        pass


@dataclass
class SoftmaxPolicy(BasePolicy):

    def normalisation_function(self, action_preferences: List[float]) -> List[float]:
        """Normalize the action preferences using the softmax function."""
        exp_preferences = np.exp(np.array(action_preferences))
        return exp_preferences / np.sum(exp_preferences)
    
    def reset(self) -> None:
        """Reset the policy parameters to their initial state."""
        self.parameters = [0.0] * len(self.parameters)

    @staticmethod
    def base_case(actions_dimension: int = 3) -> SoftmaxPolicy:
        """Create a base case instance of SoftmaxPolicy with default parameters."""
        action_preferences = [lambda theta, i=i: theta[i] for i in range(actions_dimension)]
        initial_parameters = [0.0] * actions_dimension
        return SoftmaxPolicy(action_preferences=action_preferences, parameters=initial_parameters)
    

@dataclass
class PolicyGradient(NArmedBanditAlgorithm):
    policy: BasePolicy = SoftmaxPolicy.base_case()
    learning_rate: float = 0.01
    baseline: float = 0.0

    def select_action(self) -> int:
        return self.policy.sample_action()

    def update_algorithm(self, action: int, reward: float) -> None:
        gradient = self.policy.grad_log_policy_density(action)
        self.policy.update_parameters(reward, self.learning_rate, self.baseline, gradient)


if __name__ == "__main__":   
    policy_gradient = normal_n_armed_bandit_example(PolicyGradient)
    policy_gradient.train(n_steps=1000, render=True)
