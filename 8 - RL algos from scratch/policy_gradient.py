from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
from typing import Callable, Dict, List, Optional
import numpy as np

from utils.utils import NArmedBanditEnvironment

@dataclass
class BasePolicy(ABC):
    actions_dimension: int
    action_preferences: List[Callable[[List[float]], float]]  # Action preferences
    parameters: List[float]  # Parameters for the policy (theta)
    action_average_rewards: Dict[int, float] = field(default_factory=dict)
    action_counts: Dict[int, int] = field(default_factory=dict)

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
        return np.random.choice(range(self.actions_dimension), p=probabilities)
    
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
        self.parameters = [0.0] * self.actions_dimension
        self.action_average_rewards = {i: 0.0 for i in range(self.actions_dimension)}
        self.action_counts = {i: 0 for i in range(self.actions_dimension)}
    

@dataclass
class PolicyGradient:
    environment: NArmedBanditEnvironment
    policy: BasePolicy
    learning_rate: float = 0.01
    baseline: float = 0.0
    

    def train(self, n_steps: int, render: bool = False) -> None:
        """Train the policy using the policy gradient method."""
        self.policy.reset()

        if render:
            print("Training started...")
            print(self.environment.first_render())
        
        for step in range(n_steps):
            action = self.policy.sample_action()
            reward = self.environment.step(action)
            
            gradient = self.policy.grad_log_policy_density(action)

            self.policy.update_parameters(reward, self.learning_rate, self.baseline, gradient)

            self.policy.action_counts[action] += 1
            self.policy.action_average_rewards[action] += (reward - self.policy.action_average_rewards.get(action, 0)) / self.policy.action_counts[action]
            
            if render:
                print(f"Step {step}, Action: {action}, Reward: {reward}")
                print(f"Step {step}, Gradient: {gradient}")
                print(f"Step {step}, Policy Density: {self.policy.policy_density(self.policy.parameters)}")

        if render:
            print("Training completed.")
            print("Final Policy Density:", self.policy.policy_density(self.policy.parameters))
            print("Final average rewards:", self.policy.action_average_rewards)


if __name__ == "__main__":
    n_arms = 3
    np.random.seed(42)  # For reproducibility
    random.seed(42)  # For reproducibility
    reward_distributions = [lambda: random.gauss(0, 2), lambda: random.gauss(5, 0.5), lambda: random.gauss(10, 8)]
    environment = NArmedBanditEnvironment(n_arms, reward_distributions)
    action_preferences = [lambda theta, i=i: theta[i] for i in range(n_arms)]
    initial_parameters = [0.0] * n_arms  # Initial parameters (theta)
    policy = SoftmaxPolicy(actions_dimension=n_arms, action_preferences=action_preferences, parameters=initial_parameters)
    policy_gradient = PolicyGradient(environment=environment, policy=policy, learning_rate=0.01, baseline=0.0)
    policy_gradient.train(n_steps=1000, render=True)
    print("Training completed.")