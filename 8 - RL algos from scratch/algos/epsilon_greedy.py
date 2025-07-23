from collections import deque
import random
from typing import Dict, Tuple

import pygame
import torch
from algo import NArmedBanditAlgorithm, StatefulDiscreteActionSpaceAlgorithm
from environment.environment import FiniteActionSpaceEnvironment
from utils.example_utils import cartpole_example, normal_n_armed_bandit_example
from utils.q_learning_utils import QNetwork

class EpsilonGreedy(NArmedBanditAlgorithm):
    def __init__(self, environment, epsilon: float = 0.1):
        super().__init__(environment)
        self.epsilon = epsilon

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            # Explore: select a random action
            return random.randint(0, self.environment.actions_dimension - 1)
        
        # Exploit: select the action with the highest average reward
        return max(self.action_reward_average, key=self.action_reward_average.get) if self.action_reward_average else random.randint(0, self.environment.actions_dimension - 1)


class QLearningEpsilonGreedy(StatefulDiscreteActionSpaceAlgorithm):  # TODO: Make this work
    def __init__(self, environment: FiniteActionSpaceEnvironment, epsilon: float = 0.1, epsilon_min: float = 0.01, epsilon_decay: float = 0.999):
        super().__init__(environment)
        self.environment = environment
        self.model = QNetwork(len(environment.state()), environment.actions_dimension)
        self.gamma = 0.95  # Discount factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = 0.99
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 128
        self.action_reward_average: Dict[int, float] = {}
        self.action_reward_variance: Dict[int, float] = {}
        self.action_counts: Dict[int, int] = {}
        self.training_frequency = 200
        self.target_model = QNetwork(len(environment.state()), environment.actions_dimension)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_update_frequency = 600
    
    def select_action(self) -> int:
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            q_values = self.model(torch.tensor(self.environment.state(normalized=True), dtype=torch.float32).unsqueeze(0))
            action = torch.argmax(q_values).item()
        return action
    
    def update_algorithm_step(self, action: int, reward: float, previous_state: Tuple[float, ...], next_state: Tuple[float, ...], done: bool) -> None:
        assert 0 <= action < self.environment.actions_dimension
        self.replay_buffer.append((previous_state, action, reward, next_state, done))

        if len(self.replay_buffer) < self.batch_size or self.training_step % self.training_frequency != 0:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        loss_total = 0.0
        self.model.optimizer.zero_grad()

        for s, a, r, s_next, done in batch:
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            s_next = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0)
            r = torch.tensor([r], dtype=torch.float32)

            with torch.no_grad():
                if done:
                    target = r
                else:
                    target = r + self.gamma * self.target_model(s_next).max(1)[0]

            q_pred = self.model(s)[0, a]
            loss_total += (q_pred - target[0]) ** 2

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.model.optimizer.step()
        self.stats["Loss"] = loss_total.item() / self.batch_size

        if self.training_step % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_algorithm_episode(self, total_steps: int, total_reward: float) -> None:
        """Update epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.stats["Epsilon"] = self.epsilon

    def reset(self) -> None:
        """Reset the environment and the agent's state."""
        self.environment.reset()
        self.training_step = 0
        self.episode = 0
        self.replay_buffer.clear()

    def run_episode(self) -> bool:
        """Run a single episode of the algorithm."""
        self.environment.reset()
        done = False
        while not done:
            action = self.select_action()
            next_state, reward, done, _ = self.environment.step(action)
            self.environment.render()
        
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
        return True
        

if __name__ == "__main__":
    # # Epsilon-Greedy agent example
    # agent = normal_n_armed_bandit_example(EpsilonGreedy)
    # agent.train(n_episodes=100000, render=True)

    # Q-learning with Epsilon-Greedy agent example
    q_agent = cartpole_example(QLearningEpsilonGreedy, epsilon=0.4, epsilon_min=0.01, epsilon_decay=0.9995)
    q_agent.train(n_episodes=10000, render=True, render_every=1000)
