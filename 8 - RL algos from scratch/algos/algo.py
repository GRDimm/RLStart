
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
from environment.environment import BaseEnvironment, FiniteActionSpaceEnvironment, StatelessEnvironment
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


class Algorithm(ABC):
    def __init__(self, environment: BaseEnvironment):
        self.environment = environment
        self.episode: int = 0
        self.training_step: int = 0
        self.stats: Dict[str, Any] = {}

    def train(self, n_episodes: int, render: bool = False, render_every: int = 1) -> None:
        self.reset()

        if render:
            self.environment.first_render()

        with Live(self.populate_table(), refresh_per_second=4) as live:
            for _ in range(n_episodes):
                self.environment.reset()
                total_steps, total_reward = self.train_episode(live, render, render_every)
                self.update_episode_stats(total_steps, total_reward)
                self.update_algorithm_episode(total_steps, total_reward)
        self.run()

    def update_episode_stats(self, total_steps: int, total_reward: float) -> None:
        """Update statistics after each episode."""
        self.stats.setdefault("Episode length history", deque(maxlen=5)).append(total_steps)
        self.stats.setdefault("Episode reward history", deque(maxlen=5)).append(total_reward)
        # Average episode length for last 5 episodes
        lengths = list(self.stats["Episode length history"])
        self.stats["Average episode length"] = sum(lengths) / len(lengths)
    
    def update_algorithm_episode(self, total_steps: int, total_reward: float) -> None:
        """Update algorithm-specific statistics after each episode."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment and the algorithm's state."""
        pass
    
    @abstractmethod
    def select_action(self) -> int:
        """Select an action based on the current policy."""
        pass

    def train_episode(self, live: Live, render: bool = False, render_every: int = 1) -> None:
        """Perform a single training episode."""
        done = False
        total_steps = 0
        total_reward = 0.0
        while not done:
            previous_state = self.environment.state(normalized=True)
            action = self.select_action()
            new_state, reward, done, info = self.environment.step(action)
            if render and total_steps % render_every == 0:
                self.environment.render()
            self.update_algorithm_step(action, reward, previous_state, new_state, done)
            self.update_base_stats(action, reward)
            self.update_stats(action, reward)
            self.render_stats(live)
            total_steps += 1
            total_reward += reward
            self.training_step += 1
        
        self.episode += 1
        return total_steps, total_reward

    def update_base_stats(self, action: Any, reward: float) -> None:
        """Update the base statistics for the algorithm."""
        self.stats["Average episode length"] = self.training_step / self.episode if self.episode > 0 else 0

    def update_stats(self, action: Any, reward: float) -> None:
        pass

    def update_algorithm_step(self, action: Any, reward: float, previous_state: Any, next_state: Any, done: bool) -> None:
        """Update the algorithm's state after receiving a reward."""
        pass

    def populate_table(self) -> Table:
        table = Table(title="Training Stats")
        table.add_column("Metric", justify="left")
        table.add_column("Values", justify="left")

        for stat, value in self.stats.items():
            if isinstance(value, dict):
                value = ", ".join(f"{k}: {v:.2f}" for k, v in value.items())
            elif isinstance(value, (list, deque)):
                value = ", ".join(f"{v:.2f}" for v in value)
            elif isinstance(value, float):
                value = f"{value:.2f}"
            else:
                value = str(value)
            table.add_row(stat, value)
        return table

    def render_stats(self, live: Live):
        live.update(self.populate_table())

    def run(self) -> None:
        """Run the algorithm's main loop."""
        continue_running = True
        while continue_running:
            continue_running = self.run_episode()
    
    def run_episode(self) -> bool:
        """Run a single episode of the algorithm."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class NArmedBanditAlgorithm(Algorithm):
    """
    Class for any stateless N-action space algorithm.
    """
    def __init__(self, environment: StatelessEnvironment):
        super().__init__(environment)
        self.environment = environment
        self.action_reward_average: Dict[int, float] = {}
        self.action_reward_variance: Dict[int, float] = {}
        self.action_counts: Dict[int, int] = {}
        self.reset()

    def reset(self) -> None:
        """Reset the environment and the agent's state."""
        self.environment.reset()
        self.training_step = 0
        self.action_reward_average = {action: 0.0 for action in range(self.environment.actions_dimension)}
        self.action_reward_variance = {action: 0.0 for action in range(self.environment.actions_dimension)}
        self.action_counts = {action: 0 for action in range(self.environment.actions_dimension)}

    def update_stats(self, action: int, reward: float) -> None:
        """Perform a step in the environment with the given action and update the average reward."""
        self.action_counts[action] += 1
        self.action_reward_average[action] = (self.action_reward_average.get(action, 0) * (self.action_counts[action] - 1) + reward) / self.action_counts[action]
        self.action_reward_variance[action] = (self.action_reward_variance.get(action, 0) * (self.action_counts[action] - 1) + (reward - self.action_reward_average[action]) ** 2) / self.action_counts[action]
        
        self.stats["Action Counts"] = self.action_counts
        self.stats["Action Reward Average"] = self.action_reward_average
        self.stats["Action Reward Variance"] = self.action_reward_variance      
    
    def run_episode(self) -> bool:
        """Run the algorithm's training process."""
        action = self.select_action()
        new_state, reward, done, info = self.environment.step(action)
        print(f"Action: {action}, Reward: {reward}")
        return True


class StatefulDiscreteActionSpaceAlgorithm(Algorithm, ABC):
    def __init__(self, environment: FiniteActionSpaceEnvironment):
        super().__init__(environment)
        self.environment = environment
        self.stats = {"Actions counts": {}}
    
    def update_base_stats(self, action: Any, reward: float) -> None:
        """Update the base statistics for the algorithm."""
        self.stats["Episode"] = self.episode
        self.stats["Step"] = self.training_step
        self.stats["Actions counts"][action] = self.stats["Actions counts"].get(action, 0) + 1

    