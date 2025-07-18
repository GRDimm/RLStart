
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
from environment.environment import BaseEnvironment, FiniteActionSpaceEnvironment, FiniteActionSpaceStatelessEnvironment
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


class Algorithm(ABC):
    def __init__(self, environment: BaseEnvironment):
        self.environment = environment
        self.episode: int = 0
        self.training_step: int = 0
        self.stats: Dict[str, Any] = {}

    def train(self, n_episodes: int, render: bool = False):
        self.reset()
        self.environment.reset()

        progress, task = self.init_progress(n_episodes)

        if render:
            self.environment.first_render()

        with Live(self.populate_table(), refresh_per_second=4) as live:
            for _ in range(n_episodes):
                self.train_episode()
                self.render_stats(live)
                progress.update(task, advance=1)
        self.run()

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment and the algorithm's state."""
        pass
    
    @abstractmethod
    def select_action(self) -> int:
        """Select an action based on the current policy."""
        pass

    def train_episode(self) -> None:
        """Perform a single training episode."""
        done = False
        while not done:
            action = self.select_action()
            new_state, reward, done, info = self.environment.step(action)
            self.update_algorithm_step(action, reward)
            self.update_stats(action, reward)
            self.training_step += 1
        self.episode += 1

    def update_stats(self, action: Any, reward: float) -> None:
        pass

    def update_algorithm_step(self, action: Any, reward: float) -> None:
        """Update the algorithm's state after receiving a reward."""
        pass

    def init_progress(self, n_episodes: int) -> Tuple[Progress, int]:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        task = progress.add_task("Training", total=n_episodes)
        return progress, task

    def populate_table(self) -> Table:
        table = Table(title="Training Stats")
        table.add_column("Metric", justify="left")
        table.add_column("Values", justify="left")

        table.add_row("Episode", str(self.episode))
        table.add_row("Training Step", str(self.training_step))

        for stat, value in self.stats.items():
            if isinstance(value, dict):
                value = ", ".join(f"{k}: {v:.2f}" for k, v in value.items())
            elif isinstance(value, list):
                value = ", ".join(f"{v:.2f}" for v in value)
            else:
                value = str(value)
            table.add_row(stat, value)
        return table

    def render_stats(self, live: Live):
        live.update(self.populate_table())

    def run(self) -> None:
        """Run the algorithm's main loop."""
        for _ in range(10):
            self.run_episode()
    
    def run_episode(self) -> None:
        """Run a single episode of the algorithm."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class NArmedBanditAlgorithm(Algorithm):
    """
    Class for any stateless N-action space algorithm.
    """
    def __init__(self, environment: FiniteActionSpaceStatelessEnvironment):
        super().__init__(environment)
        self.environment = environment
        self.training_step: int = 0
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
    
    def run_episode(self) -> None:
        """Run the algorithm's training process."""
        action = self.select_action()
        new_state, reward, done, info = self.environment.step(action)
        print(f"Action: {action}, Reward: {reward}")


@dataclass
class StatefulDiscreteActionSpaceAlgorithm(Algorithm, ABC):
    environment: FiniteActionSpaceEnvironment

    @abstractmethod
    def select_action(self, state: Tuple[float, ...]) -> int:
        """Select an action based on the current policy."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @abstractmethod
    def update_algorithm_step(self, action: int, new_state: Tuple[float, ...], reward: float) -> None:
        """Update the algorithm after receiving a reward."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def train(self, n_steps, render: bool = False) -> None:
        self.reset()
        self.environment.reset()

        progress, task = self.init_progress(n_steps)

        if render:
            print("Training started...")
            print(self.environment.first_render())

        with Live(self.populate_table(), refresh_per_second=4) as live:
            for _ in range(n_steps):
                action = self.select_action(self.environment.state)
                new_state, reward, done, info = self.environment.step(action)
                self.update_algorithm_step(action, new_state, reward)

                if render and self.training_step % 100 == 0:
                    self.render_stats(live)

                if render:
                    progress.update(task, advance=1)

            if render:
                self.render_stats(live)
                print("\nTraining completed.")