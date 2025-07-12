
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple

from tqdm import tqdm

from utils.environment_utils import BaseEnvironment, FiniteActionSpaceEnvironment
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

class Algorithm(ABC):
    envirornment: BaseEnvironment
    training_step: int = 0

@dataclass
class NArmedBanditAlgorithm(Algorithm, ABC):
    """
    Class for any stateless N-action space algorithm.
    """
    environment: FiniteActionSpaceEnvironment
    action_reward_average: Dict[int, float] = field(default_factory=dict)
    action_reward_variance: Dict[int, float] = field(default_factory=dict)
    action_counts: Dict[int, int] = field(default_factory=dict)

    @abstractmethod
    def select_action(self) -> int:
        """Select an action based on the current policy."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def update_algorithm(self, action: int, reward: float) -> None:
        """Update the algorithm's state after receiving a reward."""
        pass

    def reset(self) -> None:
        """Reset the environment and the agent's state."""
        self.environment.reset()
        self.training_step = 0
        self.action_reward_average = {action: 0.0 for action in range(self.environment.actions_dimension)}
        self.action_reward_variance = {action: 0.0 for action in range(self.environment.actions_dimension)}
        self.action_counts = {action: 0 for action in range(self.environment.actions_dimension)}

    def update_stats(self, action: int, reward: float) -> float:
        """Perform a step in the environment with the given action and update the average reward."""
        self.action_counts[action] += 1
        self.action_reward_average[action] = (self.action_reward_average.get(action, 0) * (self.action_counts[action] - 1) + reward) / self.action_counts[action]
        self.action_reward_variance[action] = (self.action_reward_variance.get(action, 0) * (self.action_counts[action] - 1) + (reward - self.action_reward_average[action]) ** 2) / self.action_counts[action]
        self.training_step += 1
        return reward        

    def render_table(self) -> Table:
        table = Table(title="Training Stats")
        table.add_column("Metric", justify="left")
        table.add_column("Values", justify="left")

        table.add_row("Step", str(self.training_step))
        table.add_row("Action Counts", str(self.action_counts))
        table.add_row(
            "Rewards Average", 
            ", ".join(f"{k}: {v:.2f}" for k, v in self.action_reward_average.items())
        )
        table.add_row(
            "Rewards Variance", 
            ", ".join(f"{k}: {v:.2f}" for k, v in self.action_reward_variance.items())
        )

        return table

    def init_progress(self, n_steps: int) -> Tuple[Progress, int]:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        task = progress.add_task("Training", total=n_steps)
        return progress, task

    def render_stats(self, live: Live):
        live.update(self.render_table())

    def train_step(self):
        action = self.select_action()
        reward = self.environment.step(action)
        self.update_algorithm(action, reward)
        self.update_stats(action, reward)

    def train(self, n_steps: int, render: bool = False):
        self.reset()
        self.environment.reset()

        progress, task = self.init_progress(n_steps)

        if render:
            print("Training started...")
            print(self.environment.first_render())

        with Live(self.render_table(), refresh_per_second=4) as live:
            for _ in range(n_steps):
                self.train_step()

                if render and self.training_step % 100 == 0:
                    self.render_stats(live)

                if render:
                    progress.update(task, advance=1)

            if render:
                self.render_stats(live)
                print("\nTraining completed.")

    