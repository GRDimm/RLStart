from abc import ABC
from dataclasses import dataclass

@dataclass
class BaseEnvironment(ABC):
    action_space: int

    def reset(self) -> None:
        """Reset the environment to its initial state."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def step(self, action: int) -> float:
        """Take an action in the environment and return the result."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def first_render(self) -> None:
        """Render the initial state of the environment."""
        raise NotImplementedError("This method should be overridden by subclasses.")
