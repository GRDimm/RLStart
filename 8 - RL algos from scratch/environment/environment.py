from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
import math
import platform
import random
from typing import Any, Callable, List, Optional, Tuple
from matplotlib import pyplot as plt
import pygame

@dataclass
class BaseEnvironment(ABC):
    def reset(self) -> None:
        """Reset the environment to its initial state."""
        pass

    @abstractmethod
    def step(self, action: Optional[Any] = None) -> Tuple[Any, float, bool, dict]:  # State Reward, Done, Info
        """Take an action in the environment and return the result. No action means just make the environment physics (if there is) step."""
        pass

    def render(self) -> None:
        """Render the current state of the environment."""
        pass
    
    def first_render(self) -> None:
        """Render the initial state of the environment."""
        pass

@dataclass
class FiniteActionSpaceStatelessEnvironment(BaseEnvironment, ABC):
    actions_dimension: int  # Available actions in the environment [0, actions_dimension - 1]

    def step(self, action: Optional[int] = None) -> Tuple[Any, float, bool, dict]:
        return None, *self.stateless_step(action)
    
    @abstractmethod
    def stateless_step(self, action: Optional[int] = None) -> Tuple[float, bool, dict]:
        """Perform a step in the environment with the given action."""
        pass


@dataclass
class NArmedBanditEnvironment(FiniteActionSpaceStatelessEnvironment):
    """Environment for N-armed bandit problems."""
    reward_distributions: List[Callable[[], float]]
    
    def action_reward(self, action: int) -> float:
        return self.reward_distributions[action]()
    
    def stateless_step(self, action: Optional[int] = None) -> Tuple[float, bool, dict]:
        if action is None:
            return 0.0, False, {}
        
        if action < 0 or action >= self.actions_dimension:
            raise ValueError("Invalid action: action must be between 0 and n_actions - 1")
        
        return self.action_reward(action), True, {}
    
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


@dataclass
class FiniteActionSpaceEnvironment(BaseEnvironment, ABC):
    """Environment with finite action space and state space."""
    actions_dimension: int  # Available actions in the environment [0, actions_dimension - 1]
    state: Tuple[float, ...]  # Environment's state


class CartPoleEnvironment(FiniteActionSpaceEnvironment):
    def __init__(self,
        screen_width=1920,
        screen_height=1080,
        track_length=2.4,
        cart_width=60,
        cart_height=30,
        pole_length=150,
        cart_color=(0, 0, 0),
        pole_color=(200, 0, 0),
        bg_color=(255, 255, 255),
        gravity=9.8,
        cart_mass=1.0,
        pole_mass=0.1,
        force_mag=10.0,
        dt=0.02):

        self.width = screen_width
        self.height = screen_height
        self.track_length = track_length
        self.cart_width = cart_width
        self.cart_height = cart_height
        self.pole_length_px = pole_length
        self.cart_color = cart_color
        self.pole_color = pole_color
        self.bg_color = bg_color

        self.manual_control = True
        self.manual_force = 0.0

        self.gravity = gravity
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.force_mag = force_mag
        self.dt = dt

        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_length = 0.5  # in meters (half-length)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CartPole")
        self.clock = pygame.time.Clock()

        self.px_per_meter = self.width / self.track_length
        self.reset()

    def reset(self):
        self.x = random.uniform(-0.05, 0.05)
        self.x_dot = random.uniform(-0.05, 0.05)
        self.theta = random.uniform(-0.05, 0.05)
        self.theta_dot = random.uniform(-0.05, 0.05)

    def step(self, action: Optional[int] = None) -> Tuple[Tuple[float, ...], float, bool, dict]:
        if action == 0:
            force = self.force_mag
        elif action == 1:
            force = -self.force_mag
        elif action is None:
            force = 0.0
        else:
            raise ValueError("Invalid action: action must be 0 (left) or 1 (right)")

        force = self.manual_force if self.manual_control else force

        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        temp = (force + self.pole_mass * self.pole_length * self.theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_length * (4/3 - self.pole_mass * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass * self.pole_length * theta_acc * cos_theta / self.total_mass

        self.x += self.dt * self.x_dot
        self.x_dot += self.dt * x_acc
        self.theta += self.dt * self.theta_dot
        self.theta_dot += self.dt * theta_acc

        done = abs(self.theta) > math.pi / 2 or abs(self.x) > self.track_length / 2
        state = (self.x, self.x_dot, self.theta, self.theta_dot)
        return state, 1 if not done else 0, done, {}

    def render(self):
        self.screen.fill(self.bg_color)

        base_y = self.height - 100
        cart_x_px = int(self.width / 2 + self.x * self.px_per_meter)
        cart_y_px = base_y - self.cart_height

        pygame.draw.rect(
            self.screen,
            self.cart_color,
            (cart_x_px - self.cart_width // 2, cart_y_px, self.cart_width, self.cart_height)
        )

        pole_x_end = cart_x_px + self.pole_length_px * math.sin(self.theta)
        pole_y_end = cart_y_px - self.pole_length_px * math.cos(self.theta)

        pygame.draw.line(
            self.screen,
            self.pole_color,
            (cart_x_px, cart_y_px),
            (pole_x_end, pole_y_end),
            6
        )

        # Draw the track
        pygame.draw.rect(   
            self.screen,
            (0, 0, 0),
            (self.width / 2 - self.track_length * self.px_per_meter / 2, base_y - 10, self.track_length * self.px_per_meter, 20)
        )

        pygame.display.flip()
        self.clock.tick(60)

    def run(self):
        running = True
        while running:
            if self.manual_control:
                keys = pygame.key.get_pressed()
                self.manual_force = 0.0
                if keys[pygame.K_LEFT]:
                    self.manual_force = -self.force_mag
                elif keys[pygame.K_RIGHT]:
                    self.manual_force = self.force_mag
            state, reward, done, info = self.step()
            if done:
                self.reset()

            self.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()

if __name__ == "__main__":
    pygame.init()
    env = CartPoleEnvironment()
    env.run()