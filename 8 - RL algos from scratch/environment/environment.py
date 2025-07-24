from abc import ABC, abstractmethod
import math
import random
from typing import Any, Callable, List, Optional, Tuple
from matplotlib import pyplot as plt
from numpy import log
import pygame


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset the environment to its initial state."""
        pass

    @abstractmethod
    def state(self, normalized: bool = False) -> Optional[Any]:
        """Get the current state of the environment."""
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
    

class StatelessEnvironment(BaseEnvironment, ABC):
    def __init__(self, **kwargs: Any):
        """Initialize the environment with a finite action space."""
        super().__init__(**kwargs)

    def state(self, normalized: bool = False) -> Optional[Any]:
        return None

    def step(self, action: Optional[int] = None) -> Tuple[Any, float, bool, dict]:
        return None, *self.stateless_step(action)
    
    @abstractmethod
    def stateless_step(self, action: Optional[int] = None) -> Tuple[float, bool, dict]:
        """Perform a step in the environment with the given action."""
        pass

    def reset(self) -> None:
        """nothing to reset in stateless environment"""
        pass


class FiniteActionEnvironment(BaseEnvironment, ABC):
    """Environment with finite action space and state space."""
    def __init__(self, actions_dimension: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.actions_dimension = actions_dimension  # Available actions in the environment [0, actions_dimension - 1]


class StatelessFiniteActionEnvironment(FiniteActionEnvironment, StatelessEnvironment):
    """Environment with finite action space and stateless behavior."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class NArmedBanditEnvironment(StatelessFiniteActionEnvironment):
    """Environment for N-armed bandit problems."""
    def __init__(self, reward_distributions: List[Callable[[], float]], **kwargs: Any):
        super().__init__(**kwargs)
        if len(reward_distributions) != self.actions_dimension:
            raise ValueError("reward_distributions must match the number of actions")
        self.reward_distributions = reward_distributions
    
    def action_reward(self, action: int) -> float:
        return self.reward_distributions[action]()
    
    def stateless_step(self, action: Optional[int] = None) -> Tuple[float, bool, dict]:
        if action is None:
            return 0.0, False, {}
        
        if action < 0 or action >= self.actions_dimension:
            raise ValueError("Invalid action: action must be between 0 and n_actions - 1")
        
        return self.action_reward(action), True, {}
    
    def first_render(self):
        plt.figure(figsize=(14, 7))
        plt.suptitle("Reward Distributions for Each Action", fontsize=16)
        for i in range(self.actions_dimension):
            plt.hist([self.action_reward(i) for _ in range(10000)], bins=100, alpha=0.3, label=f"Action {i}")
        plt.xlabel("Reward value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


class FiniteStateEnvironment(BaseEnvironment, ABC):
    """Environment with finite action space and state space."""
    def __init__(self, state_dimension: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.state_dimension = state_dimension


class ContinuousStateEnvironment(BaseEnvironment, ABC):
    """Environment with infinite state space."""
    def __init__(self, initial_state: Tuple[float, ...], **kwargs: Any):
        super().__init__(**kwargs)
        self.initial_state = initial_state  # Initial state of the environment

    @abstractmethod
    def state(self, normalized: bool = False) -> Tuple[float, ...]:
        """Get the current state of the environment."""
        pass


class FiniteStateFiniteActionEnvironment(FiniteActionEnvironment, FiniteStateEnvironment):
    """Environment with finite action space and finite state space."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class FrozenLakeEnvironment(FiniteStateFiniteActionEnvironment):
    def __init__(self, 
        grid_size: int = 10,
        initial_state: Optional[Tuple[int, int]] = None,
        goal_state: Optional[Tuple[int, int]] = None,
        hole_states: Optional[List[Tuple[int, int]]] = None,
        slip_probability: float = 0.1
    ):
        super().__init__(actions_dimension=4, state_dimension=grid_size**2)
        self.grid_size = grid_size

        if initial_state is None and goal_state is None and hole_states is None:
            self.initial_state, self.goal_state, self.hole_states = self.generate_random_grid(grid_size)
        else:
            self.initial_state = initial_state
            self.goal_state = goal_state
            self.hole_states = hole_states
        
        self.slip_probability = slip_probability
        self.steps = 0
        self.max_steps = grid_size * grid_size * 10  # Maximum steps to prevent infinite loop
        self.reset()
    
    @staticmethod
    def generate_random_grid(grid_size: int, hole_ratio: float = 0.2) -> Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
        initial_state = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        goal_state = initial_state
        while goal_state == initial_state:
            goal_state = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))

        max_holes = int(grid_size * grid_size * hole_ratio)
        hole_states = set()
        while len(hole_states) < max_holes:
            h = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
            if h != initial_state and h != goal_state:
                hole_states.add(h)

        return initial_state, goal_state, list(hole_states)

    def reset(self) -> None:
        """Reset the environment to its initial state."""
        self.x, self.y = self.initial_state
        self.steps = 0
    
    def state(self, normalized: bool = False) -> int:
        """Get the current state of the environment."""
        return self.x + self.y * self.grid_size
    
    def step(self, action: Optional[int] = None) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take an action in the environment and return the result."""
        if action is None:
            return self.state(), 0.0, False, {}

        if self.steps >= self.max_steps:
            return self.state(), -1, True, {}

        # Possible actions: 0=up, 1=down, 2=left, 3=right
        actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        if action not in actions:
            raise ValueError("Invalid action: action must be between 0 and 3")

        dx, dy = actions[action]
        if random.random() < self.slip_probability:
            dx, dy = random.choice(list(actions.values()))
        
        new_x = max(0, min(self.grid_size - 1, self.x + dx))
        new_y = max(0, min(self.grid_size - 1, self.y + dy))

        self.x, self.y = new_x, new_y

        if (self.x, self.y) == self.goal_state:
            return self.state(), 1.0, True, {}
        
        if (self.x, self.y) in self.hole_states:
            return self.state(), -1.0, True, {}
        
        self.steps += 1
        
        return self.state(), -0.05, False, {}
    
    def render(self, max_step_per_second: Optional[int] = None) -> None:
        """Render the current state of the environment in pygame."""

        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * 50, self.grid_size * 50))
            pygame.display.set_caption("Frozen Lake")
            self.clock = pygame.time.Clock()
        
        self.screen.fill((255, 255, 255))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * 50, i * 50, 50, 50)
                if (i, j) == self.goal_state:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)
                elif (i, j) in self.hole_states:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)
                elif (i, j) == self.initial_state:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect)
        player_rect = pygame.Rect(self.y * 50, self.x * 50, 50, 50)
        pygame.draw.rect(self.screen, (0, 0, 255), player_rect) 

        # Display reward and state
        font = pygame.font.Font(None, 36)
        state_text = font.render(f"State: {self.state()}", True, (0, 0, 0))
        self.screen.blit(state_text, (10, 10))

        # Display time step
        time_text = font.render(f"Step: {self.steps}", True, (0, 0, 0))
        self.screen.blit(time_text, (10, 40))

        pygame.display.flip()
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        if max_step_per_second:
            self.clock.tick(max_step_per_second)

    def run(self) -> None:
        """Run the Frozen Lake environment."""
        self.reset()
        running = True
        while running:
            self.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        _, _, done, _ = self.step(0)
                    elif event.key == pygame.K_DOWN:
                        _, _, done, _ = self.step(1)
                    elif event.key == pygame.K_LEFT:
                        _, _, done, _ = self.step(2)
                    elif event.key == pygame.K_RIGHT:
                        _, _, done, _ = self.step(3)
            
                    if done:
                        self.reset()
        pygame.quit()


class ContinousStateFiniteActionEnvironment(FiniteActionEnvironment, ContinuousStateEnvironment):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class StatefulFiniteActionEnvironment(FiniteStateFiniteActionEnvironment, ContinousStateFiniteActionEnvironment):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        

class CartPoleEnvironment(ContinousStateFiniteActionEnvironment):
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
        force_mag=15.0,
        dt=0.02,
        **kwargs: Any):

        super().__init__(actions_dimension=2, initial_state=(0.0, 0.0, 0.0, 0.0), **kwargs)  # x, x_dot, theta, theta_dot
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
        
        self.px_per_meter = self.width / self.track_length

        self.t = 0
        self.max_steps = 500

        self.is_rendering = False
        self.reset()

    def init_render(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CartPole")
        self.clock = pygame.time.Clock()
        self.is_rendering = True

    def reset(self):
        self.x = random.uniform(-0.03, 0.03)
        self.x_dot = random.uniform(-0.03, 0.03)
        self.theta = random.uniform(-0.03, 0.03)
        self.theta_dot = random.uniform(-0.03, 0.03)
        self.t = 0
    
    def state(self, normalized: bool = False) -> Tuple[float, float, float, float]:
        # if normalized:
        #     return (
        #         self.x / (self.track_length / 2),
        #         self.x_dot / 2,
        #         self.theta / (math.pi / 2),
        #         self.theta_dot / 2
        #     )
        return (
            self.x,
            self.x_dot,
            self.theta,
            self.theta_dot
        )

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
        self.t += 1

        timeout = self.t >= self.max_steps

        if timeout:
            return self.state(normalized=True), 10, True, {}
        
        done = abs(self.x) > self.track_length / 2 or abs(self.theta) > math.pi / 3

        reward = 1 - 0.1 * abs(self.x) - 0.1 * abs(self.theta)

        return self.state(normalized=True), reward if not done else 0, done, {}

    def render(self):
        if not self.is_rendering:
            self.init_render()
        
        pygame.event.pump()
        
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
        self.init_render()
        
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
    
    env = FrozenLakeEnvironment()
    env.run()