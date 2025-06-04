import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

class CustomExplorer(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=6, m=10):
        super(CustomExplorer, self).__init__()
        self.n = n  # Number of faces on the die (faces from 0 to n-1)
        self.m = m  # Multiple that ends the game
        self.action_space = spaces.Discrete(2)  # 0: Continue, 1: Stop
        # Set practical maximums for current_sum and rolls
        self.max_sum = 100  # Adjust if necessary
        self.max_rolls = 20  # Adjust if necessary
        # Observation space: [current_sum, current_sum % m, rolls]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1]),
            high=np.array([self.max_sum, self.m - 1, self.max_rolls]),
            dtype=np.float32)
        self.current_sum = 0
        self.rolls = 0

    def reset(self, **kwargs):
        self.current_sum = 0
        self.rolls = 0
        state = np.array([self.current_sum, self.current_sum % self.m, self.rolls + 1], dtype=np.float32)
        return state, {}

    def render(self, mode='human'):
        pass

    def print_state(self):
        print(f'Current Sum: {self.current_sum}, Rolls: {self.rolls}')

    def step(self, action):
        if action == 0:  # Continue
            face_value = random.randint(0, self.n - 1)
            self.current_sum += face_value
            self.rolls += 1
            if self.current_sum != 0 and self.current_sum % self.m == 0:
                reward = 0
                done = True
                truncated = False
                print(f"Game over! Sum {self.current_sum} is a multiple of {self.m}.")
            elif self.current_sum >= self.max_sum or self.rolls >= self.max_rolls:
                # To prevent exceeding the observation space limits
                reward = self.current_sum
                done = True
                truncated = False
                print(f"Maximum sum or rolls reached. Sum is {self.current_sum}, Rolls: {self.rolls}.")
            else:
                reward = 0
                done = False
                truncated = False
        elif action == 1:  # Stop
            reward = self.current_sum
            done = True
            truncated = False
            print(f"Player stopped. Sum is {self.current_sum}, Rolls: {self.rolls}.")
        else:
            raise ValueError("Invalid action")

        state = np.array([self.current_sum, self.current_sum % self.m, self.rolls], dtype=np.float32)
        return state, reward, done, truncated, {}

def plot_strategy_3d(model, max_rolls=40):
    print("Creating 3D plot of the model's strategy with axes (Rolls, Rolls % m, Profit)...")

    n = env.n
    m = env.m
    max_face_value = n - 1

    # Initialize lists to hold data points
    rolls_list = []
    rolls_mod_m = []
    profits = []
    actions = []

    # Generate data points
    for rolls in range(1, max_rolls + 1):
        min_sum = 0
        max_sum = max_face_value * rolls

        # Generate possible current_sum values for the given number of rolls
        current_sums = np.arange(min_sum, max_sum + 1)

        for current_sum in current_sums:
            # Check if current_sum is achievable with the given number of rolls
            min_possible_sum = 0 * rolls
            max_possible_sum = max_face_value * rolls
            if not (min_possible_sum <= current_sum <= max_possible_sum):
                continue

            # Calculate mod_m
            mod_m = rolls % m

            # Create observation
            observation = np.array([current_sum, current_sum % m, rolls], dtype=np.float32)

            # Predict action
            action, _ = model.predict(observation, deterministic=True)

            # Collect data
            rolls_list.append(rolls)
            rolls_mod_m.append(mod_m)
            profits.append(current_sum)
            actions.append(action)

    # Convert lists to numpy arrays
    rolls_list = np.array(rolls_list)
    rolls_mod_m = np.array(rolls_mod_m)
    profits = np.array(profits)
    actions = np.array(actions)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Map actions to colors
    colors = ['green' if action == 0 else 'red' for action in actions]

    # Scatter plot
    scatter = ax.scatter(rolls_list, rolls_mod_m, profits, c=colors, alpha=0.6)

    # Set labels
    ax.set_xlabel('Rolls')
    ax.set_ylabel('Rolls % m')
    ax.set_zlabel('Profit (current_sum)')

    # Set title
    ax.set_title('Model Strategy Visualization (Red: Stop, Green: Continue)')

    # Show plot
    plt.show()

def plot_strategy_2d(model, max_rolls=40):
    print("Creating 2D plot of the model's strategy with axes (Rolls, Profit)...")

    n = env.n
    max_face_value = n - 1

    # Initialize lists to hold data points
    rolls_list = []
    profits = []
    actions = []

    # Generate data points
    for rolls in range(1, max_rolls + 1):
        min_sum = 0
        max_sum = max_face_value * rolls

        # Generate possible current_sum values for the given number of rolls
        current_sums = np.arange(min_sum, max_sum + 1)

        for current_sum in current_sums:
            # Check if current_sum is achievable with the given number of rolls
            if current_sum < rolls * 0 or current_sum > rolls * max_face_value:
                continue

            # Create observation
            observation = np.array([current_sum, current_sum % env.m, rolls], dtype=np.float32)

            # Predict action
            action, _ = model.predict(observation, deterministic=True)

            # Collect data
            rolls_list.append(rolls)
            profits.append(current_sum)
            actions.append(action)

    # Convert lists to numpy arrays
    rolls_list = np.array(rolls_list)
    profits = np.array(profits)
    actions = np.array(actions)

    # Map actions to colors
    colors = ['green' if action == 0 else 'red' for action in actions]

    # Create 2D scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(rolls_list, profits, c=colors, alpha=0.6)

    # Set labels
    plt.xlabel('Rolls')
    plt.ylabel('Profit (current_sum)')

    # Set title
    plt.title('Model Strategy Visualization (Red: Stop, Green: Continue)')

    # Show plot
    plt.show()

def expected_reward(model, simulations=10000):
    total_reward = 0
    for _ in range(simulations):
        observation, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, _ = env.step(action)
        total_reward += reward
    expected_reward = total_reward / simulations
    print(f"Expected reward with the strategy: {expected_reward}")

models_dir = "./models/"
model_name = "DiceCustom"

import time
use_last_trained = True

# Create the environment with desired n and m
env = CustomExplorer(n=10, m=10)

env.reset()

if use_last_trained:
    model = PPO.load(models_dir + model_name)
else:
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=150000)
    model.save(models_dir + model_name)

# Call the plotting functions
plot_strategy_3d(model, max_rolls=20)
plot_strategy_2d(model, max_rolls=20)

expected_reward(model)

while True:
    print("START")
    env = CustomExplorer(n=6, m=10)
    observation, _info = env.reset()
    done = False
    while not done:
        env.print_state()
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
    env.print_state()
    time.sleep(1)
