import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.logger import configure
import random

logger = configure(folder="./logs", format_strings=["stdout", "log"])

class CustomExplorer(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(CustomExplorer, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 10, shape=(2,), dtype=np.float32)
        self.dice_value = random.randint(1, 10)
        self.rounds_left = 10

    def render(self, mode='human'):
        pass

    def reset(self, **kwargs):
        self.dice_value = random.randint(1, 10)
        self.rounds_left = 10
        self.state = np.array([self.rounds_left, self.dice_value], dtype=np.float32)
        return self.state, {}

    def print_state(self):
        print(f'Rolls left : {self.rounds_left}, Dice value : {self.dice_value}')
        
    def step(self, action):
        if self.rounds_left > 0:
            if action == 0:
                reward = 0
                done = False
                truncated = False
            elif action == 1:
                reward = self.dice_value - 5
                done = True
                truncated = False
            self.dice_value = random.randint(1, 10)
            self.rounds_left -= 1
            state = [self.rounds_left, self.dice_value]
        else:
            reward = 0
            done = True
            truncated = False
            state = [0, 0]

        if done:
            print(f'End reward : {reward}')

        self.state = state
        return state, reward, done, truncated, {}

def print_strategy(model, rounds=10):
    for rounds_left in range(rounds, -1, -1):
        for dice_value in range(1, 11):
            observation = np.array([rounds_left, dice_value], dtype=np.float32)
            action, _ = model.predict(observation, deterministic=True)
            if action == 1:
                print(f"For {rounds_left} rounds left, the model does action 1 if the dice value >= {dice_value}")
                break

def expected_reward(model, rounds=10, simulations=10000):
    total_reward = 0
    for _ in range(simulations):
        rounds_left = rounds
        cumulative_reward = 0
        while rounds_left > 0:
            dice_value = random.randint(1, 10)
            observation = np.array([rounds_left, dice_value], dtype=np.float32)
            action, _ = model.predict(observation, deterministic=True)
            if action == 1:
                cumulative_reward += dice_value - 5
                break
            rounds_left -= 1
        total_reward += cumulative_reward
    expected_reward = total_reward / simulations
    print(f"Expected reward with the strategy: {expected_reward}")

models_dir = "./models/"
model_name = "DiceCustom"

import time
use_last_trained = False

env = CustomExplorer()

env.reset()

if use_last_trained:
    model = PPO.load(models_dir + model_name)
else:
    model = PPO("MlpPolicy", env, verbose=2, tensorboard_log = './tensorboard_log')
    model.learn(total_timesteps=150000, tb_log_name = "PPO")
    model.save(models_dir + model_name)

print_strategy(model)
expected_reward(model)

while True:
    print("START")
    env = CustomExplorer()
    observation, _info = env.reset()
    timesteps = 0
    done = False
    while not done:
        env.print_state()
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
        timesteps += 1
    env.print_state()
    time.sleep(1)
