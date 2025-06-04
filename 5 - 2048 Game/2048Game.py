import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class Game2048:
    def __init__(self):
        self.grid_size = 2
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.reset()
        self.objective_score = 8

    def reset(self):
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.add_new_tile()
        self.add_new_tile()
        return self.grid

    def add_new_tile(self):
        empty_cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.grid[r][c] = 2 if random.random() < 0.9 else 4

    def get_current_state(self):
        for row in self.grid:
            if self.objective_score in row:
                return 'WON'
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == 0:
                    return 'GAME NOT OVER'
                if j < self.grid_size - 1 and self.grid[i][j] == self.grid[i][j + 1]:
                    return 'GAME NOT OVER'
                if i < self.grid_size - 1 and self.grid[i][j] == self.grid[i + 1][j]:
                    return 'GAME NOT OVER'
        return 'LOST'

    def compress(self):
        changed = False
        new_grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        for i in range(self.grid_size):
            pos = 0
            for j in range(self.grid_size):
                if self.grid[i][j] != 0:
                    new_grid[i][pos] = self.grid[i][j]
                    if j != pos:
                        changed = True
                    pos += 1
        self.grid = new_grid
        return changed

    def merge(self):
        changed = False
        merge_score = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                if self.grid[i][j] == self.grid[i][j + 1] and self.grid[i][j] != 0:
                    self.grid[i][j] *= 2
                    self.grid[i][j + 1] = 0
                    merge_score += self.grid[i][j]
                    changed = True
        return changed, merge_score

    def reverse(self):
        for i in range(self.grid_size):
            self.grid[i].reverse()

    def transpose(self):
        self.grid = [list(row) for row in zip(*self.grid)]

    def move_left(self):
        changed1 = self.compress()
        changed2, merge_score = self.merge()
        self.compress()
        return changed1 or changed2, merge_score

    def move_right(self):
        self.reverse()
        changed, merge_score = self.move_left()
        self.reverse()
        return changed, merge_score

    def move_up(self):
        self.transpose()
        changed, merge_score = self.move_left()
        self.transpose()
        return changed, merge_score

    def move_down(self):
        self.transpose()
        changed, merge_score = self.move_right()
        self.transpose()
        return changed, merge_score

    def render(self):
        print(np.array(self.grid))

    def move(self, direction):
        if direction == 0:  # Move right
            changed, merge_score = self.move_right()
        elif direction == 1:  # Move down
            changed, merge_score = self.move_down()
        elif direction == 2:  # Move left
            changed, merge_score = self.move_left()
        elif direction == 3:  # Move up
            changed, merge_score = self.move_up()
        else:
            print("Invalid direction! Use 0 (right), 1 (down), 2 (left), or 3 (up).")
            return False, 0
        if changed:
            self.add_new_tile()
        return changed, merge_score

    def get_score(self):
        return sum(sum(row) for row in self.grid)

    def get_highest_tile(self):
        return max(max(row) for row in self.grid)

class Custom2048Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Custom2048Env, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: right, 1: down, 2: left, 3: up
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2, 2), dtype=np.float32)
        self.game = Game2048()
        self.current_score = 0
        self.steps = 0

    def normalize_grid(self, grid):
        norm_grid = np.zeros_like(grid, dtype=np.float32)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                tile_value = grid[i][j]
                if tile_value > 0:
                    norm_grid[i][j] = np.log2(tile_value) / np.log2(self.game.objective_score)  # Normalize using log2 values
                else:
                    norm_grid[i][j] = 0.0
        return norm_grid

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        self.game.reset()
        self.current_score = 0
        self.steps = 0
        state = self.normalize_grid(self.game.grid)
        return state, {}

    def step(self, action):
        previous_highest = self.game.get_highest_tile()
        changed, merge_score = self.game.move(action)
        new_highest = self.game.get_highest_tile()
        self.steps += 1

        # Normalize merge_score
        normalized_merge_score = 0
        if merge_score > 0:
            normalized_merge_score = np.log2(merge_score) / np.log2(self.game.objective_score)  # Normalize reward

        # Calculate reward
        reward = normalized_merge_score

        if new_highest > previous_highest:
            reward += (np.log2(new_highest) / np.log2(self.game.objective_score))  # Reward for new highest tile

        if not changed:
            reward -= 0.1  # Penalty for invalid move

        done = False
        game_state = self.game.get_current_state()
        if game_state == 'WON':
            reward += 1.0  # Reward for winning
            done = True
            print(f"Game Won! Final Score: {self.game.get_score()}, Highest Tile: {self.game.get_highest_tile()}, Steps: {self.steps}")
        elif game_state == 'LOST':
            reward -= 1.0  # Penalty for losing
            done = True
            print(f"Game Over! Final Score: {self.game.get_score()}, Highest Tile: {self.game.get_highest_tile()}, Steps: {self.steps}")

        state = self.normalize_grid(self.game.grid)
        return state, reward, done, False, {}

    def render(self, mode='human'):
        for row in self.game.grid:
            print('\t'.join(str(num) if num != 0 else '.' for num in row))
        print()

env = Custom2048Env()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
