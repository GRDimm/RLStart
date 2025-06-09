# main.py
from tkinter import messagebox
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from core import Checkers

TRAIN = False
MODEL_PATH = "checkers_model"

class PlayableCheckers(Checkers):
    def __init__(self, ai_model):
        super().__init__()
        self.model = ai_model
        self.n_squares = self.rows * self.cols

    def action_masks(self):
        mask = np.zeros(self.n_squares * self.n_squares, dtype=bool)
        moves = self.get_moves(self.current_player)
        if not moves:
            return np.ones_like(mask)
        for (x, y), seqs in moves.items():
            src_idx = y * self.cols + x
            for seq in seqs:
                dx, dy = seq[-1]
                dst_idx = dy * self.cols + dx
                mask[src_idx * self.n_squares + dst_idx] = True
        return mask

    def show_winner(self, player):
        color = "Black" if player > 0 else "Red"
        messagebox.showinfo("Game Over", f"{color} wins")

    def on_click(self, event):
        x, y = event.x // self.size, event.y // self.size
        if not self.inside(x, y) or (x + y) % 2 == 0:
            return
        if self.selected is None:
            if self.board[y][x] != 0 and (self.board[y][x] > 0) == (self.current_player > 0):
                self.selected = (x, y)
                self.valid_moves = self.get_moves(self.current_player)
        else:
            seqs = self.valid_moves.get(self.selected, [])
            for seq in seqs:
                if seq[-1] == (x, y):
                    self.play(seq)
                    self.selected = None
                    next_moves = self.get_moves(self.opponent(self.current_player))
                    if not next_moves:
                        self.update_display()
                        self.show_winner(self.current_player)
                        return
                    self.current_player = self.opponent(self.current_player)
                    self.update_display()
                    self.ai_move()
                    return
            self.selected = None
        self.update_display()

    def ai_move(self):
        self.selected = None
        obs = np.array(self.board, dtype=np.int8)
        mask = self.action_masks()
        action, _ = self.model.predict(obs, action_masks=mask)
        a = int(action)
        src_idx = a // self.n_squares
        dst_idx = a % self.n_squares
        src_row, src_col = src_idx // self.cols, src_idx % self.cols
        dst_row, dst_col = dst_idx // self.cols, dst_idx % self.cols
        src = (src_col, src_row)
        dst = (dst_col, dst_row)
        seqs = self.get_moves(self.current_player)[src]
        chosen = next(seq for seq in seqs if seq[-1] == dst)
        self.play(chosen)
        next_moves = self.get_moves(self.opponent(self.current_player))
        if not next_moves:
            self.update_display()
            self.show_winner(self.current_player)
            return
        self.current_player = self.opponent(self.current_player)
        self.update_display()

class CheckersEnv(Checkers, gym.Env):
    def __init__(self):
        super().__init__()
        self.n_squares = self.rows * self.cols
        self.action_space = spaces.Discrete(self.n_squares * self.n_squares)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.rows, self.cols), dtype=np.int8)

    def reset(self, *, seed=None, options=None):
        super().reset()
        return np.array(self.board, dtype=np.int8), {}

    def action_decode(self, action):
        src_idx = action // self.n_squares
        dst_idx = action % self.n_squares
        src_row, src_col = src_idx // self.cols, src_idx % self.cols
        dst_row, dst_col = dst_idx // self.cols, dst_idx % self.cols
        return (src_col, src_row), (dst_col, dst_row), src_row, src_col

    def step(self, action):
        src, dst, src_row, src_col = self.action_decode(int(action))
        seqs = self.get_moves_from(self.current_player, *src)
        if not seqs:
            reward = -1 * self.current_player
            return np.array(self.board, dtype=np.int8), reward, True, False, {}
        chosen = next((seq for seq in seqs if seq[-1] == dst), None)
        if chosen:
            p = self.current_player
            opponent = -p
            pre_count = sum(1 for row in self.board for piece in row if piece == opponent or piece == 2*opponent)
            started_piece = self.board[src_row][src_col]
            promoting_row = 0 if started_piece > 0 else self.rows - 1
            self.play(chosen)
            post_count = sum(1 for row in self.board for piece in row if piece == opponent or piece == 2*opponent)
            captured = pre_count - post_count

            reward = 0.0
            if captured > 0:
                reward += p * 0.1 * captured
            _, end_y = chosen[-1]
            if abs(started_piece) == 1 and end_y == promoting_row:
                reward += p * 0.5

            if self.is_terminal(opponent):
                reward = p * 1.0
                return np.array(self.board, dtype=np.int8), reward, True, False, {}

            self.current_player = opponent
            return np.array(self.board, dtype=np.int8), reward, False, False, {}

        raise ValueError(f"Illegal action attempted: src={src}, dst={dst}")


    def render(self):
        pass

    def action_masks(self, env):
        mask = np.zeros(self.n_squares * self.n_squares, dtype=bool)
        moves = self.get_moves(self.current_player)
        if not moves:
            return np.ones_like(mask)
        for (x, y), seqs in moves.items():
            src = y * self.cols + x
            for seq in seqs:
                dx, dy = seq[-1]
                dst = dy * self.cols + dx
                mask[src * self.n_squares + dst] = True
        return mask

import random
from tqdm import tqdm

def evaluate(model_path, N=100):
    model = MaskablePPO.load(model_path)
    wins = 0
    env = CheckersEnv()
    for _ in tqdm(range(N)):
        obs, _ = env.reset()
        done = False
        while not done:
            if env.current_player > 0:
                mask = env.action_masks(env)
                action, _ = model.predict(obs, action_masks=mask)
            else:
                moves = env.get_moves(env.current_player)
                src, seqs = random.choice(list(moves.items()))
                chosen = random.choice(seqs)
                dst = chosen[-1]
                action = (src[1]*env.cols + src[0]) * env.n_squares + (dst[1]*env.cols + dst[0])
            obs, reward, done, _, _ = env.step(action)
        if env.current_player > 0:
            wins += 1
    print(f"Win rate over {N}: {wins/N:.2%}")

EVALUATE = True

if EVALUATE:
    evaluate("checkers_model_200000_steps.zip", N=200)
else:
    if TRAIN:
        base_env = CheckersEnv()
        env = ActionMasker(base_env, action_mask_fn="action_masks")
        model = MaskablePPO("MlpPolicy", env, verbose=1)
        checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path="./", name_prefix=MODEL_PATH)
        model.learn(total_timesteps=1_000_000, callback=checkpoint_cb)
        model.save(f"{MODEL_PATH}_final")
        print("Training complete")
    else:
        model = MaskablePPO.load(f"{MODEL_PATH}_200000_steps.zip")
        player = PlayableCheckers(model)
        player.board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
        player.display_ui()
