# main.py
from tkinter import messagebox
import numpy as np
import tkinter as tk
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from core import Checkers  # le Checkers UI complet fourni par l'utilisateur

TRAIN = False     # Passer à False pour charger le modèle et jouer via UI
MODEL_PATH = "checkers_model_100000_steps.zip"

class PlayableCheckers(Checkers):
    def __init__(self, ai_model):
        super().__init__()
        self.model = ai_model
        self.n_squares = self.rows * self.cols

    def action_masks(self):
        mask = np.zeros(self.n_squares * self.n_squares, dtype=bool)
        moves = self.get_moves(self.current_player)
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
        self.n_squares = self.rows * self.cols  # 100
        self.action_space = spaces.Discrete(self.n_squares * self.n_squares)
        self.observation_space = spaces.Box(
            low=-2, high=2,
            shape=(self.rows, self.cols),
            dtype=np.int8
        )

    def reset(self, *, seed=None, options=None):
        super().reset()
        return np.array(self.board, dtype=np.int8), {}

    def step(self, action):
        moves = self.get_moves(self.current_player)
        if not moves:
            # no moves → immediate loss
            reward = -1
            return np.array(self.board, dtype=np.int8), reward, True, False, {}

        a = int(action)
        src_idx = a // self.n_squares
        dst_idx = a % self.n_squares
        src_row, src_col = src_idx // self.cols, src_idx % self.cols
        dst_row, dst_col = dst_idx // self.cols, dst_idx % self.cols
        src = (src_col, src_row)
        dst = (dst_col, dst_row)

        seqs = moves.get(src)
        if seqs:
            chosen = next((seq for seq in seqs if seq[-1] == dst), None)
            if chosen:
                # count opponent pieces before move
                opponent = self.opponent(self.current_player)
                pre_count = sum(1 for row in self.board for p in row if p == opponent or p == opponent * 2)

                # check for promotion opportunity
                started_piece = self.board[src_row][src_col]
                promoting_row = 0 if started_piece > 0 else self.rows - 1

                self.play(chosen)

                # count opponent pieces after move
                post_count = sum(1 for row in self.board for p in row if p == opponent or p == opponent * 2)
                captured = pre_count - post_count

                # reward for captures
                reward = 0.0
                if captured > 0:
                    reward += 0.1 * captured

                # reward for promotion
                end_x, end_y = chosen[-1]
                if abs(started_piece) == 1 and end_y == promoting_row:
                    reward += 0.5

                # check terminal after move
                next_moves = self.get_moves(self.opponent(self.current_player))
                if not next_moves:
                    # current player has won
                    reward = 1 if self.current_player > 0 else -1
                    return np.array(self.board, dtype=np.int8), reward, True, False, {}

                self.current_player = self.opponent(self.current_player)
                return np.array(self.board, dtype=np.int8), reward, False, False, {}

        # illegal move
        raise ValueError(f"Illegal action attempted: src={src}, dst={dst}")

    def render(self):
        pass

    def action_masks(self):
        mask = np.zeros(self.n_squares * self.n_squares, dtype=bool)
        moves = self.get_moves(self.current_player)
        for (x, y), seqs in moves.items():
            src_idx = y * self.cols + x
            for seq in seqs:
                dx, dy = seq[-1]
                dst_idx = dy * self.cols + dx
                mask[src_idx * self.n_squares + dst_idx] = True
        return mask


if TRAIN:
    base_env = CheckersEnv()
    env = ActionMasker(base_env, action_mask_fn=lambda wrapped: wrapped.action_masks())
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path="./", name_prefix=MODEL_PATH)
    model.learn(total_timesteps=500_000, callback=checkpoint_cb)
    model.save(f"{MODEL_PATH}_final")
    print("Training complete")

else:
    model = MaskablePPO.load(f"{MODEL_PATH}")
    player = PlayableCheckers(model)
    player.display_ui()
