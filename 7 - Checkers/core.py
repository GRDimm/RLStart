# core.py
import tkinter as tk
import random

class Checkers:
    def __init__(self):
        self.size = 60
        self.rows = self.cols = 10
        self.dirs = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        self.reset()
        self.selected = None
        self.valid_moves = {}

    def reset(self):
        self.history = []
        self.board = [[0] * self.cols for _ in range(self.rows)]
        for y in range(4):
            for x in range(self.cols):
                if (x + y) % 2:
                    self.board[y][x] = -1
        for y in range(6, 10):
            for x in range(self.cols):
                if (x + y) % 2:
                    self.board[y][x] = 1
        self.current_player = 1

    def inside(self, x, y):
        return 0 <= x < self.cols and 0 <= y < self.rows

    def opponent(self, p):
        return -p

    def is_king(self, p):
        return abs(p) == 2

    def promote_if_needed(self, x, y, p):
        if p == 1 and y == 0: return 2
        if p == -1 and y == self.rows - 1: return -2
        return p

    def get_moves_from(self, player, x, y):
        p = self.board[y][x]
        if p == 0 or (p > 0) != (player > 0):
            return []
        all_caps = {}
        max_caps = 0
        for yy in range(self.rows):
            for xx in range(self.cols):
                pp = self.board[yy][xx]
                if pp == 0 or (pp > 0) != (player > 0):
                    continue
                caps = self.find_captures(xx, yy, pp)
                if caps:
                    all_caps[(xx, yy)] = caps
                    d = max(len(seq) - 1 for seq in caps)
                    if d > max_caps:
                        max_caps = d
        if max_caps:
            caps = all_caps.get((x, y), [])
            return [seq for seq in caps if len(seq) - 1 == max_caps]
        steps = self.find_steps(x, y, p)
        return [[(x, y), (nx, ny)] for nx, ny in steps]

    def get_moves(self, player):
        moves = {}
        all_caps = {}
        max_caps = 0
        for y in range(self.rows):
            for x in range(self.cols):
                p = self.board[y][x]
                if p == 0 or (p > 0) != (player > 0):
                    continue
                caps = self.find_captures(x, y, p)
                if caps:
                    all_caps[(x, y)] = caps
                    max_caps = max(max_caps, max(len(seq) - 1 for seq in caps))
        if max_caps:
            for src, seqs in all_caps.items():
                best = [seq for seq in seqs if len(seq) - 1 == max_caps]
                if best:
                    moves[src] = best
            return moves
        for y in range(self.rows):
            for x in range(self.cols):
                p = self.board[y][x]
                if p == 0 or (p > 0) != (player > 0):
                    continue
                steps = self.find_steps(x, y, p)
                if steps:
                    moves[(x, y)] = [[(x, y), (nx, ny)] for nx, ny in steps]
        return moves

    def find_steps(self, x, y, p):
        res = []
        if self.is_king(p):
            for dx, dy in self.dirs:
                nx, ny = x + dx, y + dy
                while self.inside(nx, ny) and self.board[ny][nx] == 0:
                    res.append((nx, ny))
                    nx += dx; ny += dy
        else:
            dirs = [(-1, -1), (1, -1)] if p > 0 else [(-1, 1), (1, 1)]
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if self.inside(nx, ny) and self.board[ny][nx] == 0:
                    res.append((nx, ny))
        return res

    def find_captures(self, x, y, p):
        sequences = []
        self._dfs_capture(x, y, p, [(x, y)], sequences)
        return sequences

    def _dfs_capture(self, x, y, p, path, sequences):
        found = False
        if self.is_king(p):
            for dx, dy in self.dirs:
                nx, ny = x + dx, y + dy
                while self.inside(nx, ny) and self.board[ny][nx] == 0:
                    nx += dx; ny += dy
                if not (self.inside(nx, ny) and self.board[ny][nx] != 0 and (self.board[ny][nx] > 0) != (p > 0)):
                    continue
                cx, cy = nx, ny
                tx, ty = cx + dx, cy + dy
                while self.inside(tx, ty) and self.board[ty][tx] == 0:
                    captured = self.board[cy][cx]
                    self.board[y][x] = 0
                    self.board[cy][cx] = 0
                    self.board[ty][tx] = p
                    path.append((tx, ty))
                    self._dfs_capture(tx, ty, p, path, sequences)
                    path.pop()
                    self.board[ty][tx] = 0
                    self.board[cy][cx] = captured
                    self.board[y][x] = p
                    found = True
                    tx += dx; ty += dy
        else:
            for dx, dy in self.dirs:
                nx, ny = x + dx, y + dy
                mx, my = nx + dx, ny + dy
                if not (self.inside(nx, ny) and self.inside(mx, my)):
                    continue
                if self.board[ny][nx] != 0 and (self.board[ny][nx] > 0) != (p > 0) and self.board[my][mx] == 0:
                    captured = self.board[ny][nx]
                    self.board[y][x] = 0
                    self.board[ny][nx] = 0
                    self.board[my][mx] = p
                    path.append((mx, my))
                    self._dfs_capture(mx, my, p, path, sequences)
                    path.pop()
                    self.board[my][mx] = 0
                    self.board[ny][nx] = captured
                    self.board[y][x] = p
                    found = True
        if not found and len(path) > 1:
            sequences.append(path.copy())

    def play(self, sequence, replay=False):
        if not replay:
            self.history.append((self.current_player, sequence))
        p = self.board[sequence[0][1]][sequence[0][0]]
        x0, y0 = sequence[0]
        self.board[y0][x0] = 0
        for i in range(len(sequence) - 1):
            x, y = sequence[i]
            nx, ny = sequence[i + 1]
            dx = (nx - x) // abs(nx - x)
            dy = (ny - y) // abs(ny - y)
            tx, ty = x + dx, y + dy
            while (tx, ty) != (nx, ny):
                if self.board[ty][tx] != 0 and (self.board[ty][tx] > 0) != (p > 0):
                    self.board[ty][tx] = 0
                    break
                tx += dx; ty += dy
        x_end, y_end = sequence[-1]
        self.board[y_end][x_end] = self.promote_if_needed(x_end, y_end, p)

    def is_terminal(self, player):
        return not self.get_moves(player)

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
                    self.current_player = self.opponent(self.current_player)
                    break
            self.selected = None
        self.update_display()

    def update_display(self):
        self.canvas.delete('all')
        for y in range(self.rows):
            for x in range(self.cols):
                color = 'saddle brown' if (x + y) % 2 else 'burlywood'
                if self.selected == (x, y):
                    color = 'yellow'
                elif self.selected:
                    for seq in self.valid_moves.get(self.selected, []):
                        if seq[-1] == (x, y):
                            color = 'light green'
                            break
                self.canvas.create_rectangle(
                    x * self.size, y * self.size,
                    (x + 1) * self.size, (y + 1) * self.size,
                    fill=color, outline='black'
                )
                p = self.board[y][x]
                if p:
                    col = 'black' if p > 0 else 'red'
                    r = self.size // 2 - 5
                    cx, cy = x * self.size + self.size // 2, y * self.size + self.size // 2
                    self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=col, outline='white')
                    if self.is_king(p):
                        self.canvas.create_text(cx, cy, text='K', fill='white', font=('Arial', 16, 'bold'))

    def play_random(self, n):
        for _ in range(n):
            moves = self.get_moves(self.current_player)
            if not moves:
                break
            src = random.choice(list(moves))
            seq = random.choice(moves[src])
            self.history.append((self.current_player, seq))
            self.play(seq)
            self.current_player = self.opponent(self.current_player)

    def display_ui(self):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.cols * self.size, height=self.rows * self.size)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.on_click)
        self.update_display()
        self.root.mainloop()
    
    def autoplay(self, history, interval=0):
        self.reset()
        self.history = history[:]
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.cols*self.size, height=self.rows*self.size)
        self.canvas.pack()
        self._auto_index = 0

        def _play_next():
            if self._auto_index >= len(self.history):
                self.root.destroy()
                return
            print(f"Playing step {self._auto_index + 1}/{len(self.history)}")
            player, seq = self.history[self._auto_index]
            moves = self.get_moves(player)
            legal_seqs = [s for seqs in moves.values() for s in seqs]
            if seq not in legal_seqs:
                raise ValueError(f"Illegal move at step {self._auto_index}: {seq}")
            self.current_player = player
            self.play(seq, replay=True)
            self.update_display()
            self._auto_index += 1
            self.root.after(interval, _play_next)

        self.root.after(interval, _play_next)
        self.root.mainloop()


if __name__ == "__main__":
    game = Checkers()
    # history = [   (1, [(5, 6), (6, 5)]),
    # (-1, [(4, 3), (3, 4)]),
    # (1, [(6, 5), (7, 4)]),
    # (-1, [(8, 3), (6, 5)]),
    # (1, [(7, 6), (5, 4)]),
    # (-1, [(6, 3), (4, 5)]),
    # (1, [(3, 6), (5, 4)]),
    # (-1, [(7, 2), (6, 3)]),
    # (1, [(5, 4), (7, 2)]),
    # (-1, [(6, 1), (8, 3)]),
    # (1, [(1, 6), (2, 5)]),
    # (-1, [(3, 4), (1, 6)]),
    # (1, [(0, 7), (2, 5)]),
    # (-1, [(8, 3), (9, 4)]),
    # (1, [(2, 7), (3, 6)]),
    # (-1, [(9, 4), (8, 5)]),
    # (1, [(9, 6), (7, 4)]),
    # (-1, [(9, 2), (8, 3)]),
    # (1, [(7, 4), (9, 2)]),
    # (-1, [(7, 0), (6, 1)]),
    # (1, [(9, 2), (7, 0)]),
    # (-1, [(2, 3), (1, 4)]),
    # (1, [(6, 7), (5, 6)]),
    # (-1, [(1, 4), (0, 5)]),
    # (1, [(7, 0), (8, 1)]),
    # (-1, [(9, 0), (7, 2)]),
    # (1, [(5, 6), (6, 5)]),
    # (-1, [(0, 5), (1, 6)]),
    # (1, [(2, 5), (0, 7)]),
    # (-1, [(7, 2), (6, 3)]),
    # (1, [(3, 6), (2, 5)]),
    # (-1, [(6, 3), (5, 4)]),
    # (1, [(6, 5), (4, 3)]),
    # (-1, [(5, 2), (3, 4), (1, 6)]),
    # (1, [(0, 7), (2, 5)]),
    # (-1, [(3, 2), (4, 3)]),
    # (1, [(2, 5), (1, 4)]),
    # (-1, [(0, 3), (2, 5)]),
    # (1, [(8, 7), (7, 6)]),
    # (-1, [(6, 1), (5, 2)]),
    # (1, [(4, 7), (3, 6)]),
    # (-1, [(2, 5), (4, 7)]),
    # (1, [(5, 8), (3, 6)]),
    # (-1, [(4, 3), (3, 4)]),
    # (1, [(6, 9), (5, 8)]),
    # (-1, [(5, 0), (6, 1)]),
    # (1, [(3, 8), (4, 7)]),
    # (-1, [(3, 4), (2, 5)]),
    # (1, [(3, 6), (1, 4)]),
    # (-1, [(5, 2), (4, 3)]),
    # (1, [(1, 4), (0, 3)]),
    # (-1, [(6, 1), (5, 2)]),
    # (1, [(2, 9), (3, 8)]),
    # (-1, [(4, 3), (5, 4)]),
    # (1, [(7, 8), (6, 7)]),
    # (-1, [(5, 4), (6, 5)]),
    # (1, [(7, 6), (5, 4)]),
    # (-1, [(4, 1), (3, 2)]),
    # (1, [(5, 4), (4, 3)]),
    # (-1, [(5, 2), (3, 4)]),
    # (1, [(9, 8), (8, 7)]),
    # (-1, [(3, 4), (2, 5)]),
    # (1, [(8, 9), (7, 8)]),
    # (-1, [(3, 2), (2, 3)]),
    # (1, [(8, 7), (7, 6)]),
    # (-1, [(2, 3), (1, 4)]),
    # (1, [(4, 7), (3, 6)]),
    # (-1, [(2, 5), (4, 7), (6, 9), (8, 7), (6, 5)]),
    # (1, [(0, 3), (2, 5)]),
    # (-1, [(6, 5), (7, 6)]),
    # (1, [(6, 7), (8, 5)]),
    # (-1, [(1, 2), (0, 3)]),
    # (1, [(1, 8), (2, 7)]),
    # (-1, [(0, 3), (1, 4)]),
    # (1, [(2, 5), (0, 3)]),
    # (-1, [(3, 0), (4, 1)]),
    # (1, [(2, 7), (3, 6)]),
    # (-1, [(2, 1), (1, 2)]),
    # (1, [(0, 3), (2, 1)]),
    # (-1, [(1, 0), (3, 2)]),
    # (1, [(8, 5), (9, 4)]),
    # (-1, [(3, 2), (2, 3)])]

    # game.autoplay(history)

    game.board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

    game.display_ui()
