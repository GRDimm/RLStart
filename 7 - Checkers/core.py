import tkinter as tk
import copy
import random

class Checkers:
    def __init__(self):
        self.size = 60
        self.rows = self.cols = 10
        self.reset()
        self.selected = None
        self.valid_moves = {}

    def reset(self):
        self.board = [[0]*self.cols for _ in range(self.rows)]
        for y in range(4):
            for x in range(self.cols):
                if (x+y) % 2:
                    self.board[y][x] = -1
        for y in range(6, 10):
            for x in range(self.cols):
                if (x+y) % 2:
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
        if p == -1 and y == self.rows-1: return -2
        return p

    def get_moves(self, player):
        moves = {}
        all_caps = {}
        max_caps = 0
        for y in range(self.rows):
            for x in range(self.cols):
                p = self.board[y][x]
                if p == 0 or (p > 0) != (player > 0): continue
                caps = self.find_captures(x, y, p, self.board)
                if caps:
                    all_caps[(x, y)] = caps
                    max_caps = max(max_caps, max(len(seq)-1 for seq in caps))
        if max_caps:
            for src, seqs in all_caps.items():
                best = [seq for seq in seqs if len(seq)-1 == max_caps]
                if best:
                    moves[src] = best
            return moves
        for y in range(self.rows):
            for x in range(self.cols):
                p = self.board[y][x]
                if p == 0 or (p > 0) != (player > 0): continue
                steps = self.find_steps(x, y, p)
                if steps:
                    moves[(x, y)] = [[(x, y), (nx, ny)] for nx, ny in steps]
        return moves

    def find_steps(self, x, y, p):
        dirs = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        res = []
        if self.is_king(p):
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                while self.inside(nx, ny) and self.board[ny][nx] == 0:
                    res.append((nx, ny))
                    nx += dx; ny += dy
        else:
            step_dirs = [(-1, -1), (1, -1)] if p > 0 else [(-1, 1), (1, 1)]
            for dx, dy in step_dirs:
                nx, ny = x+dx, y+dy
                if self.inside(nx, ny) and self.board[ny][nx] == 0:
                    res.append((nx, ny))
        return res

    def find_captures(self, x, y, p, board):
        sequences = []
        self._dfs_capture(x, y, p, board, [(x, y)], sequences, set())
        return sequences

    def _dfs_capture(self, x, y, p, board, path, sequences, visited):
        dirs = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        found = False
        if self.is_king(p):
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                while self.inside(nx, ny) and board[ny][nx] == 0:
                    nx += dx; ny += dy
                if self.inside(nx, ny) and board[ny][nx] != 0 and (board[ny][nx] > 0) != (p > 0) and (nx, ny) not in visited:
                    tx, ty = nx+dx, ny+dy
                    while self.inside(tx, ty) and board[ty][tx] == 0:
                        nb = copy.deepcopy(board)
                        nb[y][x] = 0
                        nb[ny][nx] = 0
                        nb[ty][tx] = p
                        self._dfs_capture(tx, ty, p, nb, path+[(tx, ty)], sequences, visited|{(nx, ny)})
                        found = True
                        tx += dx; ty += dy
        else:
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                mx, my = nx+dx, ny+dy
                if self.inside(nx, ny) and self.inside(mx, my):
                    if board[ny][nx] != 0 and (board[ny][nx] > 0) != (p > 0) and board[my][mx] == 0:
                        nb = copy.deepcopy(board)
                        nb[y][x] = 0
                        nb[ny][nx] = 0
                        nb[my][mx] = p
                        self._dfs_capture(mx, my, p, nb, path+[(mx, my)], sequences, visited|{(nx, ny)})
                        found = True
        if not found and len(path) > 1:
            sequences.append(path)

    def play(self, sequence):
        p = self.board[sequence[0][1]][sequence[0][0]]
        x0, y0 = sequence[0]
        self.board[y0][x0] = 0
        for i in range(len(sequence)-1):
            x, y = sequence[i]
            nx, ny = sequence[i+1]
            dx = (nx - x)//abs(nx - x)
            dy = (ny - y)//abs(ny - y)
            tx, ty = x+dx, y+dy
            while (tx, ty) != (nx, ny):
                if self.board[ty][tx] != 0 and (self.board[ty][tx] > 0) != (p > 0):
                    self.board[ty][tx] = 0
                    break
                tx += dx; ty += dy
        x_end, y_end = sequence[-1]
        self.board[y_end][x_end] = self.promote_if_needed(x_end, y_end, p)

    def is_terminal(self):
        # Partie terminée si le joueur actuel n'a aucun coup disponible
        return len(self.get_moves(self.current_player)) == 0

    def on_click(self, event):
        x, y = event.x // self.size, event.y // self.size
        if not self.inside(x, y) or (x+y)%2 == 0:
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
                color = 'saddle brown' if (x+y)%2 else 'burlywood'
                if self.selected == (x, y):
                    color = 'yellow'
                elif self.selected:
                    for seq in self.valid_moves.get(self.selected, []):
                        if seq[-1] == (x, y):
                            color = 'light green'
                            break
                self.canvas.create_rectangle(
                    x*self.size, y*self.size,
                    (x+1)*self.size, (y+1)*self.size,
                    fill=color, outline='black'
                )
                p = self.board[y][x]
                if p != 0:
                    col = 'black' if p > 0 else 'red'
                    r = self.size//2 - 5
                    cx, cy = x*self.size + self.size//2, y*self.size + self.size//2
                    self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=col, outline='white')
                    if self.is_king(p):
                        self.canvas.create_text(cx, cy, text='K', fill='white', font=('Arial', 16, 'bold'))

    def play_random(self, n):
        for _ in range(n):
            moves = self.get_moves(self.current_player)
            if not moves:
                break
            src = random.choice(list(moves.keys()))
            seq = random.choice(moves[src])
            self.play(seq)
            self.current_player = self.opponent(self.current_player)

    def display_ui(self):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.cols*self.size, height=self.rows*self.size)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.on_click)
        self.update_display()
        self.root.mainloop()


if __name__ == "__main__":
    game = Checkers()
    game.play_random(120)
    game.display_ui()
