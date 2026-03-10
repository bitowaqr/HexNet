import numpy as np
from collections import deque


class HexGame:
    """5x5 Hex game logic for AlphaZero training.

    Player 1 (value 1, displayed as 'B') connects TOP to BOTTOM.
    Player 2 (value -1, displayed as 'W') connects LEFT to RIGHT.
    Board is a numpy array of shape (size, size) with values 0, 1, -1.
    Actions are encoded as row * size + col.
    """

    NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    def __init__(self, size=5):
        self.size = size
        self.action_size = size * size

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.size, self.size), dtype=np.float32)

    def get_legal_moves(self, state: np.ndarray) -> np.ndarray:
        return (state.flatten() == 0).astype(np.float32)

    def make_move(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        row, col = self.action_to_coords(action)
        new_state = state.copy()
        new_state[row, col] = player
        return new_state

    def check_winner(self, state: np.ndarray) -> int:
        for player in [1, -1]:
            if self._has_connection(state, player):
                return player
        return 0

    def _has_connection(self, state: np.ndarray, player: int) -> bool:
        """BFS from one edge to check if player connects to the opposite edge."""
        visited = set()
        queue = deque()

        if player == 1:
            # Top to bottom
            for col in range(self.size):
                if state[0, col] == player:
                    queue.append((0, col))
                    visited.add((0, col))
            target_row = self.size - 1
            while queue:
                r, c = queue.popleft()
                if r == target_row:
                    return True
                for dr, dc in self.NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if (nr, nc) not in visited and state[nr, nc] == player:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
        else:
            # Left to right
            for row in range(self.size):
                if state[row, 0] == player:
                    queue.append((row, 0))
                    visited.add((row, 0))
            target_col = self.size - 1
            while queue:
                r, c = queue.popleft()
                if c == target_col:
                    return True
                for dr, dc in self.NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if (nr, nc) not in visited and state[nr, nc] == player:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
        return False

    def get_canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
        """Return board from player's perspective.
        If player is -1, transpose the board and swap piece values
        so the current player always 'connects top to bottom' with value 1."""
        if player == 1:
            return state.copy()
        # Transpose board and swap values: player -1 becomes 1, player 1 becomes -1
        return -state.T

    def canonical_action(self, action: int, player: int) -> int:
        """Convert action from canonical space back to real space.
        If player is -1, the board was transposed, so swap row/col."""
        if player == 1:
            return action
        r, c = self.action_to_coords(action)
        return self.coords_to_action(c, r)  # swap row and col

    def encode_state(self, state: np.ndarray, player: int) -> np.ndarray:
        """Encode board for neural net. Always from current player's perspective.
        The board is canonicalized so current player's stones are always 1
        and they always connect top to bottom."""
        canonical = self.get_canonical_state(state, player)
        encoded = np.zeros((3, self.size, self.size), dtype=np.float32)
        encoded[0] = (canonical == 1).astype(np.float32)   # current player's stones
        encoded[1] = (canonical == -1).astype(np.float32)  # opponent's stones
        encoded[2] = np.ones((self.size, self.size), dtype=np.float32)  # always 1 now
        return encoded

    def get_next_player(self, player: int) -> int:
        return -player

    def render(self, state: np.ndarray):
        symbols = {0: '.', 1: 'B', -1: 'W'}
        s = self.size

        # Column headers
        header = '   ' + ' '.join(str(c) for c in range(s))
        print(header)

        for row in range(s):
            indent = ' ' * row
            cells = ' '.join(symbols[int(state[row, col])] for col in range(s))
            print(f'{indent}{row:2d} \\ {cells} \\')

        # Bottom border
        print(' ' * (s + 3) + ' '.join(str(c) for c in range(s)))

    def action_to_coords(self, action: int) -> tuple:
        return (action // self.size, action % self.size)

    def coords_to_action(self, row: int, col: int) -> int:
        return row * self.size + col
