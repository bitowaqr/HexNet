import numpy as np
import torch


class Node:
    """A single node in the MCTS tree."""

    __slots__ = ('state', 'player', 'parent', 'children', 'N', 'W', 'P')

    def __init__(self, state, player, parent=None, prior=0.0):
        self.state = state      # np.ndarray board state
        self.player = player    # player to move at this node (1 or -1)
        self.parent = parent
        self.children = {}      # action (int) -> Node
        self.N = 0              # visit count
        self.W = 0.0            # total value (from this node's perspective)
        self.P = prior          # prior probability from the network

    @property
    def Q(self):
        """Mean action-value."""
        if self.N == 0:
            return 0.0
        return self.W / self.N

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    """Monte Carlo Tree Search with neural network guidance (AlphaZero style)."""

    def __init__(self, game, model, num_simulations=100, c_puct=1.4, device='cpu'):
        """
        Args:
            game: HexGame instance providing game rules.
            model: HexNet instance (should be in eval mode) for policy/value prediction.
            num_simulations: number of MCTS simulations to run per search call.
            c_puct: exploration constant for the UCB formula.
            device: torch device for model inference.
        """
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def search(self, state, player):
        """Run MCTS from the given state and return an improved policy.

        Args:
            state: np.ndarray board of shape (size, size).
            player: int, the current player (1 or -1).

        Returns:
            np.ndarray of shape (action_size,) -- visit count distribution
            normalised to sum to 1.
        """
        root = Node(state, player)

        # Expand root so it always has children (unless terminal)
        winner = self.game.check_winner(state)
        if winner != 0:
            # Terminal root -- nothing to search
            policy = np.zeros(self.game.action_size, dtype=np.float32)
            return policy  # all zeros is fine; caller should check terminal

        self._expand(root)

        for _ in range(self.num_simulations):
            node = root

            # --- SELECT ---
            while not node.is_leaf():
                node = self._select_child(node)

            # --- EVALUATE ---
            winner = self.game.check_winner(node.state)
            if winner != 0:
                # Terminal node: value is +1 for the winner.
                # We need the value from the perspective of node.player.
                value = 1.0 if winner == node.player else -1.0
            else:
                # Check for draw (no legal moves). In Hex there are no draws
                # on a standard board, but handle it defensively.
                legal = self.game.get_legal_moves(node.state)
                if legal.sum() == 0:
                    value = 0.0
                else:
                    # Expand the leaf and get the network value estimate
                    value = self._expand(node)

            # --- BACKUP ---
            self._backup(node, value)

        # Build policy from root visit counts
        action_size = self.game.action_size
        visits = np.zeros(action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.N

        total = visits.sum()
        if total > 0:
            visits /= total
        return visits

    def _select_child(self, node):
        """Select the child with the highest UCB score.

        UCB = Q + c_puct * P * sqrt(parent_N) / (1 + child_N)

        Q is from the *parent's* perspective, so we negate the child's Q
        (the child stores value from its own perspective).
        """
        best_score = -float('inf')
        best_child = None
        sqrt_parent = np.sqrt(node.N)

        for child in node.children.values():
            # child.Q is from child's perspective; parent wants to maximise
            # its own value, which is the negative of the child's value.
            q = -child.Q
            ucb = q + self.c_puct * child.P * sqrt_parent / (1 + child.N)
            if ucb > best_score:
                best_score = ucb
                best_child = child

        return best_child

    def _expand(self, node):
        """Expand a leaf node using the neural network.

        Creates child nodes for every legal action with priors set from the
        network policy (masked and renormalised over legal moves).

        The board is canonicalized before feeding to the network so the model
        always sees itself as Player 1 (top-to-bottom). The policy output is
        in canonical space and must be mapped back to real actions.

        Returns:
            float: the network's value estimate for this node's state from the
                   perspective of node.player.
        """
        encoded = self.game.encode_state(node.state, node.player)
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_policy, value_tensor = self.model(tensor)

        # policy is log-probabilities in canonical space; convert to probabilities
        canonical_policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
        value = value_tensor.item()

        # Map canonical policy back to real action space
        policy = np.zeros(self.game.action_size, dtype=np.float32)
        for canonical_action in range(self.game.action_size):
            real_action = self.game.canonical_action(canonical_action, node.player)
            policy[real_action] = canonical_policy[canonical_action]

        # Mask illegal moves and renormalise
        legal = self.game.get_legal_moves(node.state)
        policy = policy * legal

        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        else:
            policy = legal / legal.sum()

        next_player = self.game.get_next_player(node.player)

        for action in range(self.game.action_size):
            if legal[action] == 0:
                continue
            child_state = self.game.make_move(node.state, action, node.player)
            child = Node(child_state, next_player, parent=node, prior=policy[action])
            node.children[action] = child

        return value

    def _backup(self, node, value):
        """Propagate the evaluation back up to the root.

        `value` is from the perspective of `node.player`.  As we walk up, we
        flip the sign at each level because the parent is the opposing player.
        """
        while node is not None:
            node.N += 1
            node.W += value
            value = -value          # flip for the parent (opponent's perspective)
            node = node.parent
