#!/usr/bin/env python3
"""Interactive terminal UI for playing Hex against various agents."""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

from hex_game import HexGame
from model import HexNet, get_device
from mcts import MCTS
from hex_llm import HexTransformer


# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------

class RandomAgent:
    """Picks a uniformly random legal move."""

    def __init__(self, game):
        self.game = game
        self.name = "Random"

    def predict(self, state, player, move_history=None):
        legal = self.game.get_legal_moves(state)
        legal_actions = np.where(legal == 1)[0]
        return int(np.random.choice(legal_actions))

    def reset(self):
        pass


class AlphaZeroAgent:
    """Uses a trained HexNet + MCTS to pick the strongest move."""

    def __init__(self, game, simulations=100):
        self.game = game
        self.simulations = simulations
        self.name = "AlphaZero"
        self.device = get_device()

        checkpoint_path = "checkpoints/model_latest.pt"
        if not os.path.exists(checkpoint_path):
            print(f"No trained model found at '{checkpoint_path}'. Run train.py first.")
            sys.exit(1)

        self.model = HexNet(size=game.size)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.mcts = MCTS(
            game=self.game,
            model=self.model,
            num_simulations=self.simulations,
            device=self.device,
        )

    def predict(self, state, player, move_history=None):
        policy = self.mcts.search(state, player)
        return int(np.argmax(policy))

    def reset(self):
        pass


class HexLLMAgent:
    """Uses a trained HexTransformer to predict the next move autoregressively."""

    def __init__(self, game):
        self.game = game
        self.name = "HexLLM"
        self.device = get_device()
        self.move_sequence = []

        checkpoint_path = "checkpoints/hex_llm.pt"
        if not os.path.exists(checkpoint_path):
            print(f"No trained model found at '{checkpoint_path}'. Run train.py first.")
            sys.exit(1)

        vocab_size = game.size * game.size + 1
        max_seq_len = game.size * game.size + 1
        self.model = HexTransformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, state, player, move_history=None):
        legal = self.game.get_legal_moves(state)
        action = self.model.predict_move(self.move_sequence, legal, temperature=0.5)
        self.move_sequence.append(action)
        return action

    def record_opponent_move(self, action):
        """Record a move made by someone else so the sequence stays in sync."""
        self.move_sequence.append(action)

    def reset(self):
        self.move_sequence = []


class HumanAgent:
    """Prompts the human for input via the terminal."""

    def __init__(self, game):
        self.game = game
        self.name = "Human"

    def predict(self, state, player, move_history=None):
        legal = self.game.get_legal_moves(state)
        while True:
            try:
                raw = input("  Your move (row,col): ").strip()
            except EOFError:
                print()
                raise KeyboardInterrupt
            parts = raw.replace(" ", "").split(",")
            if len(parts) != 2:
                print("  Invalid format. Enter row,col (e.g. 2,3)")
                continue
            try:
                row, col = int(parts[0]), int(parts[1])
            except ValueError:
                print("  Invalid numbers. Enter row,col (e.g. 2,3)")
                continue
            if not (0 <= row < self.game.size and 0 <= col < self.game.size):
                print(f"  Out of bounds. Row and col must be 0-{self.game.size - 1}.")
                continue
            action = self.game.coords_to_action(row, col)
            if legal[action] != 1:
                print("  That cell is already occupied. Try again.")
                continue
            return action

    def reset(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLAYER_LABELS = {1: "Player 1 (B, top-to-bottom)", -1: "Player 2 (W, left-to-right)"}
PLAYER_SHORT = {1: "B", -1: "W"}


def make_agent(name, game, simulations=100):
    """Factory that builds an agent by name string."""
    if name == "random":
        return RandomAgent(game)
    elif name == "alphazero":
        return AlphaZeroAgent(game, simulations=simulations)
    elif name == "hexllm":
        return HexLLMAgent(game)
    elif name == "human":
        return HumanAgent(game)
    else:
        raise ValueError(f"Unknown agent type: {name}")


def notify_llm_agents(agents, action, acting_index):
    """Keep HexLLM agents that did NOT act in sync with the move sequence."""
    for i, agent in enumerate(agents):
        if i != acting_index and isinstance(agent, HexLLMAgent):
            agent.record_opponent_move(action)


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def play_game(game, agents, players):
    """Play one full game and return the winner (1 or -1).

    agents:  list of two agent objects, agents[0] plays as players[0], etc.
    players: list [p1_val, p2_val] e.g. [1, -1] mapping agent index to player value.
    """
    state = game.get_initial_state()
    current_player = 1  # player 1 always starts in Hex

    print()
    game.render(state)
    print()

    while True:
        # Figure out which agent is acting
        if current_player == players[0]:
            agent_idx = 0
        else:
            agent_idx = 1
        agent = agents[agent_idx]

        label = PLAYER_SHORT[current_player]
        print(f"  Turn: {PLAYER_LABELS[current_player]}  [{agent.name}]")

        action = agent.predict(state, current_player)
        row, col = game.action_to_coords(action)

        if not isinstance(agent, HumanAgent):
            print(f"  {agent.name} plays: {row},{col}")
        else:
            print(f"  You play: {row},{col}")

        # Keep any HexLLM agents that didn't act in sync
        notify_llm_agents(agents, action, agent_idx)

        # If the acting agent is HexLLM, the move was already appended inside predict()
        state = game.make_move(state, action, current_player)
        print()
        game.render(state)
        print()

        winner = game.check_winner(state)
        if winner != 0:
            return winner

        # Check for full board (theoretical draw guard -- Hex cannot draw)
        if game.get_legal_moves(state).sum() == 0:
            return 0

        current_player = game.get_next_player(current_player)

        # Small delay in agent-vs-agent so the viewer can follow along
        is_agent_vs_agent = not any(isinstance(a, HumanAgent) for a in agents)
        if is_agent_vs_agent:
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Play Hex in the terminal against various agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python play.py                              # human vs alphazero (agent first)\n"
            "  python play.py --human-first                # human plays first\n"
            "  python play.py --agent random               # human vs random agent\n"
            "  python play.py --agent alphazero --agent2 hexllm   # watch two agents play\n"
            "  python play.py --agent human --agent2 random       # human first vs random\n"
        ),
    )
    parser.add_argument(
        "--agent",
        choices=["alphazero", "hexllm", "random", "human"],
        default="alphazero",
        help="First agent type (default: alphazero)",
    )
    parser.add_argument("--size", type=int, default=5, help="Board size (default: 5)")
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations for alphazero agent (default: 100)",
    )
    parser.add_argument(
        "--human-first",
        action="store_true",
        help="Human plays first (player 1, top-to-bottom). Ignored when --agent2 is set.",
    )
    parser.add_argument(
        "--agent2",
        choices=["alphazero", "hexllm", "random", "human"],
        default=None,
        help="If provided, run agent-vs-agent (or human-vs-agent) mode.",
    )
    args = parser.parse_args()

    game = HexGame(size=args.size)

    # ---- Determine the two participants and their player assignments ----
    if args.agent2 is not None:
        # Two-agent (or human-vs-agent) mode
        agent1 = make_agent(args.agent, game, args.simulations)
        agent2 = make_agent(args.agent2, game, args.simulations)
        agents = [agent1, agent2]
        # agent1 is player 1 (B, top-to-bottom), agent2 is player 2 (W, left-to-right)
        players = [1, -1]
    else:
        # Human vs agent mode
        ai_agent = make_agent(args.agent, game, args.simulations)
        human = HumanAgent(game)
        if args.human_first:
            agents = [human, ai_agent]
            players = [1, -1]
        else:
            agents = [ai_agent, human]
            players = [1, -1]

    # ---- Title ----
    print()
    print("=" * 40)
    print(f"  HEX  {args.size}x{args.size}")
    print(f"  {PLAYER_LABELS[1]}: {agents[0].name}")
    print(f"  {PLAYER_LABELS[-1]}: {agents[1].name}")
    print("=" * 40)

    # ---- Play loop ----
    try:
        while True:
            winner = play_game(game, agents, players)

            if winner == 0:
                print("  Game ended in a draw (should not happen in Hex!).")
            else:
                # Find which agent won
                if winner == players[0]:
                    winner_agent = agents[0]
                else:
                    winner_agent = agents[1]
                w_label = PLAYER_SHORT[winner]
                print(f"  {winner_agent.name} ({w_label}) wins!")

            print()
            again = input("  Play again? (y/n): ").strip().lower()
            if again != "y":
                print("  Thanks for playing!")
                break

            # Reset agent state for a new game
            for a in agents:
                a.reset()

    except KeyboardInterrupt:
        print("\n  Game interrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
