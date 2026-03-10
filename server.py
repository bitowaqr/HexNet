# pip install fastapi uvicorn

import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hex_game import HexGame
from hex_llm import HexTransformer
from mcts import MCTS
from model import HexNet, get_device

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Hex AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state – loaded once at import / startup
# ---------------------------------------------------------------------------
device = get_device()
game = HexGame(size=5)

# AlphaZero model
alphazero_model = HexNet(size=5)
az_checkpoint = torch.load("checkpoints/model_latest.pt", map_location=device)
alphazero_model.load_state_dict(az_checkpoint["model_state_dict"])
alphazero_model.to(device)
alphazero_model.eval()

# HexLLM model
hexllm_model = HexTransformer()
hexllm_state = torch.load("checkpoints/hex_llm.pt", map_location=device)
hexllm_model.load_state_dict(hexllm_state)
hexllm_model.to(device)
hexllm_model.eval()

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class MoveRequest(BaseModel):
    board: List[List[int]]
    player: int
    agent: str
    move_history: List[int] = []
    simulations: int = 100


class MoveResponse(BaseModel):
    action: int
    row: int
    col: int
    heatmap: List[float]
    winner: int


class SimulateRequest(BaseModel):
    agent1: str
    agent2: str
    num_games: int = 100
    simulations: int = 50


class GameRecord(BaseModel):
    moves: List[int]
    winner: int


class SimulateResponse(BaseModel):
    agent1_wins: int
    agent2_wins: int
    draws: int
    avg_game_length: float
    games: List[GameRecord]


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def _get_alphazero_move(state: np.ndarray, player: int, simulations: int):
    """Run MCTS and return (action, heatmap).
    The heatmap is the raw NN policy (gut feeling before search) which is
    more spread out and visually interesting than the MCTS visit counts."""
    mcts = MCTS(game, alphazero_model, num_simulations=simulations, device=device)
    policy = mcts.search(state, player)           # normalised visit counts
    action = int(np.argmax(policy))

    # Get raw NN policy for the heatmap (before search concentrates it)
    # log_policy is log_softmax output; recover logits then apply temperature
    encoded = game.encode_state(state, player)
    tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        log_policy, _ = alphazero_model(tensor)
    logits = log_policy.squeeze(0)  # log_softmax ≈ logits (up to constant)
    softened = F.softmax(logits / 3.0, dim=-1).cpu().numpy()

    # Map from canonical to real action space and mask illegal moves
    legal = game.get_legal_moves(state)
    heatmap = np.zeros(game.action_size, dtype=np.float32)
    for ca in range(game.action_size):
        ra = game.canonical_action(ca, player)
        heatmap[ra] = softened[ca]
    heatmap = heatmap * legal
    hm_sum = heatmap.sum()
    if hm_sum > 0:
        heatmap = heatmap / hm_sum

    return action, heatmap.tolist()


def _get_hexllm_heatmap(move_history: list, legal_moves: np.ndarray, temperature: float = 0.5):
    """Do a forward pass through HexLLM and return the masked probability
    distribution over board cells (length 25)."""
    hexllm_model.eval()
    tokens = [hexllm_model.start_token] + list(move_history)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = hexllm_model(x)

    last_logits = logits[0, -1, :] / temperature
    probs = F.softmax(last_logits, dim=-1)

    # Mask: zero out start token and illegal moves
    mask = torch.zeros(hexllm_model.vocab_size, device=device)
    legal_t = torch.tensor(legal_moves, dtype=torch.float32, device=device)
    mask[: len(legal_moves)] = legal_t

    probs = probs * mask
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = mask / mask.sum()

    # Return only the board-cell probabilities (first 25 entries)
    return probs[: game.action_size].cpu().numpy()


def _get_hexllm_move(state: np.ndarray, player: int, move_history: list):
    """Return (action, heatmap) for HexLLM."""
    legal_moves = game.get_legal_moves(state)
    action = hexllm_model.predict_move(move_history, legal_moves, temperature=0.5)
    # Use higher temperature for heatmap so the distribution is more spread out
    heatmap = _get_hexllm_heatmap(move_history, legal_moves, temperature=1.5)
    return action, heatmap.tolist()


def _get_random_move(state: np.ndarray):
    """Return (action, heatmap) for the random agent."""
    legal_moves = game.get_legal_moves(state)
    num_legal = int(legal_moves.sum())
    # Uniform distribution over legal moves
    heatmap = (legal_moves / num_legal).tolist() if num_legal > 0 else [0.0] * game.action_size
    legal_indices = np.where(legal_moves == 1)[0]
    action = int(random.choice(legal_indices))
    return action, heatmap


def _pick_move(agent: str, state: np.ndarray, player: int,
               move_history: list, simulations: int):
    """Dispatch to the correct agent and return (action, heatmap)."""
    if agent == "alphazero":
        return _get_alphazero_move(state, player, simulations)
    elif agent == "hexllm":
        return _get_hexllm_move(state, player, move_history)
    elif agent == "random":
        return _get_random_move(state)
    else:
        raise ValueError(f"Unknown agent: {agent}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/move", response_model=MoveResponse)
def post_move(req: MoveRequest):
    state = np.array(req.board, dtype=np.float32)
    action, heatmap = _pick_move(req.agent, state, req.player,
                                 req.move_history, req.simulations)

    row, col = game.action_to_coords(action)
    new_state = game.make_move(state, action, req.player)
    winner = int(game.check_winner(new_state))

    return MoveResponse(
        action=action,
        row=row,
        col=col,
        heatmap=heatmap,
        winner=winner,
    )


@app.post("/api/simulate", response_model=SimulateResponse)
def post_simulate(req: SimulateRequest):
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    total_length = 0
    recorded_games: list[GameRecord] = []

    for game_idx in range(req.num_games):
        state = game.get_initial_state()
        player = 1  # player 1 always starts
        moves: list[int] = []
        winner = 0

        while True:
            agent = req.agent1 if player == 1 else req.agent2
            action, _ = _pick_move(agent, state, player, moves, req.simulations)
            state = game.make_move(state, action, player)
            moves.append(action)

            winner = game.check_winner(state)
            if winner != 0:
                break

            # Hex on a full board always has a winner, but guard anyway
            if game.get_legal_moves(state).sum() == 0:
                break

            player = game.get_next_player(player)

        total_length += len(moves)
        if winner == 1:
            agent1_wins += 1
        elif winner == -1:
            agent2_wins += 1
        else:
            draws += 1

        # Keep the last 5 games for replay
        if game_idx >= req.num_games - 5:
            recorded_games.append(GameRecord(moves=moves, winner=winner))

    avg_length = total_length / max(req.num_games, 1)

    return SimulateResponse(
        agent1_wins=agent1_wins,
        agent2_wins=agent2_wins,
        draws=draws,
        avg_game_length=round(avg_length, 2),
        games=recorded_games,
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Static files (must be mounted after all API routes)
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8042"))
    uvicorn.run(app, host=host, port=port)
