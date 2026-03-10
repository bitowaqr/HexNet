import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ProcessPoolExecutor, as_completed

from hex_game import HexGame
from hex_llm import HexTransformer
from model import HexNet, get_device
from mcts import MCTS


def _generate_games_worker(args):
    """Worker: generate games using AlphaZero on CPU."""
    size, model_state_dict, num_games, num_simulations = args

    game = HexGame(size=size)
    model = HexNet(size=size)
    model.load_state_dict(model_state_dict)
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    mcts = MCTS(game, model, num_simulations=num_simulations, device=device)

    games = []
    for _ in range(num_games):
        state = game.get_initial_state()
        player = 1
        moves = []

        while True:
            policy = mcts.search(state, player)
            temp_policy = policy ** 2.0  # temperature=0.5 -> exponent=1/0.5=2
            temp_policy = temp_policy / temp_policy.sum()
            action = np.random.choice(len(temp_policy), p=temp_policy)
            moves.append(action)
            state = game.make_move(state, action, player)
            winner = game.check_winner(state)
            if winner != 0:
                break
            player = game.get_next_player(player)

        games.append(moves)

    return games


def generate_games_parallel(size, model_state_dict, num_games=3000, num_simulations=100, num_workers=6):
    """Generate games from trained AlphaZero in parallel."""
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    worker_args = []
    for i in range(num_workers):
        n = games_per_worker + (1 if i < remainder else 0)
        if n > 0:
            worker_args.append((size, model_state_dict, n, num_simulations))

    all_games = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_generate_games_worker, wa) for wa in worker_args]
        done = 0
        for future in as_completed(futures):
            games = future.result()
            all_games.extend(games)
            done += 1
            print(f"  Worker {done}/{len(worker_args)} finished ({len(games)} games)")

    return all_games


def train_hex_llm(transformer, games, epochs=50, batch_size=64, lr=0.001, device='cpu'):
    """Train the HexTransformer on game sequences."""
    START_TOKEN = transformer.start_token
    PAD_TOKEN = -1

    inputs = []
    targets = []
    for game_moves in games:
        inp = [START_TOKEN] + list(game_moves[:-1])
        tgt = list(game_moves)
        inputs.append(inp)
        targets.append(tgt)

    max_len = max(len(seq) for seq in inputs)

    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        padded_inputs.append(inp + [0] * pad_len)
        padded_targets.append(tgt + [PAD_TOKEN] * pad_len)

    input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=device)
    target_tensor = torch.tensor(padded_targets, dtype=torch.long, device=device)

    transformer = transformer.to(device)
    transformer.train()
    optimizer = optim.Adam(transformer.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    num_samples = len(input_tensor)

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        input_tensor = input_tensor[perm]
        target_tensor = target_tensor[perm]

        total_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_input = input_tensor[i:i + batch_size]
            batch_target = target_tensor[i:i + batch_size]

            logits = transformer(batch_input)
            logits_flat = logits.view(-1, transformer.vocab_size)
            target_flat = batch_target.view(-1)

            loss = criterion(logits_flat, target_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return transformer


if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    game = HexGame(size=5)
    num_workers = 6
    num_games = 3000

    # Load trained AlphaZero model
    model = HexNet(size=5)
    checkpoint_path = 'checkpoints/model_latest.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded AlphaZero model from {checkpoint_path}")
    else:
        print(f"Warning: {checkpoint_path} not found, using untrained model")

    # CPU state dict for workers
    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # Generate games in parallel
    print(f"Generating {num_games} games across {num_workers} workers...")
    games = generate_games_parallel(
        size=5,
        model_state_dict=cpu_state_dict,
        num_games=num_games,
        num_simulations=100,
        num_workers=num_workers,
    )
    print(f"Generated {len(games)} games total")

    # Train HexTransformer
    vocab_size = game.action_size + 1
    max_seq_len = game.action_size + 1
    transformer = HexTransformer(vocab_size=vocab_size, max_seq_len=max_seq_len)

    print("Training HexTransformer...")
    transformer = train_hex_llm(transformer, games, epochs=50, batch_size=64, lr=0.001, device=device)

    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/hex_llm.pt'
    torch.save(transformer.state_dict(), save_path)
    print(f"Saved HexTransformer to {save_path}")
