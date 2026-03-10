import os
import time
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.optim as optim

from hex_game import HexGame
from model import HexNet, get_device
from mcts import MCTS


def _self_play_worker(args):
    """Run self-play games in a worker process (CPU only)."""
    size, model_state_dict, num_games, num_simulations, temp_threshold = args

    game = HexGame(size=size)
    model = HexNet(size=size)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Workers always use CPU to avoid MPS/CUDA multiprocessing issues
    device = torch.device('cpu')
    model.to(device)

    mcts = MCTS(game, model, num_simulations=num_simulations, device=device)

    all_examples = []
    game_lengths = []

    for _ in range(num_games):
        state = game.get_initial_state()
        player = 1
        trajectory = []
        move_count = 0

        while True:
            policy = mcts.search(state, player)
            encoded = game.encode_state(state, player)
            trajectory.append((encoded, policy, player))

            if move_count < temp_threshold:
                policy_safe = policy + 1e-8
                policy_safe = policy_safe / policy_safe.sum()
                action = np.random.choice(len(policy_safe), p=policy_safe)
            else:
                noisy = policy + np.random.uniform(0, 1e-6, size=policy.shape)
                action = np.argmax(noisy)

            state = game.make_move(state, action, player)
            player = game.get_next_player(player)
            move_count += 1

            winner = game.check_winner(state)
            if winner != 0:
                for encoded_state, mcts_policy, move_player in trajectory:
                    outcome = 1.0 if move_player == winner else -1.0
                    all_examples.append((encoded_state, mcts_policy, outcome))
                game_lengths.append(len(trajectory))
                break

            if game.get_legal_moves(state).sum() == 0:
                for encoded_state, mcts_policy, move_player in trajectory:
                    all_examples.append((encoded_state, mcts_policy, 0.0))
                game_lengths.append(len(trajectory))
                break

    return all_examples, game_lengths


class AlphaZeroTrainer:
    def __init__(self, game, model, device='cpu', args=None):
        self.game = game
        self.model = model
        self.device = device

        default_args = {
            'num_iterations': 30,
            'num_episodes': 200,
            'num_simulations': 200,
            'batch_size': 64,
            'lr': 0.001,
            'temp_threshold': 10,
            'num_workers': 6,
        }
        if args is not None:
            default_args.update(args)
        self.args = default_args

    def self_play_parallel(self):
        """Run self-play games across multiple CPU worker processes."""
        num_workers = self.args['num_workers']
        total_games = self.args['num_episodes']
        games_per_worker = total_games // num_workers
        remainder = total_games % num_workers

        # Prepare worker args
        model_state_dict = self.model.state_dict()
        # Move state dict to CPU for workers
        cpu_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}

        worker_args = []
        for i in range(num_workers):
            n_games = games_per_worker + (1 if i < remainder else 0)
            if n_games > 0:
                worker_args.append((
                    self.game.size,
                    cpu_state_dict,
                    n_games,
                    self.args['num_simulations'],
                    self.args['temp_threshold'],
                ))

        all_examples = []
        all_lengths = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_self_play_worker, wa) for wa in worker_args]
            done = 0
            for future in as_completed(futures):
                examples, lengths = future.result()
                all_examples.extend(examples)
                all_lengths.extend(lengths)
                done += 1
                print(f"  Worker {done}/{len(worker_args)} finished ({len(lengths)} games)")

        return all_examples, all_lengths

    def train_on_examples(self, examples):
        """Train neural network on collected examples."""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'])

        random.shuffle(examples)

        batch_size = self.args['batch_size']
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]

            states = torch.tensor(
                np.array([e[0] for e in batch]),
                dtype=torch.float32
            ).to(self.device)
            target_policies = torch.tensor(
                np.array([e[1] for e in batch]),
                dtype=torch.float32
            ).to(self.device)
            target_values = torch.tensor(
                np.array([e[2] for e in batch]),
                dtype=torch.float32
            ).unsqueeze(1).to(self.device)

            log_policy_pred, value_pred = self.model(states)

            value_loss = torch.nn.functional.mse_loss(value_pred, target_values)
            policy_loss = -torch.sum(target_policies * log_policy_pred) / target_policies.size(0)
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def run(self):
        """Main training loop with parallel self-play."""
        os.makedirs('checkpoints', exist_ok=True)

        for iteration in range(self.args['num_iterations']):
            iter_start = time.time()

            # --- Self-play (parallel) ---
            print(f"  Iteration {iteration + 1} | Self-play ({self.args['num_episodes']} games across {self.args['num_workers']} workers)...")
            all_examples, game_lengths = self.self_play_parallel()

            # --- Train ---
            avg_loss = self.train_on_examples(all_examples)

            iter_time = time.time() - iter_start
            avg_game_len = np.mean(game_lengths) if game_lengths else 0

            print(
                f"Iteration {iteration + 1}/{self.args['num_iterations']} | "
                f"Examples: {len(all_examples)} | "
                f"Avg loss: {avg_loss:.4f} | "
                f"Avg game length: {avg_game_len:.1f} | "
                f"Time: {iter_time:.1f}s"
            )

            # --- Save checkpoint ---
            checkpoint = {
                'iteration': iteration + 1,
                'model_state_dict': self.model.state_dict(),
                'args': self.args,
                'size': self.game.size,
            }
            torch.save(checkpoint, f'checkpoints/model_iter_{iteration + 1}.pt')
            torch.save(checkpoint, 'checkpoints/model_latest.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlphaZero training for Hex')
    parser.add_argument('--size', type=int, default=5, help='Board size (default: 5)')
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--simulations', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    cli_args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    game = HexGame(size=cli_args.size)
    model = HexNet(size=cli_args.size).to(device)

    train_args = {}
    if cli_args.iterations is not None:
        train_args['num_iterations'] = cli_args.iterations
    if cli_args.episodes is not None:
        train_args['num_episodes'] = cli_args.episodes
    if cli_args.simulations is not None:
        train_args['num_simulations'] = cli_args.simulations
    if cli_args.workers is not None:
        train_args['num_workers'] = cli_args.workers

    trainer = AlphaZeroTrainer(game, model, device=device, args=train_args)

    print(f"Board size: {cli_args.size}x{cli_args.size}")
    print(f"Config: {trainer.args}")
    print()

    start_time = time.time()
    trainer.run()
    total_time = time.time() - start_time

    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"\nTraining complete. Total time: {minutes}m {seconds:.1f}s")
