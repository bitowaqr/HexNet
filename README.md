# HexNet

A demo that trains two small AI models — one based on AlphaZero, one based on the GPT architecture — and lets you play against them or watch them compete. The game is 5x5 Hex. 

## The Two Models

**HexZero** (AlphaZero architecture) — a small convolutional neural network combined with tree search. Learned from scratch by playing itself ~4,500 games. No human input. Before each move, it simulates hundreds of possible futures and picks the best path.

**HexGPT** (GPT architecture) — a small transformer trained on 3,000 game recordings produced by HexZero. Same idea as large language models: read sequences, predict what comes next. It picks moves based on pattern recognition. No search, just predicting the next token.

Both trained on a MacBook (M1) in about two hours.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open [localhost:8042](http://localhost:8042).

## Train From Scratch

```bash
python train.py --iterations 15 --episodes 300 --simulations 200 --workers 6
python train_llm.py
```

~80 minutes for HexZero, ~10 minutes for HexGPT. Trained checkpoints are in `checkpoints/`.