import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        # Causal mask: prevent attending to future tokens
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        # Reshape back to (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Simple two-layer feed-forward network with GELU activation."""

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: LN -> Attention -> residual -> LN -> FF -> residual."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class HexTransformer(nn.Module):
    """Small GPT-style transformer for next-move prediction in Hex.

    Vocabulary: size*size move tokens (0 to 24 for 5x5) + 1 start token (25)
    So vocab_size = size*size + 1 = 26 for 5x5.

    Input: sequence of move indices, starting with start token
    Output: logits over next move
    """

    def __init__(self, vocab_size=26, max_seq_len=26, d_model=64, n_heads=4, n_layers=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.start_token = vocab_size - 1  # 25 for 5x5

        # Token and positional embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x of shape (batch, seq_len) -- token indices
        Returns: logits of shape (batch, seq_len, vocab_size)
        """
        B, T = x.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        # Token + positional embeddings
        positions = torch.arange(T, device=x.device)
        tok_emb = self.token_emb(x)          # (B, T, d_model)
        pos_emb = self.pos_emb(positions)     # (T, d_model)
        h = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        # Final layer norm + linear head
        h = self.ln_f(h)
        logits = self.head(h)  # (B, T, vocab_size)
        return logits

    def predict_move(self, move_sequence: list, legal_moves: np.ndarray, temperature=1.0) -> int:
        """Given a list of moves so far and legal move mask, predict next move.
        Apply temperature and mask illegal moves before sampling.

        Args:
            move_sequence: list of move indices played so far (not including start token)
            legal_moves: binary numpy array of shape (action_size,) where 1=legal
            temperature: sampling temperature (higher = more random)

        Returns:
            action index (int) of the predicted next move
        """
        self.eval()
        device = next(self.parameters()).device

        # Build input sequence: [START, m1, m2, ...]
        tokens = [self.start_token] + list(move_sequence)
        x = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = self.forward(x)

        # Take logits at the last position
        last_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply temperature
        last_logits = last_logits / temperature

        # Convert to probabilities
        probs = F.softmax(last_logits, dim=-1)

        # Build a full-vocab mask: zero out the start token and any illegal moves
        mask = torch.zeros(self.vocab_size, device=device)
        legal_moves_t = torch.tensor(legal_moves, dtype=torch.float32, device=device)
        mask[: len(legal_moves)] = legal_moves_t

        # Zero out illegal moves
        probs = probs * mask

        # Renormalize
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            # Fallback: uniform over legal moves if all probabilities were zeroed
            probs = mask / mask.sum()

        # Sample from the distribution
        action = torch.multinomial(probs, num_samples=1).item()
        return action
