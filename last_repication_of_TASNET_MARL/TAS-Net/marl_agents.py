# marl_agents.py
"""
Minimal Independent Multi-Agent PPO for TAS-Net.

Design principles
-----------------
* N independent agents, each responsible for a contiguous temporal chunk of
  the EEG sequence.
* Each agent has a tiny linear head (in_dim -> 1) that produces a per-frame
  importance scalar for its chunk.  The DSN hidden representation (GRU output
  before sigmoid) is used as state — no new feature extraction.
* Outputs are concatenated and passed through a sigmoid to produce a
  sig_probs tensor with the EXACT same shape as the baseline DSN output
  (1, seq_len, 1).  Downstream code (rewards, evaluation, ranking) is
  therefore completely unchanged.
* All agents share one optimizer with the rest of the network; no separate
  training loops, no replay buffers, no target networks.
* A tiny diversity regularization (scale 0.001) optionally encourages agents
  to differ — this is purely additive and does not change the PPO objective.

Usage
-----
    from marl_agents import MARLAgents
    marl = MARLAgents(hid_dim=256, num_agents=4, device=DEVICE)
    # forward returns same shape as DSN: (1, seq_len, 1)
    sig_probs = marl(dsn_hidden)   # dsn_hidden: (1, seq_len, hid_dim*2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentHead(nn.Module):
    """
    A single agent's decision head.
    Maps DSN hidden state (hid_dim*2) -> importance score per frame.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (chunk_len, hid_dim*2)  — hidden states for this agent's chunk
        Returns:
            scores: (chunk_len, 1)     — raw importance logits
        """
        return self.fc(h)


class MARLAgents(nn.Module):
    """
    Independent multi-agent wrapper for TAS-Net DSN.

    The DSN RNN is run as usual (full sequence) so temporal context is
    preserved.  Then each agent applies its own linear head to its temporal
    chunk, the logits are reassembled in order, and a single sigmoid converts
    them to the same sig_probs format as the baseline.

    Parameters
    ----------
    hid_dim   : int   — DSN hid_dim (same as --hid_dim arg, default 256)
    num_agents: int   — number of agents N (default 4)
    device    : torch.device
    """

    def __init__(self, hid_dim: int = 256, num_agents: int = 4,
                 device: torch.device = None):
        super().__init__()
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.num_agents = num_agents
        in_dim = hid_dim * 2          # bidirectional GRU output size

        # One lightweight head per agent
        self.heads = nn.ModuleList([AgentHead(in_dim) for _ in range(num_agents)])
        self.to(device)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _split_chunks(self, seq_len: int):
        """
        Divide [0, seq_len) into self.num_agents contiguous chunks.
        Returns a list of (start, end) index pairs.
        """
        chunk_size = max(1, seq_len // self.num_agents)
        chunks = []
        start = 0
        for i in range(self.num_agents):
            if i < self.num_agents - 1:
                end = start + chunk_size
            else:
                end = seq_len          # last agent takes any remainder
            chunks.append((start, end))
            start = end
        return chunks

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, dsn_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dsn_hidden : (1, seq_len, hid_dim*2)
                         — the raw GRU/LSTM output from DSN *before* sigmoid.
                           We intercept it in main.py by calling dsn.rnn()
                           separately.

        Returns:
            sig_probs  : (1, seq_len, 1)
                         — identical shape/semantics to baseline DSN output.
        """
        # dsn_hidden: (1, seq_len, H)
        h = dsn_hidden.squeeze(0)           # (seq_len, H)
        seq_len = h.size(0)
        chunks = self._split_chunks(seq_len)

        logit_parts = []
        for agent_idx, (start, end) in enumerate(chunks):
            h_chunk = h[start:end, :]       # (chunk_len, H)
            logit_chunk = self.heads[agent_idx](h_chunk)   # (chunk_len, 1)
            logit_parts.append(logit_chunk)

        logits = torch.cat(logit_parts, dim=0)   # (seq_len, 1)
        sig_probs = torch.sigmoid(logits)
        sig_probs = sig_probs.unsqueeze(0)        # (1, seq_len, 1)
        return sig_probs

    # ------------------------------------------------------------------
    # diversity regularization (optional, very small weight 0.001)
    # ------------------------------------------------------------------
    def diversity_loss(self, dsn_hidden: torch.Tensor) -> torch.Tensor:
        """
        Encourages agent heads to have different weight directions.
        This is purely additive and scaled to 0.001 in main.py — it cannot
        dominate the original PPO objective.

        Returns a scalar tensor.
        """
        # Stack agent head weights: (num_agents, hid_dim*2)
        weights = torch.stack([h.fc.weight.squeeze(0) for h in self.heads], dim=0)
        # Normalize
        weights = F.normalize(weights, p=2, dim=1)
        # Gram matrix of cosine similarities
        gram = torch.mm(weights, weights.t())    # (N, N)
        # We want off-diagonal entries to be small (agents differ)
        n = self.num_agents
        eye = torch.eye(n, device=self.device)
        off_diag = gram * (1.0 - eye)
        div_loss = off_diag.abs().sum() / max(1, n * (n - 1))
        return div_loss
