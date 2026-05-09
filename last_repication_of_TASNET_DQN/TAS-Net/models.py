import torch
import torch.nn as nn

__all__ = ['DSN']

class DSN(nn.Module):
    """Deep Summarization Network (Actor-Critic for PPO)"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        
        # Shared backbone: Processes GNN-extracted features
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        
        # Actor Head: Outputs probability of selecting a frame
        self.actor = nn.Linear(hid_dim * 2, 1)
        
        # Critic Head: Outputs scalar state value (V) for Advantage calculation
        self.critic = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        """
        Args:
            x: Input features (batch, seq_len, in_dim)
        Returns:
            p: Action probabilities (batch, seq_len, 1)
            v: State value estimates (batch, seq_len, 1)
        """
        # h shape: (batch, seq_len, hid_dim * 2)
        h, _ = self.rnn(x)
        
        # Actor: Using sigmoid to get Bernoulli probability p
        # Added a tiny epsilon to prevent log(0) issues during training
        p = torch.sigmoid(self.actor(h))
        p = torch.clamp(p, min=1e-8, max=1.0 - 1e-8)
        
        # Critic: Value estimation (Linear output, no activation)
        v = self.critic(h)
        
        return p, v