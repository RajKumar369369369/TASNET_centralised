import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Actor', 'Critic', 'TransformerActor', 'MultiScaleActor']

class Actor(nn.Module):
    """SAC Actor: Outputs Bernoulli logits for frame selection"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='gru'):
        super(Actor, self).__init__()
        
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
            
        self.fc = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        """
        Returns logits to allow for stable log-softmax/entropy calculations in SAC.
        """
        h, _ = self.rnn(x)
        logits = self.fc(h)
        return logits

    def get_action_probs(self, x):
        """Helper for evaluation/testing mode"""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return torch.clamp(probs, 1e-8, 1.0 - 1e-8)


class Critic(nn.Module):
    """Twin Q-Network for SAC"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='gru'):
        super(Critic, self).__init__()
        
        # Backbone (Identical to Actor for feature processing)
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)

        # Twin Q-networks to handle overestimation bias
        self.q1_head = nn.Linear(hid_dim * 2, 2)
        self.q2_head = nn.Linear(hid_dim * 2, 2)

    def forward(self, x):
        """
        Returns two Q-values for the current state (all actions simultaneously)
        """
        h, _ = self.rnn(x)
        
        q1 = self.q1_head(h)
        q2 = self.q2_head(h)
        return q1, q2

class TransformerActor(nn.Module):
    """SAC Actor with Transformer Encoder"""
    def __init__(self, in_dim=1024, hid_dim=256, nhead=2, num_layers=1):
        super(TransformerActor, self).__init__()
        # Match the output dimension of the original bidirectional GRU (hid_dim * 2)
        self.input_proj = nn.Linear(in_dim, hid_dim * 2)
        
        # Learnable Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 2000, hid_dim * 2) * 0.02)
        
        # Smaller Transformer config with stabilization layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim * 2, 
            nhead=nhead, 
            dim_feedforward=hid_dim * 2, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        h = h + self.pos_encoder[:, :x.size(1), :]
        h = self.transformer(h)
        logits = self.fc(h)
        return logits

    def get_action_probs(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return torch.clamp(probs, 1e-8, 1.0 - 1e-8)

class MultiScaleActor(nn.Module):
    """SAC Actor with Multi-scale Temporal Convolutions"""
    def __init__(self, in_dim=1024, hid_dim=256):
        super(MultiScaleActor, self).__init__()
        # Output total channels = hid_dim * 2 to match original Actor
        out_channels = (hid_dim * 2) // 3
        rem = (hid_dim * 2) - (out_channels * 2)
        
        self.conv1 = nn.Conv1d(in_dim, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_dim, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_dim, rem, kernel_size=7, padding=3)
        
        self.fc = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        
        h1 = F.relu(self.conv1(x_t))
        h2 = F.relu(self.conv2(x_t))
        h3 = F.relu(self.conv3(x_t))
        
        h = torch.cat([h1, h2, h3], dim=1)
        h = h.transpose(1, 2)
        
        logits = self.fc(h)
        return logits

    def get_action_probs(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return torch.clamp(probs, 1e-8, 1.0 - 1e-8)
