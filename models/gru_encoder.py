"""
Per-node GRU temporal encoder.
"""
import torch
import torch.nn as nn

__all__ = ["NodeGRUEncoder"]

class NodeGRUEncoder(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features, 
            hidden_size=hidden, 
            num_layers=num_layers, 
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x of shape (N, T, F)  — N nodes, T time steps, F features
        Output: h of shape (N, hidden) — last hidden state
        """
        output, hn = self.gru(x)
        # hn shape: (num_layers, N, hidden)
        return hn[-1]
