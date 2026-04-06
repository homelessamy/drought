"""
Stage-1: GRU + GraphSAGE + linear head for weather prediction.
"""
import torch
import torch.nn as nn
from .gru_encoder import NodeGRUEncoder
from .graphsage import GraphSAGEBackbone

__all__ = ["WeatherForecastModel"]

class WeatherForecastModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        V = len(config["data"]["variables"])
        gru_hidden = config["model"]["gru_hidden"]
        sage_hidden = config["model"]["sage_hidden"]
        sage_layers = config["model"]["sage_layers"]
        dropout = config["model"]["dropout"]
        
        self.gru = NodeGRUEncoder(
            in_features=V,
            hidden=gru_hidden
        )
        
        self.sage = GraphSAGEBackbone(
            in_channels=gru_hidden,
            hidden_channels=sage_hidden,
            out_channels=sage_hidden,
            num_layers=sage_layers,
            dropout=dropout
        )
        
        self.head = nn.Linear(sage_hidden, V)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Input:  x_seq shape (N, T, V)
        Output: x_hat shape (N, V)   — predicted next weather state
        """
        h = self.gru(x_seq)
        feats = self.sage(h, edge_index)
        return self.head(feats)
        
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
