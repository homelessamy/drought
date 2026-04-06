"""
Stage-2: GraphSAGE + MLP classifier for drought prediction.
"""
import torch
import torch.nn as nn
from .graphsage import GraphSAGEBackbone

__all__ = ["DroughtClassifier"]

class DroughtClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        V = len(config["data"]["variables"])
        C = config["data"]["C"]
        
        sage_hidden = config["model"]["sage_hidden"]
        dropout = config["model"]["dropout"]
        
        self.sage = GraphSAGEBackbone(
            in_channels=V,
            hidden_channels=sage_hidden,
            out_channels=sage_hidden,
            dropout=dropout
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(sage_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, C)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Input:  x shape (N, V)
        Output: logits shape (N, C)
        """
        feats = self.sage(x, edge_index)
        logits = self.mlp(feats)
        return logits
        
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probabilities (N, C).
        """
        logits = self.forward(x, edge_index)
        return torch.softmax(logits, dim=-1)
        
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
