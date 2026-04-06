"""
Two-layer GraphSAGE backbone (shared).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

__all__ = ["GraphSAGEBackbone"]

class GraphSAGEBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = self.relu(x)
                x = self.dropout(x)
        return x

    def clip_weights(self, max_norm: float):
        """
        Clips the L2 norm of every weight matrix in every SAGEConv to max_norm.
        Call this after each optimizer.step() in the training loop.
        """
        with torch.no_grad():
            for conv in self.convs:
                for param in conv.parameters():
                    if param.dim() > 1:  # identify weight matrices (ignore 1D biases/BN params if any)
                        norm = param.norm(2)
                        if norm > max_norm:
                            param.data.mul_(max_norm / norm)
