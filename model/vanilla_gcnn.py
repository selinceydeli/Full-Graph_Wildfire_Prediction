import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from torch_geometric.nn import GCNConv

class VanillaGCN(nn.Module):

    def __init__(self,
                 S_spatial: sp.spmatrix,
                 in_channels: int = 4,
                 hidden_channels: int = 24,
                 out_channels: int = 1,
                 num_layers: int = 10,
                 dropout: float = 0.1):
        super().__init__()

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout, inplace=True)

        # Prepare edge_index and edge_weight from the spatial adjacency matrix
        row, col = S_spatial.nonzero()
        self.edge_index = np.vstack((row, col))
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        self.edge_weight = S_spatial[row, col].A1
        self.edge_weight = torch.tensor(self.edge_weight, dtype=torch.float)

        # Build GCNN layers 
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else: # multiple layers
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B = x.size(0) # x.shape = [B, N, T]
    
        outs = []
        for b in range(B):
            h = x[b]
            for conv in self.convs[:-1]:
                h = conv(x=h, edge_index=self.edge_index, edge_weight=self.edge_weight)
                h = self.dropout(h)
                h = self.act(h)
            h = self.convs[-1](h, self.edge_index, self.edge_weight)
            outs.append(h.squeeze())

        return torch.stack(outs, dim=0) # [B, N]
