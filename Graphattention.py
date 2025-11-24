import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch

class GNNModule(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=256,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        device = x.device
        edge_index = edge_index.to(device)
        
        x = self.input_proj(x)
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            identity = x
            
            x = gnn_layer(x, edge_index)
            
            x = x + identity
            
            x = self.norms[i](x)
            x = F.gelu(x)
            x = self.dropout(x)
        
        x = self.output_proj(x)
        
        return x


class GNNEEGEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        num_channels=62,
        num_bands=5,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.num_bands = num_bands
        
        assert input_dim == num_channels * num_bands
        
        self.gnn_module = GNNModule(
            input_dim=num_bands,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.register_buffer('edge_index', self._create_edge_index(num_channels))
        
    def _create_edge_index(self, num_channels):
        edges = []
        for i in range(num_channels):
            edges.append((i, (i + 1) % num_channels))
            edges.append(((i + 1) % num_channels, i))
            
            edges.append((i, (i + 2) % num_channels))
            edges.append(((i + 2) % num_channels, i))
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index
    
    def forward(self, x):
        device = x.device
        B, T, D = x.shape
        
        x = x.view(B, T, self.num_channels, self.num_bands)
        
        outputs = []
        
        for t in range(T):
            x_t = x[:, t, :, :]
            
            x_t = x_t.reshape(B * self.num_channels, self.num_bands)
            
            batch_edge_index = self.edge_index.clone()
            batch_edges_list = [batch_edge_index]
            
            for b in range(1, B):
                offset = b * self.num_channels
                batch_edges = self.edge_index.clone()
                batch_edges = batch_edges + offset
                batch_edges_list.append(batch_edges)
            
            batch_edge_index = torch.cat(batch_edges_list, dim=1).to(device)
            
            output_t = self.gnn_module(x_t, batch_edge_index)
            
            output_t = output_t.view(B, self.num_channels, self.output_dim)
            
            output_t = output_t.mean(dim=1)
            
            outputs.append(output_t)
        
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
