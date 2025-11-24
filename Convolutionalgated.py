import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
class SelectiveScanModule(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 3)

        self.norm = nn.LayerNorm(d_model)
        kernel_size = 4
        self.conv_padding = (kernel_size - 1) // 2  
        
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=self.conv_padding, 
            groups=d_model
        )

        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))

        self.state_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=2,
            padding=0,
            groups=1 
        )
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = x
        x = self.norm(x)

        x_proj = self.in_proj(x)
        x_z, x_delta, x_input = x_proj.chunk(3, dim=-1)
        
        x_delta = x_delta.transpose(1, 2)
        x_delta = self.conv(x_delta)
        x_delta = x_delta.transpose(1, 2)
        
        delta = F.softplus(x_delta) 
        z = torch.sigmoid(x_z) 
        B, L, D = x_input.shape

        x_input_conv = x_input.transpose(1, 2) 

        if L > 1: 
            x_state = self.state_conv(F.pad(x_input_conv, (1, 0)))
        else:
            x_state = x_input_conv

        x_state = x_state.transpose(1, 2)

        y = x_input * self.D.unsqueeze(0).unsqueeze(0) + x_state
        output = y * z
        output = self.dropout(self.out_proj(output))
        output = output + shortcut
        
        return output

class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=256,
        num_layers=2,
        state_dim=16,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.mamba_layers = nn.ModuleList([
            SelectiveScanModule(
                d_model=hidden_dim,
                d_state=state_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        
        for layer in self.mamba_layers:
            x = layer(x)
        
        output = self.output_proj(x)
        
        return output