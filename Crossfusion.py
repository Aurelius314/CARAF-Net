import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from Convolutionalgated import *
from Graphattention import *

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.spatial_to_temporal = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.temporal_to_spatial = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        
    def forward(self, spatial_features, temporal_features):
        device = spatial_features.device
        temporal_features = temporal_features.to(device)
        
        norm_spatial = self.norm1(spatial_features)
        norm_temporal = self.norm2(temporal_features)
        
        st_attn_output, _ = self.spatial_to_temporal(
            query=norm_spatial,
            key=norm_temporal,
            value=norm_temporal
        )
        spatial_enhanced = spatial_features + st_attn_output
        spatial_enhanced = self.norm3(spatial_enhanced)
        
        ts_attn_output, _ = self.temporal_to_spatial(
            query=norm_temporal,
            key=norm_spatial,
            value=norm_spatial
        )
        temporal_enhanced = temporal_features + ts_attn_output
        temporal_enhanced = self.norm4(temporal_enhanced)
        
        combined_features = torch.cat([
            spatial_features,
            temporal_features,
            spatial_enhanced,
            temporal_enhanced
        ], dim=-1)
        
        fused_features = self.fusion_mlp(combined_features)
        
        return fused_features


class CrossFusionEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=128,
        num_gnn_layers=2,
        num_mamba_layers=2,
        mamba_state_dim=16,
        num_channels=62,
        num_bands=5,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.spatial_encoder = GNNEEGEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            num_channels=num_channels,
            num_bands=num_bands
        )
        
        self.temporal_encoder = ConvEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_mamba_layers,
            state_dim=mamba_state_dim,
            dropout=dropout
        )
        
        self.fusion_module = CrossAttentionFusion(
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        spatial_features = self.spatial_encoder(x)
        
        temporal_features = self.temporal_encoder(x)
        
        fused_features = self.fusion_module(spatial_features, temporal_features)
        
        output = self.output_proj(fused_features) 
        
        return output


def to_device(model, device):
    model = model.to(device)
    for buffer_name, buffer in model.named_buffers():
        buffer.data = buffer.data.to(device)
    return model