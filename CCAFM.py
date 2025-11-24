import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import time
from collections import defaultdict
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelCrossAttentionFeatureFusion(nn.Module):
    def __init__(self, input_dim=310, num_channels=62, num_bands=5, num_heads=4, dropout=0.1, temperature=1.0):
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.temperature = temperature

        self.q_proj = nn.Linear(num_channels, num_channels)
        self.k_proj = nn.Linear(num_channels, num_channels)
        self.v_proj = nn.Linear(num_channels, num_channels)
        self.out_proj = nn.Linear(num_channels, num_channels)

        self.gate_net = nn.Sequential(
            nn.Linear(num_channels * 2, num_channels), # 假设 residual 和 attn_output 都是 num_channels
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels, 1) # 输出单个门控值
        )

        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights = None

    def forward(self, target_data, source_data):
        batch_size, time_steps, features = target_data.shape
        assert features == self.input_dim, f"Expected input features to be {self.input_dim}, got {features}"
        channels_x_bands = self.num_channels * self.num_bands
        assert features == channels_x_bands, f"Input features ({features}) must equal channels*bands ({channels_x_bands})"

        target_reshaped = target_data.view(batch_size, time_steps, self.num_channels, self.num_bands)
        source_reshaped = source_data.view(batch_size, time_steps, self.num_channels, self.num_bands)
        fused_output = torch.zeros_like(target_reshaped)

        all_attn_weights_for_logging = [] 
        all_band_beta_gates = []

        for band_idx in range(self.num_bands):
            target_band = target_reshaped[:, :, :, band_idx]
            source_band = source_reshaped[:, :, :, band_idx] 

            target_flat = target_band.reshape(-1, self.num_channels) 
            source_flat = source_band.reshape(-1, self.num_channels) 

            target_norm = self.norm1(target_flat)
            source_norm = self.norm2(source_flat)

            residual = target_norm

            q = self.q_proj(target_norm)
            k = self.k_proj(source_norm)
            v = self.v_proj(source_norm)

            channels_per_head = self.num_channels // self.num_heads
            flat_batch_size = batch_size * time_steps

            q = q.view(flat_batch_size, self.num_heads, channels_per_head)
            k = k.view(flat_batch_size, self.num_heads, channels_per_head)
            v = v.view(flat_batch_size, self.num_heads, channels_per_head)
            scores = torch.bmm(q, k.transpose(1, 2)) / (channels_per_head ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights) 

            all_attn_weights_for_logging.append(attn_weights)

            context = torch.bmm(attn_weights, v)
            context = context.reshape(flat_batch_size, self.num_channels) 
            attn_output = self.out_proj(context) 

            gate_input = torch.cat([residual, attn_output], dim=-1) 
            beta_logits = self.gate_net(gate_input)  
            current_band_beta_gate = torch.sigmoid(beta_logits / self.temperature)


            all_band_beta_gates.append(current_band_beta_gate.view(batch_size, time_steps, 1))


            fused_band_flat = current_band_beta_gate * attn_output + (1 - current_band_beta_gate) * residual
            fused_band_reshaped = fused_band_flat.view(batch_size, time_steps, self.num_channels)
            fused_output[:, :, :, band_idx] = fused_band_reshaped


        if len(all_attn_weights_for_logging) > 0:
             self.last_attention_weights = torch.cat(all_attn_weights_for_logging, dim=1) 
        else:
            self.last_attention_weights = None


        fused_output = fused_output.reshape(batch_size, time_steps, features)

        if len(all_band_beta_gates) > 0:
            stacked_beta_gates = torch.stack(all_band_beta_gates, dim=3)
            beta_avg = stacked_beta_gates.mean(dim=[1, 3]).squeeze(dim=-1)
        else:
            beta_avg = torch.zeros(batch_size, 1, device=target_data.device) 

        return fused_output, beta_avg