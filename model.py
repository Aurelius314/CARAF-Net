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
from Mambamodel import *
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from Mambamodel import *
    
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x 
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    
class Domainclassifica(nn.Module):
    def __init__(self, input_dim):
        super(Domainclassifica, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1), 
            nn.Dropout(0.2),   
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, alpha=1.0):
        reversed_features = ReverseLayerF.apply(x, alpha)
        domain_pred = self.layer(reversed_features)
        return domain_pred
    
def generate_intermediate_label(beta, y_s_true, y_t_pseudo_soft, num_classes):
    device = beta.device
    if y_s_true.dim() > 1:
        y_s_true = y_s_true.squeeze() 
    y_s_true = y_s_true.long()
    y_s_onehot = F.one_hot(y_s_true, num_classes=num_classes).float().to(device) 
    if beta.dim() == 1:
        beta = beta.unsqueeze(1) 
    beta = torch.clamp(beta, 0.1, 0.9)
    y_i = beta * y_s_onehot + (1 - beta) * y_t_pseudo_soft 
    y_i = y_i / (y_i.sum(dim=1, keepdim=True) + 1e-9) 

    return y_i

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
        self.last_attention = None
        
    def forward(self, x):  
        attn = torch.softmax(self.attention_weights(x), dim=1) 
        self.last_attention = attn 
        out = torch.sum(attn * x, dim=1) 
        return out
        

def cosine_similarity_loss(feat1, feat2):
    return F.cosine_similarity(feat1, feat2, dim=1)

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
            nn.Linear(num_channels * 2, num_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels, 1)
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

    
class CARAF_NetModel(nn.Module):
    def __init__(self, cuda, number_of_source=1, number_of_category=3, batch_size=10, time_steps=15):
        super(CARAF_NetModel, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_category = number_of_category
        self.number_of_source = number_of_source
        self.eeg_extractor = GNNConCrossFusionEncoder(
            input_dim=310,
            hidden_dim=256,
            output_dim=128,
            num_gnn_layers=2,
            num_mamba_layers=2,
            mamba_state_dim=16,
            num_channels=62,
            num_bands=5,
            dropout=0.1,
            gnn_type='gat'
        )
        self.domain_classifica = Domainclassifica(128)
        self.AdaptiveFeatureFusion = ChannelCrossAttentionFeatureFusion(
            input_dim=310,
            num_channels=62,
            num_bands=5,
            num_heads=2,
            dropout=0.1,
            temperature=1.0 
        )
        self.attention_pool = AttentionPooling(input_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
        
        self.temperature = 2 
        self.integrated_logits = None
        
        self.train_steps = 0
        self.center_start_epoch = 10
        self.initial_fusion_temp = 1.0
        self.initial_pseudo_temp = 0.5
        self.initial_center_temp = 0.8 
        self.current_phase = 0  
        self.use_distillation = False  
    
    def forward(self, x, corres, corres_label, m=1.0, current_epoch=0, total_epochs=100, warmup_epochs=20, 
            lambda_cls=1.0, lambda_dom=0.1, lambda_dom_I=0.1, lambda_pcls=0.5, 
            lambda_l=0.1, lambda_f=0.1, lambda_entropy=0.05,
            pseudo_label_threshold=0.7):
        
        progress = current_epoch / total_epochs
        warmup_ratio = min(1.0, current_epoch / warmup_epochs)
        adapt_ratio = 0.0 if current_epoch < warmup_epochs else (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        
        self.current_phase = 0 if current_epoch < warmup_epochs else 1
        self.use_distillation = current_epoch >= warmup_epochs
    

        
        fusion_temp = max(0.8, self.initial_fusion_temp - 0.2 * progress)
        self.AdaptiveFeatureFusion.temperature = fusion_temp
        source_features = self.eeg_extractor(corres)
        
        fused_input, beta = self.AdaptiveFeatureFusion(x, corres)
        beta = torch.clamp(beta, min=0.0, max=1.0)
        fused_features = self.eeg_extractor(fused_input)
        
        target_features = self.eeg_extractor(x)
        
        source_features1 = self.attention_pool(source_features)
        fused_features1 = self.attention_pool(fused_features)
        target_features1 = self.attention_pool(target_features)


        logits_s = self.classifier(source_features1)
        logits_i = self.classifier(fused_features1)
        logits_t = self.classifier(target_features1)
        
        
        p_dom_s = self.domain_classifica(source_features1, m)
        p_dom_i = self.domain_classifica(fused_features1, m)
        p_dom_t = self.domain_classifica(target_features1, m)


        target_source = torch.ones_like(p_dom_s)
        target_target = torch.zeros_like(p_dom_t)
        loss_s = F.binary_cross_entropy(p_dom_s, target_source, reduction='mean')
        loss_t = F.binary_cross_entropy(p_dom_t, target_target, reduction='mean')
        L_dom = loss_s + loss_t
        
        beta_clamped = torch.clamp(beta.detach(), min=0.0, max=1.0)
        domain_confusion_target = 0.5 * torch.ones_like(p_dom_i)
        if domain_confusion_target.dim() == 1 and p_dom_i.dim() == 2:
            domain_confusion_target = domain_confusion_target.unsqueeze(1)
        L_dom_I = F.binary_cross_entropy(p_dom_i, domain_confusion_target, reduction='mean')


        L_cls_S = F.cross_entropy(logits_s, corres_label.long().view(-1))
        
        L_cls_T_pseudo = torch.tensor(0.0, device=x.device)
        if self.current_phase == 1: 
            with torch.no_grad():
                probs_t = F.softmax(logits_t, dim=1)
                conf_t, pseudo_labels_t = torch.max(probs_t, dim=1)

                pseudo_label_threshold = 0.7
                mask = conf_t >= pseudo_label_threshold
                mask_ratio = mask.float().mean()
                
            if self.current_phase == 1 and mask.sum() > 0: 
                L_cls_T_pseudo = F.cross_entropy(logits_t[mask], pseudo_labels_t[mask])
                lambda_pcls_applied = lambda_pcls
            else:
                lambda_pcls_applied = 0.0
        else:
            lambda_pcls_applied = 0.0
        with torch.no_grad():
            target_probs_detached = F.softmax(logits_t.detach(), dim=1)

        y_i_soft = generate_intermediate_label(
            beta,  
            corres_label, 
            target_probs_detached, 
            self.number_of_category
        )

        L_cls_I = -torch.sum(y_i_soft * F.log_softmax(logits_i, dim=1), dim=1).mean()
        
        lambda_cls_i = 0.4 if self.current_phase == 0 else 0.5 
    
        L_cls_total = L_cls_S + lambda_pcls_applied * L_cls_T_pseudo + lambda_cls_i * L_cls_I

    
        beta_mean = beta_clamped.mean()
        beta_global = beta_clamped.mean().detach()  
        beta_global = torch.clamp(beta_global, min=0.1, max=0.9)

        L_f_mse = beta_global * F.mse_loss(fused_features, source_features) + \
                (1 - beta_global) * F.mse_loss(fused_features, target_features)
        
        d_source = cosine_similarity_loss(fused_features1, source_features1)
        d_target = cosine_similarity_loss(fused_features1, target_features1)
        L_I_S_f = beta_global * torch.mean(-F.logsigmoid(d_source))
        L_I_T_f = (1 - beta_global) * torch.mean(-F.logsigmoid(d_target))
        L_f_new = L_I_S_f + L_I_T_f
        
        L_distill = torch.tensor(0.0, device=x.device)
        
        if current_epoch < warmup_epochs:
            lambda_dom = 0.5 * (1.0 - 0.3 * progress) 
            lambda_dom_I = 0.2
            lambda_f = 0.2 * warmup_ratio 

        else:
            lambda_dom = 0.35 * (1.0 - 0.3 * adapt_ratio) 
            lambda_dom_I = 0.2 * (1.0 - 0.3 * adapt_ratio) 
            lambda_f = 0.3 + 0.2 * adapt_ratio  
            
            
        if self.use_distillation:
            distill_temp = self.temperature * (1.0 + adapt_ratio)
            soft_s = F.softmax(logits_s / distill_temp, dim=1)
            soft_i = F.softmax(logits_i / distill_temp, dim=1)
            loss_s2i_per_sample = F.kl_div(
                F.log_softmax(logits_i / distill_temp, dim=1),
                soft_s.detach(),
                reduction='none'
            ).sum(dim=1)
            loss_i2t_per_sample = F.kl_div(
                F.log_softmax(logits_t / distill_temp, dim=1),
                soft_i.detach(),
                reduction='none'
            ).sum(dim=1)
            confidence_s = torch.max(F.softmax(logits_s, dim=1), dim=1)[0]
            confidence_i = torch.max(F.softmax(logits_i, dim=1), dim=1)[0]
            
            weighted_s2i = beta_clamped.squeeze() * confidence_s * loss_s2i_per_sample
            weighted_i2t = (1 - beta_clamped.squeeze()) * confidence_i * loss_i2t_per_sample
            
            L_distill = (weighted_s2i.mean() + weighted_i2t.mean()) / 2

            lambda_distill = min(0.3, 0.1 + 0.2 * adapt_ratio)
        else:
            lambda_distill = 0.0

        
        L_total = lambda_cls * L_cls_total + \
                lambda_dom* L_dom + \
                lambda_dom_I * (L_dom_I) + \
                lambda_f * (L_f_mse+L_f_new) + \
                lambda_distill * L_distill


        
        return_dict = {
            "total_loss": L_total,
            "L_cls_S": L_cls_S,
            "L_cls_T_pseudo": L_cls_T_pseudo,
            "L_cls_I": L_cls_I,
            "L_dom": L_dom,
            "L_dom_I": L_dom_I,
            "L_f": L_f_new,
            "L_distill": L_distill,
            "lambda_pcls": torch.tensor(lambda_pcls_applied),
            "lambda_distill": torch.tensor(lambda_distill),
            "center_weight": torch.tensor(1.0),
            "beta_mean": beta.mean(),
            "beta_std": beta.std(),
            "fusion_temp": torch.tensor(fusion_temp),
            "phase": torch.tensor(float(self.current_phase)),
        }
        
        if self.current_phase == 1:
            with torch.no_grad():
                return_dict["pseudo_conf_mean"] = conf_t.mean()
                return_dict["pseudo_mask_ratio"] = mask_ratio if mask.sum() > 0 else torch.tensor(0.0)
        
        return return_dict
        
def testCARAF(model, test_loader, cuda, args):
    device = torch.device('cuda' if cuda else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            if cuda:
                data = data.to(device)
                labels = labels.to(device)

            features = model.eeg_extractor(data)
            
            if len(features.shape) == 3: 
                features = model.attention_pool(features)
 
            logits = model.classifier(features)

            if hasattr(model, 'use_distillation') and model.use_distillation:

                temp_scaled_logits = logits / 1.5
                temp_probs = F.softmax(temp_scaled_logits, dim=1)
                
 
                conf_values, _ = torch.max(F.softmax(logits, dim=1), dim=1)
                conf_weights = conf_values.unsqueeze(1).expand_as(logits)

                smoothed_probs = F.softmax(logits, dim=1) * 0.8 + 0.2 / args.cls_classes

                final_probs = conf_weights * temp_probs + (1 - conf_weights) * smoothed_probs
                
                _, preds = torch.max(final_probs, 1)
            else:
                _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(np.array(all_labels).reshape(-1), np.array(all_preds).reshape(-1))
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    class_report = classification_report(
        all_labels, 
        all_preds, 
        zero_division=0,
        output_dict=False
    )
    
    print(f"\n分类报告:\n{class_report}")
    
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    
    print("\n类别分布统计:")
    print("真实标签分布:", dict(zip(unique_labels, label_counts)))
    print("预测标签分布:", dict(zip(unique_preds, pred_counts)))
    
    missing_classes = set(range(args.cls_classes)) - set(unique_preds)
    if missing_classes:
        print(f"警告: 以下类别在预测中缺失: {missing_classes}")
    
    print(f"准确率: {accuracy*100:.2f}%, 精确率: {precision*100:.2f}%, 召回率: {recall*100:.2f}%, F1分数: {f1*100:.2f}%")
    
    if hasattr(model, 'current_phase') and model.current_phase > 0:
        phase_names = ["预热阶段", "适应阶段"]
        print(f"当前训练阶段: {phase_names[model.current_phase]}")
        if model.use_distillation:
            print("知识蒸馏已启用，使用增强型推理")
    
    return accuracy * 100