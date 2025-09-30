import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class EfficientNetBackbone(nn.Module):
    def __init__(self, in_channels):
        super(EfficientNetBackbone, self).__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights)

        if in_channels != 3:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(24),
                nn.SiLU(inplace=True)
            )
        else:
            self.stem = model.features[0]

        self.block1 = model.features[1]  
        self.block2 = model.features[2]
        self.block3 = model.features[3]
        self.block4 = model.features[4]
        self.block5 = model.features[5]

        del model

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.block1(x1)  
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x4 = self.block5(x4)

        return x1, x2, x3, x4
     
class SSAM(nn.Module):
    '''Spectral Spatial Attention Module (SSAM)'''
    def __init__(self, in_channels, groups=4, reduction_ratio=2, dropout=0.1):
        super(SSAM, self).__init__()
        assert in_channels % groups == 0, "in_channels harus bisa dibagi habis oleh groups"
        self.groups = groups
        self.group_channels = in_channels // groups

        # Spectral Group Attention
        self.spectral_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.group_channels, max(4, self.group_channels // reduction_ratio), 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(max(4, self.group_channels // reduction_ratio), self.group_channels, 1),
                nn.Sigmoid()
            ) for _ in range(groups)
        ])

        # Spatial Attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # Global Semantic Feedback
        self.semantic_feedback = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

        # Final Fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        identity = x

        # Spectral Group Attention
        chunks = torch.chunk(x, self.groups, dim=1)
        spec_out = []
        for i in range(self.groups):
            attn = self.spectral_convs[i](chunks[i])
            spec_out.append(chunks[i] * attn)
        spectral = torch.cat(spec_out, dim=1)

        # Spatial & Semantic Attention
        spatial = self.spatial_attn(x)
        semantic = self.semantic_feedback(x)

        # Fusion
        combined = spectral * (spatial * semantic)
        out = self.final_fusion(combined)

        return identity + out

class CAAM(nn.Module):
    '''Convolutional Axial Attention Module (CAAM)'''
    def __init__(self, dim, squeeze_factor=4, dropout=0.1):
        super(CAAM, self).__init__()
        self.squeeze_factor = squeeze_factor
        squeezed_dim = max(1, dim // squeeze_factor)
        
        # Squeeze projections
        self.squeeze_proj = nn.Sequential(
            nn.Conv2d(dim, squeezed_dim, 1, bias=False),
            nn.BatchNorm2d(squeezed_dim)
        )
        self.unsqueeze_proj = nn.Sequential(
            nn.Conv2d(squeezed_dim, dim, 1, bias=False), # Output dari attn (setelah h_conv/w_conv) adalah squeezed_dim
            nn.BatchNorm2d(dim)
        )
        
        # Axial attention components - Average Pooling
        self.h_avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.w_avg_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Axial attention components - Max Pooling
        self.h_max_pool = nn.AdaptiveMaxPool2d((None, 1))
        self.w_max_pool = nn.AdaptiveMaxPool2d((1, None))
        
        self.h_conv = nn.Conv2d(2 * squeezed_dim, squeezed_dim, (1, 3), padding=(0, 1), padding_mode='replicate', bias=False) # groups default adalah 1
        self.w_conv = nn.Conv2d(2 * squeezed_dim, squeezed_dim, (3, 1), padding=(1, 0), padding_mode='replicate', bias=False) # groups default adalah 1

        # Context-aware global descriptor
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        H, W = x.shape[2:]
        identity = x
        
        # Squeeze channel dimension
        x_squeezed = self.squeeze_proj(x) # [B, squeezed_dim, H, W]
        
        # Height-wise attention
        h_avg_features = self.h_avg_pool(x_squeezed) # [B, squeezed_dim, H, 1]
        h_max_features = self.h_max_pool(x_squeezed) # [B, squeezed_dim, H, 1]
        # Concat at channel dim (dim=1)
        h_pooled_features = torch.cat((h_avg_features, h_max_features), dim=1) # [B, 2 * squeezed_dim, H, 1]
        
        h_attn = self.h_conv(h_pooled_features) # Input: 2*squeezed_dim, Output: squeezed_dim. [B, squeezed_dim, H, 1]
        h_attn = h_attn.expand(-1, -1, H, W)    # [B, squeezed_dim, H, W]
        
        # Width-wise attention
        w_avg_features = self.w_avg_pool(x_squeezed) # [B, squeezed_dim, 1, W]
        w_max_features = self.w_max_pool(x_squeezed) # [B, squeezed_dim, 1, W]
        # Concat at channel dim
        w_pooled_features = torch.cat((w_avg_features, w_max_features), dim=1) # [B, 2 * squeezed_dim, 1, W]

        w_attn = self.w_conv(w_pooled_features) # Input: 2*squeezed_dim, Output: squeezed_dim. [B, squeezed_dim, 1, W]
        w_attn = w_attn.expand(-1, -1, H, W)    # [B, squeezed_dim, H, W]
        
        # Combine axial attentions
        attn = h_attn + w_attn
        attn = self.dropout(attn)
        
        # Restore channel dimension
        out = self.unsqueeze_proj(attn)

        # Context-aware modulation
        context = self.global_context(identity)
        out = out * context

        return identity + out

class FFM(nn.Module):
    '''Feature Fusion Module (FFM)'''
    def __init__(self, channels: List[int], embedding_dim, dropout: float = 0.1):
        super(FFM, self).__init__()

        total_channels = sum(channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, embedding_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, features: List[torch.Tensor], target_size: tuple):
        resized_feats = [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) if f.shape[2:] != target_size else f
            for f in features
        ]

        output = self.fusion(torch.cat(resized_feats, dim=1))

        return output
    
class SpatialPath(nn.Module):
    def __init__(self, spatial_channels, dropout):
        super(SpatialPath, self).__init__()

        self.ssam = SSAM(spatial_channels, dropout=dropout)

    def forward(self, x_spatial):
        return self.ssam(x_spatial)
    
class ContextPath(nn.Module):
    def __init__(self, context_channels_1, context_channels_2, dropout):
        super(ContextPath, self).__init__()

        self.caam_1 = CAAM(context_channels_1, squeeze_factor=4, dropout=dropout)
        self.caam_2 = CAAM(context_channels_2, squeeze_factor=4, dropout=dropout)

    def forward(self, x_context_1, x_context_2):
        x_context_1 = self.caam_1(x_context_1)
        x_context_2 = self.caam_2(x_context_2)
        return x_context_1, x_context_2
    
class EAM(nn.Module):
    '''Edge Attention Module (EAM)'''
    def __init__(self, f_in_channels=16, m_coarse_channels=160, attention_kernel_size=7):
        super(EAM, self).__init__()

        # 1. Initial projection
        self.initial_block = nn.Sequential(
            nn.Conv2d(f_in_channels, m_coarse_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(m_coarse_channels),
            nn.LeakyReLU(inplace=True)
        )

        # Combined Separable Conv block
        self.separable_conv = nn.Sequential(
            nn.Conv2d(m_coarse_channels, m_coarse_channels, kernel_size=3, stride=1, padding=1,
                      groups=m_coarse_channels, bias=False),
            nn.BatchNorm2d(m_coarse_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(m_coarse_channels, m_coarse_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(m_coarse_channels),
            nn.LeakyReLU(inplace=True)
        )

        # Spatial Attention block
        assert attention_kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if attention_kernel_size == 7 else 1
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=attention_kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

        # Edge attention block
        self.edge_attention = nn.Sequential(
            nn.Conv2d(m_coarse_channels, m_coarse_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, f_in, m_coarse):
        x = self.initial_block(f_in)
        x = self.separable_conv(x)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        internal_attention_map = self.spatial_attention(attention_map)

        # Cast attention map to float16 in inference mode only
        if not self.training:
            internal_attention_map = internal_attention_map.half()
        x = x * internal_attention_map

        # Edge attention map
        edge_attention_map = self.edge_attention(x)

        # Upsample m_coarse with scale_factor
        m_coarse_upsampled = F.interpolate(m_coarse, scale_factor=2, mode='bilinear', align_corners=False)

        # Refined output
        m_refined = m_coarse_upsampled * (1 + edge_attention_map)
        return m_refined
    
class SegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegHead, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.classifier(x)
       
class TerraSegNet(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        num_classes: int,
        embedding_dim: int = 160,
        dropout: float = 0.1
    ):
        super().__init__()
               
        self.backbone = EfficientNetBackbone(in_channels)
        backbone_ch = [24, 48, 64, 160]

        self.spatial_path = SpatialPath(backbone_ch[1], dropout=dropout)
        self.context_path = ContextPath(backbone_ch[2], backbone_ch[3], dropout=dropout)
        self.ffm = FFM(backbone_ch[1:], embedding_dim, dropout=dropout)
        self.eam = EAM(backbone_ch[0], embedding_dim)
        self.seg_head = SegHead(embedding_dim, num_classes)
        
    def forward(self, x):       
        # Multi-scale feature extraction
        x1, x2, x3, x4 = self.backbone(x)

        x_spatial = self.spatial_path(x2)
        x_context_1, x_context_2 = self.context_path(x3, x4)
        
        # Prepare all features for fusion
        all_features = [x_spatial, x_context_1, x_context_2]
        target_size = (x_spatial.shape[2:])  # Match fast_shared resolution
        fused_features = self.ffm(all_features, target_size) # 272, 64, 64            
        enhanced_features = self.eam(x1, fused_features) # 2, 64, 64
        seg_head = self.seg_head(enhanced_features)
        out = F.interpolate(seg_head, scale_factor=2, mode='bilinear', align_corners=False)
               

        return out

