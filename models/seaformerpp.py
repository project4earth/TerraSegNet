import math
import torch
from torch import nn
import torch.nn.functional as F

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        
        # Simplified batch norm instead of build_norm_layer
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super(ConvModule, self).__init__()
        
        # Calculate padding if not provided
        if padding == 0 and kernel_size > 1:
            padding = (kernel_size - 1) // 2
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding, dilation, groups, bias=bias)
        
        # Add batch normalization if specified
        if norm_cfg is not None:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
            
        # Add activation if specified
        if act_cfg is not None:
            if act_cfg.get('type') == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            # Add other activations if needed, e.g., ReLU6
            elif act_cfg.get('type') == 'ReLU6':
                self.act = nn.ReLU6(inplace=True)
            else:
                self.act = nn.ReLU(inplace=True)  # Default to ReLU
        else:
            self.act = None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.1, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        ks: int,
        stride: int,
        expand_ratio: int,
        activations = None, # Note: 'activations' should be a class, not an instance
        norm_cfg=dict(type='BN', requires_grad=True)
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU # Default activation class

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations()) # Instantiate activation
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks//2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(), # Instantiate activation
            # pw-linear
            Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class StackedMV2Block(nn.Module):
    def __init__(
            self,
            cfgs,
            stem,
            inp_channel=16,
            in_channels=3, # Renamed from 'in_channels' in original to avoid confusion with ConvModule
            activation=nn.ReLU, # Should be a class
            norm_cfg=dict(type='BN', requires_grad=True),
            width_mult=1.):
        super().__init__()
        self.stem = stem
        current_input_channels = in_channels # Used for the stem block's input
        if stem:
            self.stem_block = nn.Sequential(
                Conv2d_BN(current_input_channels, inp_channel, 3, 2, 1, norm_cfg=norm_cfg),
                activation() # Instantiate activation
            )
        
        self.cfgs = cfgs
        # This inp_channel is the starting number of channels for the sequence of InvertedResidual blocks
        current_block_input_channels = inp_channel 

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(current_block_input_channels, output_channel, ks=k, stride=s, expand_ratio=t, norm_cfg=norm_cfg,
                                     activations=activation) # Pass activation class
            self.add_module(layer_name, layer)
            current_block_input_channels = output_channel # Update for the next layer
            self.layers.append(layer_name)

    def forward(self, x):
        if self.stem:
            x = self.stem_block(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        return x
    
class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        # Ensure shape is an integer for 1D interpolation
        if isinstance(shape, (list, tuple)) and len(shape) == 1:
            shape = shape[0]
        elif not isinstance(shape, int):
            raise ValueError(f"Shape for SqueezeAxialPositionalEmbedding must be an int or a single-element list/tuple, got {shape}")
            
        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        interpolated_pos_embed = F.interpolate(self.pos_embed, size=(N,), mode='linear', align_corners=False)
        x = x + interpolated_pos_embed
        return x
        
class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None, # Should be a class
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim) 
        self.dh = self.d * num_heads # Total dimension for values across all heads
        self.attn_ratio = attn_ratio

        if activation is None:
            activation = nn.ReLU # Default activation class

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg) # v has total dimension self.dh
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN( # Instantiate activation
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg)) # Project concatenated head values back to 'dim'
        
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16) 
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN( 
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg)) 
        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg)) 
        
        self.shortcut_conv = Conv2d_BN(dim, dim, ks=1, norm_cfg=norm_cfg) 
        self.dwconv_gate = Conv2d_BN(dim, dim, ks=3, stride=1, pad=1, dilation=1, groups=dim, norm_cfg=norm_cfg)
        self.act_gate = activation() 
        self.pwconv_gate = Conv2d_BN(dim, dim, ks=1, norm_cfg=norm_cfg)
        
        self.sigmoid = h_sigmoid()
        
    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape

        q = self.to_q(x) # B, nh_kd, H, W
        k = self.to_k(x) # B, nh_kd, H, W
        v = self.to_v(x) # B, self.dh, H, W
        
        qrow_embed = q.mean(dim=-1) # B, nh_kd, H
        krow_embed = k.mean(dim=-1) # B, nh_kd, H
        
        # qrow: (B, num_heads, H, key_dim)
        qrow = self.pos_emb_rowq(qrow_embed).reshape(B, self.num_heads, self.key_dim, H).permute(0, 1, 3, 2)
        # krow: (B, num_heads, key_dim, H) -> this is K_transposed form for Q @ K^T
        krow = self.pos_emb_rowk(krow_embed).reshape(B, self.num_heads, self.key_dim, H)
        
        # vrow: (B, num_heads, H, self.d) (self.d is value dimension per head)
        vrow = v.mean(dim=-1).reshape(B, self.num_heads, self.d, H).permute(0, 1, 3, 2)
        
        # Corrected: Q @ K^T. krow is already K^T effectively.
        attn_row = torch.matmul(qrow, krow) * self.scale 
        attn_row = attn_row.softmax(dim=-1) # Softmax over key sequence (which is H for krow)
        xx_row = torch.matmul(attn_row, vrow)  # (B, nH, H, d_v_head)
        # Reshape and project: (B, self.dh, H, 1)
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        qcol_embed = q.mean(dim=-2) # B, nh_kd, W
        kcol_embed = k.mean(dim=-2) # B, nh_kd, W

        # qcolumn: (B, num_heads, W, key_dim)
        qcolumn = self.pos_emb_columnq(qcol_embed).reshape(B, self.num_heads, self.key_dim, W).permute(0, 1, 3, 2)
        # kcolumn: (B, num_heads, key_dim, W) -> K_transposed form
        kcolumn = self.pos_emb_columnk(kcol_embed).reshape(B, self.num_heads, self.key_dim, W)
        
        # vcolumn: (B, num_heads, W, self.d)
        vcolumn = v.mean(dim=-2).reshape(B, self.num_heads, self.d, W).permute(0, 1, 3, 2)
        
        # Corrected: Q @ K^T
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1) # Softmax over key sequence (W for kcolumn)
        xx_column = torch.matmul(attn_column, vcolumn)  # (B, nH, W, d_v_head)
        # Reshape and project: (B, self.dh, 1, W)
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        # Combine axial attentions
        attn_out = xx_row + xx_column # Broadcasting sum
        attn_out = v + attn_out 
        attn_out = self.proj(attn_out) # Project back to 'dim' channels: B, dim, H, W

        # Gating mechanism
        gate = self.shortcut_conv(x) 
        gate = self.act_gate(self.dwconv_gate(gate))
        gate = self.pwconv_gate(gate) 
        
        gated_attn_out = self.sigmoid(gate) * attn_out 
        return gated_attn_out

class Block(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.1,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)): 
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = Sea_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                  activation=act_layer, norm_cfg=norm_cfg)
    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg) 

    def forward(self, x1): 
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1

class BasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0.1, attn_drop=0.1, drop_path=0., 
                 norm_cfg=dict(type='BN2d', requires_grad=True), 
                 act_layer=None): 
        super().__init__()
        self.block_num = block_num

        if act_layer is None:
            act_layer = nn.ReLU 

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg,
                act_layer=act_layer)) 

    def forward(self, x):
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

class Fusion_block(nn.Module):
    def __init__(
            self,
            inp: int, 
            oup: int, 
            embed_dim: int, 
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None, 
    ) -> None:
        super(Fusion_block, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(inp, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(oup, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid() 

    def forward(self, x_l, x_g):
        B_l, C_l, H_l, W_l = x_l.shape
        B_g, C_g, H_g, W_g = x_g.shape 

        local_feat = self.local_embedding(x_l) 
        global_act_map = self.global_act(x_g)   
        
        sig_act = F.interpolate(self.act(global_act_map), size=(H_l, W_l), mode='bilinear', align_corners=False)
        
        out = local_feat * sig_act 
        return out

class SeaFormerBackbone(nn.Module):
    def __init__(self, 
                 cfgs, 
                 channels, 
                 emb_dims, 
                 key_dims, 
                 depths=[2,2], 
                 num_heads=8,  
                 attn_ratios=2, 
                 mlp_ratios=[2,4], 
                 drop_path_rate=0.,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6, 
                 in_channels=3, 
                 init_cfg=None): 
        super().__init__()
        self.channels = channels 
        self.emb_dims = emb_dims # Storing emb_dims for reference if needed, e.g. in forward pass logic checks
        self.depths = depths
        self.cfgs = cfgs
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg 
        self.in_channels = in_channels 

        num_smb_stages = len(cfgs)
        num_trans_stages = len(depths)
        if len(channels) < num_smb_stages:
            raise ValueError(f"Length of 'channels' ({len(channels)}) must be at least number of smb_stages ({num_smb_stages})")
        if len(self.emb_dims) != num_trans_stages:
            raise ValueError(f"Length of 'emb_dims' ({len(self.emb_dims)}) must match 'depths' ({num_trans_stages})")
        if len(key_dims) != num_trans_stages:
            raise ValueError(f"Length of 'key_dims' ({len(key_dims)}) must match 'depths' ({num_trans_stages})")
        if isinstance(mlp_ratios, list) and len(mlp_ratios) != num_trans_stages:
            raise ValueError(f"Length of 'mlp_ratios' list ({len(mlp_ratios)}) must match 'depths' ({num_trans_stages})")

        current_input_channels_to_smb = self.in_channels 
        smb_output_channels_list = []

        for i in range(num_smb_stages):
            is_stem = (i == 0)
            smb_inp_channel_arg = self.channels[i] 

            smb = StackedMV2Block(
                cfgs=self.cfgs[i], 
                stem=is_stem, 
                inp_channel=smb_inp_channel_arg, 
                in_channels=current_input_channels_to_smb if is_stem else smb_output_channels_list[-1], 
                norm_cfg=self.norm_cfg,
                activation=act_layer 
            )
            setattr(self, f"smb{i + 1}", smb)
            
            stage_output_channels = _make_divisible(self.cfgs[i][-1][2] * 1.0, 8)
            smb_output_channels_list.append(stage_output_channels)

        self.smb_output_channels_list = smb_output_channels_list 

        for j in range(num_trans_stages):
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths[j])]  
            
            current_mlp_ratio = mlp_ratios[j] if isinstance(mlp_ratios, list) else mlp_ratios

            trans = BasicLayer(
                block_num=depths[j],
                embedding_dim=self.emb_dims[j], 
                key_dim=key_dims[j],
                num_heads=num_heads,
                mlp_ratio=current_mlp_ratio,
                attn_ratio=attn_ratios, 
                drop=0, attn_drop=0, 
                drop_path=dpr,
                norm_cfg=norm_cfg, 
                act_layer=act_layer 
            )
            setattr(self, f"trans{j + 1}", trans)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): 
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        outputs = []
        num_smb_stages = len(self.cfgs)
        num_trans_stages = len(self.depths)
        
        current_feature_map = x

        for i in range(num_smb_stages):
            smb = getattr(self, f"smb{i + 1}")
            current_feature_map = smb(current_feature_map)
            
            if i == 1: 
                outputs.append(current_feature_map)
            
            if i >= (num_smb_stages - num_trans_stages) :
                trans_stage_index_from_zero = i - (num_smb_stages - num_trans_stages)
                if trans_stage_index_from_zero < num_trans_stages: # Ensure we don't go out of bounds
                    trans = getattr(self, f"trans{trans_stage_index_from_zero + 1}")
                    
                    expected_trans_input_channels = self.emb_dims[trans_stage_index_from_zero]
                    if current_feature_map.shape[1] != expected_trans_input_channels:
                        print(f"Warning: Channel mismatch for trans{trans_stage_index_from_zero+1}. Input has {current_feature_map.shape[1]}, trans expects {expected_trans_input_channels}. This might lead to errors.")

                    transformed_feature_map = trans(current_feature_map)
                    outputs.append(transformed_feature_map)
                    current_feature_map = transformed_feature_map # Output of trans becomes input for next potential trans (if logic were nested)
                                                                  # or just the last feature map if this is the last trans.

        return outputs

class LightHead(nn.Module):
    def __init__(self, 
                 in_channels, 
                 channels,    
                 embed_dims,  
                 num_classes, 
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'), 
                 align_corners=False,
                 is_dw=False): 
        super(LightHead, self).__init__()
        
        self.in_channels_list = in_channels 
        self.channels = channels 
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg 
        self.align_corners = align_corners
        self.embed_dims = embed_dims 

        if len(self.in_channels_list) != len(self.embed_dims) + 1:
            raise ValueError(f"LightHead: len(in_channels) ({len(self.in_channels_list)}) must be len(embed_dims) ({len(self.embed_dims)}) + 1.")

        current_local_channels = self.in_channels_list[0]
        for i in range(len(self.embed_dims)):
            fuse = Fusion_block(
                inp=current_local_channels,         
                oup=self.in_channels_list[i+1],     
                embed_dim=self.embed_dims[i],       
                norm_cfg=self.norm_cfg
            )
            setattr(self, f"fuse{i + 1}", fuse)
            current_local_channels = self.embed_dims[i] 
        
        self.linear_fuse = ConvModule(
            in_channels=self.embed_dims[-1], 
            out_channels=self.channels,      
            kernel_size=1,
            stride=1,
            groups=self.channels if is_dw else 1, 
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg 
        )
        
        self.cls_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1) 
        
    def forward(self, inputs): 
        if len(inputs) != len(self.in_channels_list):
             raise ValueError(f"LightHead forward: expected {len(self.in_channels_list)} input feature maps, got {len(inputs)}")

        x_detail = inputs[0] 
        for i in range(len(self.embed_dims)): 
            fuse = getattr(self, f"fuse{i + 1}")
            x_detail = fuse(x_detail, inputs[i+1]) 
        
        _c = self.linear_fuse(x_detail) 
        x = self.cls_seg(_c) 
        return x

class SeaFormer(nn.Module):
    def __init__(self,
                 cfgs,
                 channels, 
                 emb_dims, 
                 key_dims, 
                 depths=[2,2],
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=[2,4],
                 drop_path_rate=0.,
                 num_classes=1000,
                 head_channels=128, 
                 embed_dims_head=None, 
                 in_channels=3, 
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6, 
                 align_corners=False,
                 init_cfg=None): 
        super().__init__()
        
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.in_channels = in_channels 
        
        self.backbone = SeaFormerBackbone(
            cfgs=cfgs,
            channels=channels, 
            emb_dims=emb_dims, 
            key_dims=key_dims,
            depths=depths,
            num_heads=num_heads,
            attn_ratios=attn_ratios,
            mlp_ratios=mlp_ratios,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            act_layer=act_layer, 
            in_channels=self.in_channels, 
            init_cfg=init_cfg
        )
        
        smb2_output_channels = _make_divisible(cfgs[1][-1][2] * 1.0, 8) 
        
        head_in_channels_list = [smb2_output_channels] + emb_dims 

        if embed_dims_head is None:
            embed_dims_head = emb_dims 
            
        self.decode_head = LightHead(
            in_channels=head_in_channels_list, 
            channels=head_channels,            
            embed_dims=embed_dims_head,        
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg={'type': act_layer.__name__ if act_layer else 'ReLU'}, 
            align_corners=align_corners
        )
        
        self.init_weights() 
    
    def init_weights(self):
        self.backbone.init_weights() 
        
        for m in self.decode_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def encode_decode(self, img):
        x = self.extract_feat(img)
        out = self.decode_head(x)
        
        out = F.interpolate(
            input=out,
            size=img.shape[2:], 
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    
    def forward(self, img):
        return self.encode_decode(img)

# Model configuration
def get_seaformer_config():
    return dict(
        cfg1=[ 
            [3, 1, 16, 1],  
            [3, 4, 32, 2],  
            [3, 3, 32, 1]], 
        cfg2=[ 
            [5, 3, 64, 2],  
            [5, 3, 64, 1]], 
        cfg3=[ 
            [3, 3, 128, 2],  
            [3, 3, 128, 1]],
        cfg4=[ 
            [5, 4, 192, 2]], 
        cfg5=[ 
            [3, 6, 256, 2]], 
        
        channels=[16, 32, 64, 128, 192, 256], 
        num_heads=8, 
        emb_dims=[192, 256], 
        key_dims=[16, 24], 
        depths=[4, 4],      
        drop_path_rate=0.1,
        mlp_ratios=[4, 4],  
        attn_ratios=2          
    )

def SeaFormerPP(in_channels, num_classes, **kwargs):
    config = get_seaformer_config()
    
    backbone_cfgs = [config['cfg1'], config['cfg2'], config['cfg3'], config['cfg4'], config['cfg5']]

    norm_cfg = kwargs.pop('norm_cfg', dict(type='BN', requires_grad=True))
    act_layer = kwargs.pop('act_layer', nn.ReLU6) 

    model = SeaFormer(
        cfgs=backbone_cfgs,
        channels=config['channels'],
        key_dims=config['key_dims'],
        emb_dims=config['emb_dims'],
        depths=config['depths'],
        num_heads=config['num_heads'],
        attn_ratios=config['attn_ratios'], 
        mlp_ratios=config['mlp_ratios'],
        drop_path_rate=config['drop_path_rate'],
        num_classes=num_classes,
        in_channels=in_channels,
        norm_cfg=norm_cfg,
        act_layer=act_layer,
        **kwargs 
    )
    
    return model