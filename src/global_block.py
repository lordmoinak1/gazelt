#!/usr/bin/env python3

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# written by Ali Hatamizadeh and Pavlo Molchanov from NVResearch


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models._registry import register_model
from timm.models._builder import build_model_with_cfg


def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000, 
            'input_size': (3, 224, 224), 
            'pool_size': None,
            'crop_pct': 0.875, 
            'interpolation': 'bicubic', 
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406), 
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


default_cfgs = {
    'gc_vit_xxtiny': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_xxtiny.pth.tar',
                          crop_pct=1.0, 
                          input_size=(3, 224, 224), 
                          crop_mode= 'center'),
    'gc_vit_xtiny': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_xtiny.pth.tar',
                         crop_pct=0.875, 
                         input_size=(3, 224, 224), 
                         crop_mode= 'center'),
    'gc_vit_tiny': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_tiny.pth.tar',
                        crop_pct=1.0, 
                        input_size=(3, 224, 224), 
                        crop_mode= 'center'),
    'gc_vit_tiny2': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_tiny2.pth.tar',
                         crop_pct=0.875, 
                         input_size=(3, 224, 224), 
                         crop_mode= 'center'),
    'gc_vit_small': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_small.pth.tar',
                         crop_pct=0.875, 
                         input_size=(3, 224, 224), 
                         crop_mode= 'center'),
    'gc_vit_small2': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_small2.pth.tar',
                          crop_pct=0.875, 
                          input_size=(3, 224, 224), 
                          crop_mode= 'center'),
    'gc_vit_base': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_base.pth.tar',
                        crop_pct=1.0, 
                        input_size=(3, 224, 224), 
                        crop_mode= 'center'),
    'gc_vit_large': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_large.pth.tar',
                         crop_pct=1.0, 
                         input_size=(3, 224, 224), 
                         crop_mode= 'center'),
    'gc_vit_large_224_21k': _cfg(url='https://drive.google.com/uc?export=download&id=1maGDr6mJkLyRTUkspMzCgSlhDzNRFGEf', 
                                 crop_pct=1.0, 
                                 input_size=(3, 224, 224), 
                                 crop_mode= 'center'),
    'gc_vit_large_384_21k': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_21k_large_384.pth.tar', 
                                 crop_pct=1.0, 
                                 input_size=(3, 384, 384), 
                                 crop_mode='squash'),
    'gc_vit_large_512_21k': _cfg(url='https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_21k_large_512.pth.tar', 
                                 crop_pct=1.0, 
                                 input_size=(3, 512, 512), 
                                 crop_mode='squash'),                             
}


def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)

    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, C)

    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size, h_w, w_w):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size

    Returns:
        local window features (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, h_w, window_size, w_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, h_w, w_w, B):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    # B = int(windows.shape[0] // (H * W // window_size // window_size))
    x = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ReduceSize(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 norm_layer=nn.LayerNorm,
                 keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False)
        self.norm2 = norm_layer(dim_out)
        self.norm1 = norm_layer(dim)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = _to_channel_first(x)
        x = x + self.conv(x)
        x = self.reduction(x)
        x = _to_channel_last(x)
        x = self.norm2(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, in_chans=3, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """

        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, 3, 2, 1)
        self.conv_down = ReduceSize(dim=dim, keep_dim=True)

    def forward(self, x):
        x = self.proj(x)
        x = _to_channel_last(x)
        x = self.conv_down(x)
        return x


class FeatExtract(nn.Module):
    """
    Feature extraction block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x


class WindowAttention(nn.Module):
    """
    Local window attention based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttentionGlobal(nn.Module):
    """
    Global window attention based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape
        B = q_global.shape[0]
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        B_dim = torch.div(B_, B, rounding_mode='floor')
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q_global = q_global.repeat(1, B_dim, 1, 1, 1)
        q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GCViTBlock(nn.Module):
    """
    GCViT block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=WindowAttentionGlobal,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 ):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            num_heads: number of attention head.
            window_size: window size.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            act_layer: activation function.
            attention: attention block type.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """

        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)

        self.attn = attention(dim,
                              num_heads=num_heads,
                              window_size=window_size,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

        inp_w = torch.div(input_resolution, window_size, rounding_mode='floor')
        self.num_windows = int(inp_w * inp_w)

    def forward(self, x, q_global):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        h_w = torch.div(H, self.window_size, rounding_mode='floor')
        w_w = torch.div(W, self.window_size, rounding_mode='floor')
        x_windows = window_partition(x, self.window_size, h_w, w_w)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, q_global)
        x = window_reverse(attn_windows, self.window_size, H, W, h_w, w_w, B)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class GlobalQueryGen(nn.Module):
    """
    Global query generator based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 image_resolution,
                 window_size,
                 num_heads):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            num_heads: number of heads.

        For instance, repeating log(56/7) = 3 blocks, with input window dimension 56 and output window dimension 7 at
        down-sampling ratio 2. Please check Fig.5 of GC ViT paper for details.
        """

        super().__init__()
        if input_resolution == image_resolution//4:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == image_resolution//8:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == image_resolution//16:

            if window_size == input_resolution:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=True)
                )

            else:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=True)
                )

        elif input_resolution == image_resolution//32:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=True)
            )

        self.resolution = input_resolution
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = torch.div(dim, self.num_heads, rounding_mode='floor')

    def forward(self, x):
        x = _to_channel_last(self.to_q_global(x))
        B = x.shape[0]
        x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        return x


class GCViTLayer(nn.Module):
    """
    GCViT layer based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 depth,
                 input_resolution,
                 image_resolution,
                 num_heads,
                 window_size,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            input_resolution: input image resolution.
            window_size: window size in each stage.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """

        super().__init__()
        self.blocks = nn.ModuleList([
            GCViTBlock(dim=dim,
                       num_heads=num_heads,
                       window_size=window_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias,
                       qk_scale=qk_scale,
                       attention=WindowAttention if (i % 2 == 0) else WindowAttentionGlobal,
                       drop=drop,
                       attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer,
                       layer_scale=layer_scale,
                       input_resolution=input_resolution)
            for i in range(depth)])
        self.downsample = None if not downsample else ReduceSize(dim=dim, norm_layer=norm_layer)
        self.q_global_gen = GlobalQueryGen(dim, input_resolution, image_resolution, window_size, num_heads)

    def forward(self, x):
        q_global = self.q_global_gen(_to_channel_first(x))
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is None:
            return x
        return self.downsample(x)


class GCViT(nn.Module):
    """
    GCViT based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            resolution: input image resolution.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """
        super().__init__()

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            level = GCViTLayer(dim=int(dim * 2 ** i),
                               depth=depths[i],
                               num_heads=num_heads[i],
                               window_size=window_size[i],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer,
                               downsample=(i < len(depths) - 1),
                               layer_scale=layer_scale,
                               input_resolution=int(2 ** (-2 - i) * resolution),
                               image_resolution=resolution)
            self.levels.append(level)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        layer_list = []
        for level in self.levels:
            x = level(x)
            layer_list.append(x)

        x = self.norm(x)
        # x = _to_channel_first(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return layer_list, x

    def forward(self, x):
        layer_list, x = self.forward_features(x)
        # x = self.head(x)
        return layer_list, x



def _create_gc_vit(variant, pretrained=False, **kwargs):

    return build_model_with_cfg(
        GCViT,
        variant,
        pretrained,
        **kwargs,
    )


@register_model
def gc_vit_xxtiny(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    model_kwargs = dict(depths=[2, 2, 6, 2],
                        num_heads=[2, 4, 8, 16],
                        window_size=[7, 7, 14, 7],
                        dim=64,
                        mlp_ratio=3,
                        drop_path_rate=drop_path_rate,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_xxtiny', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def gc_vit_xtiny(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    model_kwargs = dict(depths=[3, 4, 6, 5],
                        num_heads=[2, 4, 8, 16],
                        window_size=[7, 7, 14, 7],
                        dim=64,
                        mlp_ratio=3,
                        drop_path_rate=drop_path_rate,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_xtiny', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def gc_vit_tiny(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    model_kwargs = dict(depths=[3, 4, 19, 5],
                        num_heads=[2, 4, 8, 16],
                        window_size=[7, 7, 14, 7],
                        dim=64,
                        mlp_ratio=3,
                        drop_path_rate=drop_path_rate,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_tiny', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def gc_vit_tiny2(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.25)
    model_kwargs = dict(depths=[3, 4, 29, 5],
                        num_heads=[2, 4, 8, 16],
                        window_size=[7, 7, 14, 7],
                        dim=64,
                        mlp_ratio=3,
                        drop_path_rate=drop_path_rate,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_tiny2', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def gc_vit_small(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    model_kwargs = dict(depths=[3, 4, 19, 5],
                        num_heads=[3, 6, 12, 24],
                        window_size=[7, 7, 14, 7],
                        dim=96,
                        mlp_ratio=2,
                        drop_path_rate=drop_path_rate,
                        layer_scale=1e-5,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_small', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def gc_vit_small2(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.35)
    model_kwargs = dict(depths=[3, 4, 23, 5],
                        num_heads=[3, 6, 12, 24],
                        window_size=[7, 7, 14, 7],
                        dim=96,
                        mlp_ratio=3,
                        drop_path_rate=drop_path_rate,
                        layer_scale=1e-5,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_small2', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def gc_vit_base(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.5)
    model_kwargs = dict(depths=[3, 4, 19, 5],
                        num_heads=[4, 8, 16, 32],
                        window_size=[7, 7, 14, 7],
                        dim=128,
                        mlp_ratio=2,
                        drop_path_rate=drop_path_rate,
                        layer_scale=1e-5,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_base', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def gc_vit_large(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.5)
    model_kwargs = dict(depths=[3, 4, 19, 5],
                        num_heads=[6, 12, 24, 48],
                        window_size=[7, 7, 14, 7],
                        dim=192,
                        mlp_ratio=2,
                        drop_path_rate=drop_path_rate,
                        layer_scale=1e-5,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_large', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def gc_vit_large_224_21k(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.5)
    model_kwargs = dict(depths=[3, 4, 19, 5],
                        num_heads=[6, 12, 24, 48],
                        window_size=[7, 7, 14, 7],
                        dim=192,
                        mlp_ratio=2,
                        drop_path_rate=drop_path_rate,
                        layer_scale=1e-5,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_large_224_21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def gc_vit_large_384_21k(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.1)
    model_kwargs = dict(depths=[3, 4, 19, 5],
                        num_heads=[6, 12, 24, 48],
                        window_size=[12, 12, 24, 12],
                        dim=192,
                        mlp_ratio=2,
                        drop_path_rate=drop_path_rate,
                        layer_scale=1e-5,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_large_384_21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def gc_vit_large_512_21k(pretrained=False, **kwargs) -> GCViT:
    drop_path_rate = kwargs.pop("drop_path_rate", 0.1)
    model_kwargs = dict(depths=[3, 4, 19, 5],
                        num_heads=[6, 12, 24, 48],
                        window_size=[12, 12, 32, 16],
                        dim=192,
                        mlp_ratio=2,
                        drop_path_rate=drop_path_rate,
                        layer_scale=1e-5,
                        **kwargs
                        )
    model = _create_gc_vit('gc_vit_large_512_21k', pretrained=pretrained, **model_kwargs)
    return model

if __name__ == '__main__':
    img_size = 224
    x = torch.rand(16, 3, img_size, img_size).cuda()
    model = gc_vit_tiny(pretrained=False).cuda()
    # print(model); 
    model(x)

    # flops = model.flops()
    # print(f"number of GFLOPs: {flops / 1e9}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")