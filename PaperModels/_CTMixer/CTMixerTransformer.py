import math

import torch
from einops import rearrange
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.layernorm(x), **kwargs)


class Mhsa(nn.Module):
    """
    输入形状： b n dim
    输出形状：b n dim
    """
    def __init__(self, dim, num_patch, num_heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.num_patch = num_patch
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Conv2d(dim, self.inner_dim * 3, kernel_size=1, padding=0, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.avgpool = nn.AdaptiveAvgPool1d(dim)

    def forward(self, x):
        """ """
        b, n, _ = x.shape
        x = rearrange(x, 'b (h w) d -> b d h w', h=self.num_patch)
        qkv = self.to_qkv(self.relu(self.bn(x)))
        qkv = qkv.contiguous().view(b, self.num_patch * self.num_patch, self.inner_dim * 3)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        # 做注意力
        q1 = q[:, : self.num_heads // 2]
        k1 = k[:, : self.num_heads // 2]
        v1 = v[:, : self.num_heads // 2]

        sourse = torch.einsum('b h i k, b h j k -> b h i j', q1, k1) * self.scale
        atten = self.attend(sourse)
        out = torch.einsum('b h i j, b h j k -> b h i k', atten, v1)
        ans1 = rearrange(out, 'b h n d -> b n (h d)')

        # 做卷积捕获局部的语义
        q2 = q[:, self.num_heads // 2: self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))
        k2 = k[:, self.num_heads // 2: self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))
        v2 = v[:, self.num_heads // 2: self.num_heads, :, :].reshape(b, -1, int(math.sqrt(n)), int(math.sqrt(n)))

        ans2 = (self.conv1(q2) + self.conv2(k2) + v2).view(b, n, -1)

        ans = torch.cat((ans1, ans2), dim=-1)
        to_out = self.avgpool(ans)

        return to_out


class ResCNN(nn.Module):
    def __init__(self, dim, num_patch, patch_size, hidden_dim=64, dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_patch = num_patch
        self.patch_size = patch_size

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False)
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False, groups=hidden_dim)
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=False)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _ = x.shape
        Res = rearrange(x, 'b (h w) d -> b d h w', h=self.num_patch)
        x = self.conv1(Res)
        x = self.conv2(x)
        x = self.conv3(x) + Res
        x = rearrange(x, 'b d h w -> b (h w) d')
        return self.dropout(x)


class Transformer(nn.Module):
    """
    包含两大部分
    1. 多头注意力机制部分
    2. 前馈神经网络部分
    """

    def __init__(self, dim, n_layers, n_heads, patch_size, num_patch, dim_head, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Mhsa(dim, num_patch, num_heads=n_heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, ResCNN(dim, num_patch, patch_size, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
