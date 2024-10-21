import torch
from timm.layers import to_2tuple
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbeddings(nn.Module):
    """
    Patch embedding layer
    patch_dim: ph * pw * c, 每一个patch 拉开多长
    dim: 映射的每一个patch的维度
    """

    def __init__(self, patch_size, patch_dim, dim):
        super().__init__()
        self.patch_height, self.patch_width = to_2tuple(patch_size)
        self.proj = nn.Sequential(
            Rearrange('b c (h_len h) (w_len w) -> b (h w) (h_len w_len c)',
                      h_len=self.patch_height, w_len=self.patch_width),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        return self.proj(x)


class CLSToken(nn.Module):
    """
    Prepend cls token to each embedding
    """

    def __init__(self, dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # 形状为 batch * 1 * dim
        x = torch.cat((cls_tokens, x), dim=1)
        return x


class PositionalEmbeddings(nn.Module):
    """
    Learned positional embeddings
    """

    def __init__(self, num_pos, dim):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_pos, dim))

    def forward(self, x):
        return x + self.pos