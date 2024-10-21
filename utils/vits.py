import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from transformer import MultiHeadAttention


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.Layernorm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.Layernorm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * n_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.n_heads = n_heads
        "用于消除误差，保证方差为1，避免向量内积过大导致的softmax将许多输出置0的情况"
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def forward(self, X):
        """注意到这里的X形状为 b * n * dim"""
        qkv = self.to_qkv(X).chunk(3, dim=-1)
        """qkv的形状为b * n * 3 * inner_dim, 这也就相当于平分原来一个大矩阵当作q, k, v"""
        Q, K, V = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)
        """类似于transformer里的把多头和batch_size放在一起， 这样就可以只用一次矩阵运算"""
        dots = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, X):
        return self.net(X)


class Transformer(nn.Module):
    """
    包含两大部分
    1. 多头注意力机制部分
    2. 前馈神经网络部分
    """

    def __init__(self, dim, n_layers, n_heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, n_heads=n_heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(torch.nn.Module):
    """"
        参数解释大全
        image_size: (h, w) 图片的高宽
        patch_size: (h, w) 切成小块的大小
        dim: 相当于 transformer 里的 d_models， 固定词向量的长度
        n_classes: 分类任务的分类数量
        n_layers: 迭代块的数量
        n_heads: 多头的数量
        dim_head: 每一个头的维度是多少
        mlp_dim: 前馈神经网络的隐藏维度的大小
        channels: 图片通道的数量
    """

    def __init__(self, image_size, patch_size, dim, n_classes, n_layers, n_heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0.1, emb_dropout=0.1):
        super(ViT, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        "必须能把图片正好分割"
        assert image_height % patch_height == 0 and image_width % patch_width == 0

        "一个大图片能分成多少个块"
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        "一个块能能拉成多长"
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        "一种优雅的方式变换向量的维度"
        self.to_path_emb = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, n_layers, n_heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )

    def forward(self, image):
        X = self.to_path_emb(image)
        b, n, _ = X.shape  # X 的形状为 batch * n * dim
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)  # 形状为 batch * 1 * dim
        "头部添加一个 cls 的 token 引用 bert 的技巧, 该cls可以同时获取全局的信息(由于自注意力机制)"
        X = torch.cat((cls_token, X), dim=1)
        "利用广播机制把 每一个n加上位置信息"
        X += self.pos_embedding[:, :(n + 1)]

        X = self.dropout(X)
        X = self.transformer(X)
        "其实不加cls也行也就是最后池化的时候取个平均， 两种方法"
        X = X.mean(dim=1) if self.pool == 'mean' else X[:, 0]  # b n + 1 dim 只取第第一个 b 1 dim, 第一个包含了所有的信息!
        return self.mlp_head(X)


v = ViT(
    image_size=224,
    patch_size=16,
    n_classes=1000,
    dim=1024,
    n_layers=6,
    n_heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(1, 3, 224, 224)

preds = v(img)  # (1, 1000)
print(max(nn.Softmax()(preds)))
print(preds.shape)