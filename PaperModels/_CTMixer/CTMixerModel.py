import math
import torch
from einops import rearrange
from torch import nn
from torchinfo import summary

from PaperModels._CTMixer.CTMixerEmbeddings import PatchEmbeddings, PositionalEmbeddings
from PaperModels._CTMixer.CTMixerTransformer import Transformer


class GPRB(nn.Module):
    """
    两组卷积
    正常卷积和分组卷积

    """

    def __init__(self, in_channels, kernel_size, padding, groups):
        super(GPRB, self).__init__()
        # 分组卷积
        self.GrConv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 padding=padding, groups=groups)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.GrConv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 padding=padding, groups=groups)
        self.bn2 = nn.BatchNorm2d(in_channels)

        # 正常卷积
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, in_channels, kernel_size=kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.bn1(self.GrConv1(x))
        y = self.relu(y)
        y = self.bn2(self.GrConv2(y))
        z = self.bn3(self.conv1(x))
        z = self.relu(z)
        z = self.bn4(self.conv2(z))
        return self.relu(x + y + z)


class Classify(nn.Module):
    def __init__(self, dim, num_classes):
        super(Classify, self).__init__()
        self.linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        return self.linear(x)


class CNNB(nn.Module):
    """
    残差卷积
    """
    def __init__(self, dim, hidden_dim=64):
        super(CNNB, self).__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = (self.conv1(self.relu(self.bn1(x))))
        z = (self.conv2(self.relu(self.bn2(y))))
        return self.conv3(self.relu(self.bn3(z)))


class CTMixer(nn.Module):
    """
    image_size : 11 * 11 * 200

    深度通道和光谱通道 才可以变成特征维度 dim
    """
    def __init__(self, image_size=11, channels=200, dim=128, num_layer=1, num_classes=16, patch_size=1,
                 num_head=4, hidden_dim=64, groups=11, groups_width=37, pool='mean'):
        super(CTMixer, self).__init__()
        # 预处理阶段
        "必须能把图片正好分割"
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.groups = groups
        self.pool = pool

        self.num_patches = (image_size // patch_size) ** 2
        self.num_patch = int(math.sqrt(self.num_patches))
        self.relu = nn.ReLU(inplace=True)

        # patch_dim = channels * patch_size ** 2
        new_channels = math.ceil(channels / groups) * groups  # 为了让每一组的分组卷积数量相同, 新地输入通道维度

        # 分组分成 groups 组 每组的输出维度为 groups_width (为了更好抓取局部特征), 也就是把原来的框
        new_patch_dim = groups * groups_width
        patch_dim = (groups * groups_width) * patch_size * patch_size  # 每个框拉的特征维度

        # 新加的通道维度, 记录下来要给训练数据X加上
        pad_size = new_channels - channels
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, pad_size))

        self.conv1 = nn.Conv2d(new_channels, new_patch_dim, kernel_size=1, groups=groups)
        self.bn1 = nn.BatchNorm2d(new_patch_dim)
        self.gprb = GPRB(new_patch_dim, (3, 3), (1, 1), groups)

        self.rbb = CNNB(dim)
        self.dropout = nn.Dropout(0.1)

        self.patchEmbedding = PatchEmbeddings(patch_size=patch_size, patch_dim=patch_dim, dim=dim)
        self.positionalEmbedding = PositionalEmbeddings(self.num_patches, dim)

        # transformer
        self.transformer = Transformer(dim, num_layer, num_head, patch_size, self.num_patch, hidden_dim, dropout=0.1)
        self.Classify = Classify(dim, num_classes)

    def forward(self, x):
        x = self.pad(x)
        "x.shape = b 1 c h w"
        x = rearrange(x, "b y c h w -> b (y c) h w")
        b, c, h, w = x.shape
        "x.shape = b c h w"
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.gprb(x)
        "x.shape = b c h w"
        x4 = self.patchEmbedding(x)
        x5 = self.positionalEmbedding(x4)
        res1 = self.transformer(x5)

        x7 = rearrange(x4, 'b (h w) d -> b d h w', h=h)
        x8 = self.rbb(x7)

        res2 = rearrange(x8, 'b d h w -> b (h w) d')
        res = self.dropout(res1 + res2)

        res = res.mean(dim=1) if self.pool == 'mean' else res[:, 0]
        res = self.Classify(res)
        return res


if __name__ == '__main__':
    input = torch.randn(size=(100, 1, 200, 11, 11))
    print("input shape:", input.shape)
    model = CTMixer(image_size=11, channels=200, num_classes=16, patch_size=1)
    parameters = sum([layer.numel() for layer in model.parameters()])
    summary(model, input_size=(100, 1, 200, 11, 11), device='cpu')
    print("output shape:", model(input).shape)
    print('Total number of parameters:', parameters)
