import torch
from einops import rearrange
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio


def loadData():  # 读取数据集文件
    # 读入数据
    data = sio.loadmat('./data/Indian_pines/Indian_pines_corrected.mat')['indian_pines_corrected']
    labels = sio.loadmat('./data/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
    return data, labels


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def ImageCut(X, y, window_size=11, remove_zero_labels=True):
    """
    从输入图像 X 中提取每个像素周围的 patch，并与对应的标签 y 结合形成符合 Keras 处理的数据格式。

    参数:
    X: 输入图像，形状为 (height, width, channels)
    y: 标签矩阵，形状为 (height, width)
    window_size: 提取的 patch 大小，必须为奇数 (默认为 5)
    remove_zero_labels: 是否移除标签为 0 的 patch (默认为 True)

    返回:
    patches_data: 提取的 patch 数据，形状为 (num_patches, window_size, window_size, channels)
    patches_labels: 对应的标签，形状为 (num_patches,)
    """

    # 计算填充的边界大小
    margin = window_size // 2
    padded_X = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

    # 初始化 patch 数据和标签
    height, width = X.shape[:2]
    num_patches = height * width
    patches_data = np.zeros((num_patches, window_size, window_size, X.shape[2]))
    patches_labels = np.zeros(num_patches)

    # 遍历每个像素提取 patch
    patch_idx = 0
    for i in range(margin, margin + height):
        for j in range(margin, margin + width):
            patch = padded_X[i - margin:i + margin + 1, j - margin:j + margin + 1]
            patches_data[patch_idx] = patch
            patches_labels[patch_idx] = y[i - margin, j - margin]
            patch_idx += 1

    # 可选地移除标签为 0 的 patch
    if remove_zero_labels:
        mask = patches_labels > 0
        patches_data = patches_data[mask]
        patches_labels = patches_labels[mask] - 1  # 使标签从 0 开始

    return patches_data, patches_labels


class PatchDataset(Dataset):
    """
    自定义 Dataset 用于加载 patch 数据
    """

    def __init__(self, patches_data, patches_labels):
        self.patches_data = torch.FloatTensor(patches_data)
        self.patches_labels = torch.LongTensor(patches_labels)

    def __len__(self):
        return len(self.patches_labels)

    def __getitem__(self, idx):
        return self.patches_data[idx], self.patches_labels[idx]


def DataLoaders(X, y, testRatio=0.9, batch_size=64, randomState=324, shuffle=True):
    """
    训练和测试数据加载器

    参数:
    X: 提取的 patch 数据，形状为 (num_patches, window_size, window_size, channels)
    y: 对应的标签，形状为 (num_patches,)
    testRatio: 测试集所占比例 (默认为 0.2，即 20%)
    batch_size: 每个批次的数据大小 (默认为 32)
    randomState: 随机种子
    shuffle: 是否在每个 epoch 开始时打乱数据 (默认为 True)

    返回:
    train_loader: 训练集的 DataLoader
    形状为: b * 1 * c * h * w
    test_loader: 测试集的 DataLoader
    """

    # 使用 sklearn 的 train_test_split 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    print('\n为了适应 pytorch 结构，数据要做 transpose')
    X_train = rearrange(X_train, 'b h w c -> b 1 c h w')
    X_test = rearrange(X_test, 'b h w c -> b 1 c h w')
    X = rearrange(X, 'b h w c -> b 1 c h w')

    print('after transpose: Xtrain shape: ', X_train.shape)
    print('after transpose: Xtest  shape: ', X_test.shape)
    print('\n')

    # 创建 PyTorch 的训练和测试集 Dataset
    train_dataset = PatchDataset(X_train, y_train)
    test_dataset = PatchDataset(X_test, y_test)
    all_datatest = PatchDataset(X, y)

    # 创建 DataLoader
    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集一般不需要 shuffle
    allLoader = DataLoader(all_datatest, batch_size=batch_size, shuffle=False)  # 所有的数据

    return trainLoader, testLoader, allLoader


def PreprocessedData(batch_size=64, patch_size=13, test_ratio=0.9, pca_components=30):
    X, y = loadData()
    print('高光谱图形的维度形状为: ', X.shape)
    print('正确的标签形状为: ', y.shape)

    if pca_components != 0:
        print('\n... ... PCA(降维度) 转变 ... ...')
        X = applyPCA(X, numComponents=pca_components)
        print('降维度后的高光谱形状为: ', X.shape)

    print(f'\n以每一个像素点为中心形成宽高为 patch = {patch_size} 的立方体')
    X_pca, y_pca = ImageCut(X, y, window_size=patch_size, remove_zero_labels=True)
    print('所有立方体的 X 的形状为: ', X_pca.shape)
    print('立方体的标签 y 为: ', y_pca.shape)

    print('\n... ... 创建训练和测试数据 ... ...')
    trainLoader, testLoader, allLoader = DataLoaders(X_pca, y_pca, batch_size=batch_size, testRatio=test_ratio)
    return trainLoader, testLoader, allLoader, y


if __name__ == '__main__':
    "批量大小, 测试数据比例, 通道的压缩维度, 分类数量"
    BATCH_SIZE = 64
    TEST_RATIO = 0.9
    PcaNum = 30
    class_num = 16
    train_loader, test_loader, all_data_loader, y_all = PreprocessedData(
        batch_size=BATCH_SIZE, test_ratio=TEST_RATIO, pca_components=PcaNum)
