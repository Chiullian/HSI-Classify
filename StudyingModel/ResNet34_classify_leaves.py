import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils import data
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.nn import functional as F
from tqdm import tqdm

learning_rate, num_epochs, batch_size = 0.05, 10, 8

# 读取所有labels 并做好映射
labels_dataframe = pd.read_csv('./data/train.csv')
data_labels = sorted(list(set(labels_dataframe['label'])))

n_classes = len(data_labels)

class_to_num = dict(zip(data_labels, range(n_classes)))
num_to_class = {v: k for k, v in class_to_num.items()}


def get_device():
    if torch.cuda.device_count() > 0:
        return torch.device('cuda')
    return torch.device('cpu')


# 继承pytorch的dataset，创建自己的
class MyDataset(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """

        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # 第一列包含图像文件的名称
            # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])
            # 第二列是图像的 label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of  Dataset ({} samples found)'.format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len


# 读取数据路径
train_path = './data/train.csv'
test_path = './data/test.csv'
img_path = './data/'

# 加载数据集
train_dataset = MyDataset(train_path, img_path, mode='train')
valid_dataset = MyDataset(train_path, img_path, mode='valid')
test_dataset = MyDataset(test_path, img_path, mode='test')

# 加载 data迭代器
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 展示图片
def show_imges(imges, titles=None, columns=4, rows=2):
    def im_convert(tensor):
        """ 展示数据"""
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image.clip(0, 1)
        return image

    fig = plt.figure(figsize=(10, 8))

    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        ax.set_title(num_to_class[int(titles[idx])])
        plt.imshow(im_convert(imges[idx]))
    plt.show()


# inputs, labels = next(iter(train_loader))
# show_imges(inputs, labels)


class Residual(nn.Module):  # 残差块
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = (self.bn2(self.conv2(Y)))
        return F.relu(Y + self.conv3(X)) if self.conv3 else F.relu(Y + X)


def ResNet_block(input_channels, output_channels, num_residuals, first_block=False):
    Block_list = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            Block_list.append(Residual(input_channels, output_channels, use_1x1conv=True, strides=2))
        else:
            Block_list.append(Residual(output_channels, output_channels))
    return Block_list


b1 = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*ResNet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*ResNet_block(64, 128, 2))
b4 = nn.Sequential(*ResNet_block(128, 256, 2))
b5 = nn.Sequential(*ResNet_block(256, 512, 2))

model = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                      nn.Flatten(), nn.Linear(512, n_classes))


def start_train(model, train_loader, valid_loader, num_epochs, learning_rate, device):
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)
    print('training on', device)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # --------------- Training ---------------
        model.train()
        print(f'Starting Epoch {epoch + 1}')
        train_loss = []  # 计算损失值()
        train_accs = []  # 记录训练正确率

        for images, labels in tqdm(train_loader, colour='blue'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            # 我们不许要使用softmax因为求交叉熵的时候会自动done
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        # 训练集的平均损失和准确度的平均值
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # -------------------- Validation --------------------
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        for images, labels in tqdm(valid_loader, colour='green'):
            images, labels = images.to(device), labels.to(device)
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(images)

            # We can still compute the loss (but not the gradient).
            loss = loss_function(logits, labels)

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        # 验证集集的平均损失和准确度的平均值
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


start_train(model, train_loader, valid_loader, num_epochs, learning_rate, get_device())


def predict(model, test_loader, device):
    # 预测并且保存
    model.eval()
    predictions = []
    # Iterate the testing set by batches.
    for images in tqdm(test_loader):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    preds = []
    for i in predictions:
        preds.append(num_to_class[i])

    test_data = pd.read_csv(test_path)
    test_data['label'] = pd.Series(preds)
    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    saveFileName = './submission.csv'
    submission.to_csv(saveFileName, index=False)


predict(model, test_loader, get_device())
print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")

