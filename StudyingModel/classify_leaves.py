import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils import data
from torchvision import models
from torchvision.transforms import transforms
from torch.nn import functional as F
from tqdm import tqdm
from webencodings import labels

from utils.Z import Animator, Accumulator, Timer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 超参数
learning_rate = 0.0005
weight_decay = 1e-3
num_epochs = 100
batch_size = 16

# 读取所有labels 并做好映射
labels_dataframe = pd.read_csv('data/leaves/train.csv')
data_labels = sorted(list(set(labels_dataframe['label'])))

n_classes = len(data_labels)

class_to_num = dict(zip(data_labels, range(n_classes)))
num_to_class = {v: k for k, v in class_to_num.items()}


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# 继承pytorch的dataset，创建自己的
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            file_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """

        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

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
        # 这是重写的核心方法，Dataloader只在意我们的这个方法的返回结果
        single_image_name = self.image_arr[index]  # 从 image_arr中得到索引对应的文件名
        image_path = os.path.join(self.file_path, single_image_name)  # 获取图片路径
        img_as_img = Image.open(image_path)  # 读取图像文件

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # 颜色亮度色调
                # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
train_path = 'data/leaves/train.csv'
test_path = 'data/leaves/test.csv'
img_path = './data/leaves/'

# 加载数据集
train_dataset = MyDataset(train_path, img_path, mode='train')
valid_dataset = MyDataset(train_path, img_path, mode='valid')
test_dataset = MyDataset(test_path, img_path, mode='test')

# 加载 data迭代器
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


num_batches = len(train_dataset) // batch_size


def train_and_valid(Model, TrainLoader, ValidLoader, NumEpochs, LearningRate, Device):
    print('training on', Device)
    Model.to(Device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=LearningRate, weight_decay=weight_decay)

    best_acc = 0.80
    model_path = './person_models/leaves_pre_res_model.ckpt'
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'valid acc'])
    timer = Timer()

    for epoch in range(NumEpochs):
        # ----------------------------------- Training --------------------------------------
        Model.train()
        print(f'Starting Epoch {epoch + 1}')
        metricT = Accumulator(3)  # 损失值, 正确的数量,  总数
        for i, (images, labels) in enumerate(TrainLoader):
            images, labels = images.to(Device), labels.to(Device)
            timer.start()
            optimizer.zero_grad()
            logits = Model(images)
            # 我们不许要使用softmax因为求交叉熵的时候会自动done
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # 损失的总和， 正确的总数， 训练的数据总数
            metricT.add(loss.sum(), accuracy(logits, labels), labels.shape[0])
            timer.stop()

            if i == batch_size - 1 or (i + 1) % (num_batches // 5) == 0:
                # print(epoch + (i + 1) / num_batches)
                animator.add(epoch + (i + 1) / num_batches, (metricT[0] / metricT[2], metricT[1] / metricT[2], None))

        # 训练集的平均损失和准确度的平均值
        train_loss = metricT[0] / metricT[2]
        train_acc = metricT[1] / metricT[2]
        print(f"[ Train | {epoch + 1:03d}/{NumEpochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ----------------------------------- Validation ----------------------------------------
        Model.eval()
        metricV = Accumulator(2)

        for i, (images, labels) in enumerate(ValidLoader):
            images, labels = images.to(Device), labels.to(Device)
            with torch.no_grad():
                logits = Model(images)
                #  正确数量， 验证总数
                metricV.add(accuracy(logits, labels), labels.shape[0])

        # 验证集集准确度的平均值
        valid_acc = metricV[0] / metricV[1]
        animator.add(epoch + 1, (None, None, valid_acc))

        print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] acc = {valid_acc:.5f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(Model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))


# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(Model, feature_extracting):
    if feature_extracting:
        Model = Model
        for i, param in enumerate(Model.children()):
            if i == 8:
                break
            param.requires_grad = False


# 加载ResNet34
def resnet34(num_classes, feature_extract=False, weights=models.ResNet34_Weights.DEFAULT):
    model_ft = models.resnet34(weights=weights)
    num_ftrs = model_ft.fc.in_features  # 输出的数量
    nn.init.xavier_uniform_(model_ft.fc.weight)

    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.fc.requires_grad = True

    # model_ft.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 512),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(.3),
    #     nn.Linear(512, num_classes)
    # )

    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
    )

    return model_ft


model = resnet34(n_classes)

train_and_valid(model, train_loader, valid_loader, num_epochs, learning_rate, get_device())


def predict(Model, TestLoader, Device):
    # 预测并且保存
    Model.eval()
    predictions = []

    for images in tqdm(TestLoader, colour='yellow'):
        images = images.to(Device)
        with torch.no_grad():
            logits = Model(images)

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    preds = [num_to_class[int(i)] for i in predictions]

    test_data = pd.read_csv(test_path)
    test_data['label'] = pd.Series(preds)
    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    saveFileName = './submission.csv'
    submission.to_csv(saveFileName, index=False)


predict(model, test_loader, get_device())
print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")
