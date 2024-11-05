# Spectral–Spatial Morphological Attention  Transformer for Hyperspectral Image Classification

研究的背景:
该论文提及之前出现的论文要么是从光谱维度进行抽特征和从空间维度进行抽取特征, 没有很好的去融合两个维度的特征, 所以本文就提出了一种用于空谱联合的训练模型

![](https://image.chiullian.cn/img/202411051208794.png)

分为几个步骤:
1. 这里没有使用PCA进行降维度,而是使用卷积进行学习并且降维
> ![](https://image.chiullian.cn/img/202411051309378.png)
> 正常的一个patch 进行输入,然后进行卷积(K=(9 * 3 * 3), p=(0 * 1 * 1))降维度
> 然后Reshae成二维的, 然后进行两个并行的卷积, 一半用分组卷积,一半用点卷积
> , 过程如下
> ![](https://image.chiullian.cn/img/202411051326495.png)

这里的输出模块的形状为: 11 * 11 * 64 (h * w * c)

![](https://image.chiullian.cn/img/202411051328254.png)
