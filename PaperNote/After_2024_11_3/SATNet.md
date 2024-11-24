## Improved Transformer Net for Hyperspectral Image Classification

> 同时使用光谱注意力和自注意力,防止特征信息的丢失

#### 整体模型架构图:

![](https://image.chiullian.cn/img/202411061445371.png)

#### 频谱注意力模块
模块简介: (小创新点) 这一模块主要用于加权光谱维度的特征，从而更好地利用高光谱图像的数据特性。

![](https://image.chiullian.cn/img/202411061417938.png)

> 用两种池化(平均和最大)方式去抽取光谱更加抽象的信息, 然后经过两个全连接层和以Relu层去训练各个光谱的权重信息.
> 公式如下:
> 
> ![](https://image.chiullian.cn/img/202411061431089.png)
 

> 小创新:残差连接结构：SAT Net在多层编码器之间使用了多层残差连接，以减少信息在多层堆叠中的丢失并提高模型的收敛速度