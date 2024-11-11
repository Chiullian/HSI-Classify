# Multiscale Dynamic Graph Convolutional Network  for Hyperspectral Image Classification

#### superpixels（超像素）
> 貌似大部分基于GCN的网络都会使用SLIC先去进行
> 
> 超像素的概念: 超像素是一系列像素的集合，这些像素具有类似的颜色、纹理等特征，距离也比较近。用超像素对一张图片进行分割的结果见下图，其中每个白色线条区域内的像素集合就是一个超像素。需要注意的是，超像素很可能把同一个物体的不同部分分成多个超像素。

![](https://image.chiullian.cn/img/202411112101522.jpg)

> 在一定程度上能`减少过拟合`或者 `减少噪音`的影响

变成一块一块的后再使用GCN, 极大的降低了复杂度

简单说一下GCN的优势:
> 可以处理不规则的形状 这对分类来说很重要, !!!!!

劣势:
> 计算的复杂度过高, 优化出来你就是王!!!!

#### Graph attention networks (图注意机制)

> 高端的模型往往只需要更简单的架构 (


![](https://image.chiullian.cn/img/202411112141343.png)


如上图: a23, a34 对三号节点的权重, 不一定是相同的, 这个“重要性”可以进行量化，更可以通过网络训练得出。这个“重要性”，在文中叫attention，可以通过训练得到。这便是GAT的核心创新点了
> 要注意的是两个点的注意关系没有对称性
> ![](https://image.chiullian.cn/img/202411112148354.png)


乘以共享矩阵W也就是抽取i节点向量的特征 (常规操作共享一个参数矩阵, 进入同一个特征空间, 也就是相当于对一个节点进行特征抽取)
![](https://image.chiullian.cn/img/202411112216109.png)


图注意力的总体模型:

![](https://image.chiullian.cn/img/202411112242773.png)
