# SpectralFormer: Rethinking Hyperspectral Image Classification With Transformers

该论文总结的一些常用的高光谱分类的模型:

![](https://image.chiullian.cn/img/202410301834050.png)

附: 该论文的分类方式, 用相邻的像素信息和通道进行分类(不像之前是基于 patch 块)


结构图如下:

![](https://image.chiullian.cn/img/202410302021268.png)


优化点:
> 创新点一(GSE 多组嵌入):
> 
> 传统的注意力也就是正常的高光谱嵌入的时候不是一个一个像素点或者一个小patch块单独嵌入,这样会损失相邻光谱的关联性,所以这里采用了相邻分组的方式进行嵌入,即将一个像素的连续的波段或者一个连续的块进行嵌入
> 这样就能更好的结合相邻光谱维度之间的信息
> 
> 创新点二 (CAF 软残差):
> 
> 采用(短)软残差的方式, 增强各层级直线的联结, 减少信息的确实和梯度消失, (但是短残差的记忆力还是有限的, 长残差的记忆力长但是, 深层和浅层的特征差距太大)
> 残差的相加方式: ![](https://image.chiullian.cn/img/202410302141149.png)

![](https://image.chiullian.cn/img/202410302142772.png)
上面(可以用一个像素点或者用一个patch块 俺光谱维用长度为k的滑动窗口进行嵌入[论文提出了两种方式])


