# SLIC 超像素分割算法 

SLIC超像素分割详解：
> 具有相似纹理、颜色、亮度等特征的相邻像素构成的有一定视觉意义的不规则像素块。,用少量的超像素代替大量的像素来表达图片特征,很大程度上降低了图像后处理的复杂度，所以通常作为分割算法的预处理步骤。

算法的思想：
> 将彩色图像转化为CIELAB颜色空间和XY坐标下的5维特征向量，然后对5维特征向量构造距离度量标准，对图像像素进行局部聚类的过程。

下面是改算法功效图：

![](https://image.chiullian.cn/img/202411131747713.png)

SLIC主要优点如下：
* 1）生成的超像素如同细胞一般紧凑整齐，邻域特征比较容易表达。这样基于像素的方法可以比较容易的改造为基于超像素的方法。
* 2）不仅可以分割彩色图，也可以兼容分割灰度图。
* 3）需要设置的参数非常少，默认情况下只需要设置一个预分割的超像素的数量。
* 4）相比其他的超像素分割方法，SLIC在运行速度、生成超像素的紧凑度、轮廓保持方面都比较理想。

SLIC的步骤：
* 1.撒种子。根据图像大小和定义的超像素数目K，将k个超像素中心均匀分布到图像的像素点上。假设图片总共有 N 个像素点，预分割为 K 个相同尺寸的超像素，那么每个超像素的大小为N/ K ，则相邻种子点的距离（步长）近似为S=sqrt(N/K)。
* 2.调整种子位置。以k为中心的nn（一般n取3，故为33）范围内，将超像素中心移动到这9个点中梯度最小的点上。目的是为了避免超像素点落入噪点或边界处。
* 3.初始化数据。在每个种子点周围的邻域内为每个像素点分配类标签lable（即该像素点是属于哪个聚类中心的）。做法：取一个数组label，保存每个像素点是属于哪个超像素。dis数据保存像素点到它属于的那个超像素中心之间的距离。
* 4.更新数据。对每一个超像素中心x，在其2s*2s范围内搜索；如果点到超像素中心x的距离(五维)小于点到它原来属于的超像素中心的距离，说明这个点属于超像素x。（新距离小于旧距离，则点属于新范围。更新dis，更新label）----》每个像素必须与所有聚类中心比较，通过引入距离测量D值。**【期望的超像素尺寸为SS，但是搜索的范围是2S2S】【由于每个像素点都会被多个种子点搜索到，所以每个像素点都会有一个与周围种子点的距离，取最小值对应的种子点作为该像素点的聚类中心。】**
* 5.计算。对每一个超像素中心，重新计算它的位置（属于该超像素的所有像素的位置中心）以及lab值。
* 6.迭代4、5步骤。【理论上上述步骤不断迭代直到误差收敛（可以理解为每个像素点聚类中心不再发生变化为止），实践发现10次迭代对绝大部分图片都可以得到较理想效果，所以一般迭代次数取10。】
后处理步骤：