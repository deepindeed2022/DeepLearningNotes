# 基本概念
@(Deep-Learning)[detection]

### 深度学习与传统机器学习的区别与联系

深度学习相比较传统机器学习，解决了一个关键问题：数据的抽象表示问题
https://zhuanlan.zhihu.com/p/26769864

### Embedding Layer的作用

https://gshtime.github.io/2018/06/01/tensorflow-embedding-lookup-sparse/

### 卷积神经网络的旋转不变性

> Q1: 卷积神经网络提取图像特征时具有旋转不变性吗？

A1: 其实CNN是具有一定的旋转不变性，其实之所以有旋转不变性是因为maxpooling采样层得来的，当我们正面拍一些照片的时候，也许我们学习得到的一些地方的activation会比较大，当我们略微的转过一点角度之后，由于maxpooling的存在，会使得我们的maxpooling依然在那个地方取到最大值，所以有一定的旋转不变性，但是，对于角度很大的来说，maxpooling可能也会失效，所以需要一定的data augmentation，或者你也可以提高maxpooling的窗口，来提高一定的旋转不变性。

> 拓展：**空间变换器（spatial transformer network）**
> 普通的CNN能够显示的学习平移不变性，以及隐式的学习旋转不变性，但attention model 告诉我们，与其让网络隐式的学习到某种能力，不如为网络设计一个显式的处理模块，专门处理以上的各种变换。因此，DeepMind就设计了Spatial Transformer Layer，简称STL来完成这样的功能。
> 关于平移不变性 ，对于CNN来说，如果移动一张图片中的物体，那应该是不太一样的。假设物体在图像的左上角，我们做卷积，采样都不会改变特征的位置，糟糕的事情在我们把特征平滑后后接入了全连接层，而全连接层本身并不具备平移不变性的特征。但是 CNN 有一个采样层，假设某个物体移动了很小的范围，经过采样后，它的输出可能和没有移动的时候是一样的，这是 CNN 可以有小范围的平移不变性 的原因。

一个空间变换器的运作机制可以分为三个部分，如下图所示：1） 本地网络（Localisation Network）；2）网格生成器( Grid Genator)；3）采样器（Sampler）。
![STN](./images/STN.png)

- 论文出处：https://arxiv.org/abs/1506.02025
- [Spatial Transformer Network paperblog](https://www.cnblogs.com/liaohuiqiang/p/9226335.html)
-  https://blog.csdn.net/qq_39422642/article/details/78870629

## 数据增强 Data Augmentation

### 数据增强的目的与作用
一个卷积神经网络有一个称作不变性的性质，即使卷积神经网络被放在不同方向上，它也能进行对象分类。
更具体的说，卷积神经网络对平移、视角、尺寸或照度（或以上组合）保持不变性。
这就是数据增强的本质前提。在现实世界中，我们可能会有一组在有限的条件下拍摄的图像 。但是，我们的目标应用可能是在多变的环境中，
例如，不同的方向、位置、比例、亮度等。我们通过使用经综合修改过的数据来训练神经网络，以应对这些情形。

1. 补充数据样本不足
2. 减少网络的过拟合现象，通过对训练图片进行变换可以得到泛化能力更强的网络，更好的适应应用场景。

### 基本方法
* 旋转 | 反射变换(Rotation/reflection): 随机旋转图像一定角度; 改变图像内容的朝向;
* 翻转变换(flip): 沿着水平或者垂直方向翻转图像;
* 缩放变换(zoom): 按照一定的比例放大或者缩小图像;
* 平移变换(shift): 在图像平面上对图像以一定方式进行平移;
* 可以采用随机或人为定义的方式指定平移范围和平移步长, 沿水平或竖直方向进行平移. 改变图像内容的位置;
* 尺度变换(scale): 对图像按照指定的尺度因子, 进行放大或缩小; 或者参照SIFT特征提取思想, 利用指定的尺度因子对图像滤波构造尺度空间. 改变图像内容的大小或模糊程度;
* 对比度变换(contrast): 在图像的HSV颜色空间，改变饱和度S和V亮度分量，保持色调H不变. 对每个像素的S和V分量进行指数运算(指数因子在0.25到4之间), 增加光照变化;
* 噪声扰动(noise): 对图像的每个像素RGB进行随机扰动, 常用的噪声模式是椒盐噪声和高斯噪声;
* 颜色变化：在图像通道上添加随机扰动。
PCA Jittering
### caffe中的数据增强
只发现mirror、scale、crop三种

### SSD中的数据增强
http://cwlseu.github.io/2017/04/05/SSD-Data-Augmentation/

### 海康威视MSCOCO比赛中的数据增强
第一，对颜色的数据增强，包括色彩的饱和度、亮度和对比度等方面，主要从Facebook的代码里改过来的。
第二，PCA Jittering，最早是由Alex在他2012年赢得ImageNet竞赛的那篇NIPS中提出来的. 我们首先按照RGB三个颜色通道计算了均值和标准差，对网络的输入数据进行规范化，随后我们在整个训练集上计算了协方差矩阵，进行特征分解，得到特征向量和特征值，用来做PCA Jittering。
第三，在图像进行裁剪和缩放的时候，我们采用了随机的图像差值方式。
第四， Crop Sampling，就是怎么从原始图像中进行缩放裁剪获得网络的输入。比较常用的有2种方法：
- 是使用Scale Jittering，VGG和ResNet模型的训练都用了这种方法。
- 二是尺度和长宽比增强变换，最早是Google提出来训练他们的Inception网络的。我们对其进行了改进，提出Supervised Data Augmentation方法。

### 可参考链接
- [输入图像随机选择一块区域涂黑，《Random Erasing Data Augmentation》](https://arxiv.org/pdf/1511.05635.pdf)
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501v1)
- [海康威视研究院ImageNet2016竞赛经验分享](https://zhuanlan.zhihu.com/p/23249000)
- https://github.com/kevinlin311tw/caffe-augmentation 
- https://github.com/codebox/image_augmentor
- https://github.com/aleju/imgaug.git
- [The art of Data Augmentation](http://lib.stat.cmu.edu/~brian/905-2009/all-papers/01-jcgs-art.pdf)
- [Augmentation for small object detection](https://arxiv.org/abs/1902.07296)