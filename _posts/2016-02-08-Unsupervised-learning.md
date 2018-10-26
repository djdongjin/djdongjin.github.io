---
layout: post
title: "Unsupervised Learning"
categories: [Machine Learning]
---

##聚类/clustering

Supervised: 给定训练数据x以及对应的标签y进行训练
Unsupervised: 仅给定训练数据x进行训练

### K均值算法/K-means

**输入**
K：聚类个数
训练集：$\{x^{(1)}, ..., x^{(m)}\}$

**步骤**
```
Randomly initialize K cluster centroids (mu_1, ..., mu_K) in R^n
Repeat{
    for i =1:m
        c^i := index of cluster centroid closet to $x^(i)$
    for k = 1:K
        mu_k := average of points assigned to cluster k
}
```

**优化目标/cost function**
首先规定一下符号表示方法：
$c^{(i)}$ = $x^{(i)}$ 目前所处的聚类序号(1~K)
$\mu_k$ = 第k个聚类的centroid k
$\mu_{c^{(i)}}$ = $x^{(i)}$所处聚类的centroid 
K-means的cost function是各个点到对应聚类中心的距离，即我们的优化目标，公式如下：
$$
J\left(c^{(1)},...,c^{(m)}, \mu_1,...,\mu_K\right) = \frac{1}{m}\sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2
$$
我们可以通过K-means算法中循环部分的第一个内循环，将每个点划分到最近的聚类中心所在的类中，来最小化cost function

**随机初始化/Random initialization**
算法的第一步是随机化聚类中心，合理的方法是随机选择K个训练样本作为初始聚类中心，随着迭代来动态的调整中心，使得算法收敛；该方法可能导致最后陷入一个比较差的结果，比如初始中心距离过近，使得它们不具有代表性，因此会生成比较差的聚类，解决方法是多次迭代K-means产生多个聚类划分，选择cost function最小的，算法如下:

```
for i = 1:100{
    randomly initialize K-means
    run K-means, generate c^(1),..,c^(m), mu_1,..,mu_K$
    compute cost function J
}
pick clustering that gave lowest cost $J$
```

**确定聚类数量K**
一个理想化的方法是肘部法则(Elbow method)：让K从1开始逐个增加，求每个K值划分最后得到的$J$，会得到如下图左。但是一般来说，该过程得到的图是如右图所示，因此该方法实际当中使用并不多。

![](/assets/2016-02-08-Elbow-method.jpg)

相反，我们应该根据聚类之后所做决策的情况，来确定聚类数量K。比如确定一个衣服的生产尺码，有两个聚类，一个是S/M/L，另一个是XS/S/M/L/XL，我们可以分别在两个市场进行这两种聚类，然后看它们各自的表现，从而进行对比。

## 降维/Dimensionality Reduction
一般来说，降维有两个目的：数据压缩、数据可视化
数据压缩的本质是将数据中具有一定相关性的特征进行压缩，使得数据维度变低；而数据可视化是由于高维度的数据很难进行可视化，因此我们先对数据进行降维，比如从5维降到2维，然后对2维数据进行可视化。

### 主成分分析/PCA
主成分分析(Principal Component Analysis, PCA)是最常用的降维方法，工作原理如下：假如将数据从n维降到k维，首先找到k个n维向量，由它们形成一个k维的面；然后将所有数据向该面进行投影，最小化每个点到该面的距离，也就是投影误差。距离来说，将2维降到1维，就是找到一个2维的向量，构成一条直线，然后每个点向该直线投影，通过更改直线来最小化每个点到该直线的距离。如果是3维降到2维，就是通过两个3维向量构成一个平面，最小化各个点到面的距离。
![](/assets/2016-02-08-PCA.jpg)

**PCA ！= Linear regression**
可以发现，PCA和线性回归具有一定相似性，比如它们均是通过构造一条直线来拟合数据，但是它们并不相同，如下图所示(此处PCA以2维降到1维为例)：

![](/assets/2016-02-08-PCA_LS.jpg)

线性回归我们是通过大量的数据(x, y)，来拟合一条直线y=ax+b，通过该直线来预测给定输入x时的输出y'，其误差为预测值-实际值，y'-y，也就是图中和y轴平行的小线段；而PCA，我们是通过一条直线，来最小化数据点和其在该直线上的投影的距离，假设原数据是(x1, x2)，投影为(x1', x2')，其误差为这两个点之间的距离，也就是图中和直线垂直的小线段。

**PCA algorithm**
首先我们需要进行特征放缩(feature scaling)以及均值归一化(mean normalization)

$$\begin{aligned}
\mu_j=\frac{1}{m}\sum_{i=1}^mx_j^{(i)} \\
s_j = \max x_j - \min x_j\\
x_j^{(i)} = \frac{x_j^{(i)} - \mu_j}{s_j}
\end{aligned}$$

PCA算法如下(n-D --> k-D):

1. mean normalization, and optionally feature scaling
2. compute "covariance matrix"(协方差矩阵):
        $\Sigma = \frac{1}{m}\sum_{i=1}^m x^{(i)}x^{(i)T}$
3. compute "eigenvectors" of matrix $\Sigma$:
        [U, S, V] = svd(Sigma);
4. Ureduce = U(:, 1:k);
5. z = Ureduce' * x;

x是n\*1向量，因此步骤2结果为一个n\*n矩阵，也就是协方差矩阵，注意其表示符号是Sigma而不是求和符号；步骤三中得到的特征矩阵如下图所示，是一个n\*n矩阵，我们取前k列，得到了一个n\*k的矩阵Ureduce；最后通过Ureduce' \* x，得到了降维之后的数据z

![](/assets/2016-02-08-SVD.jpg)

**还原数据**
接下来我们考虑如何从降维后的数据z，得到原有数据x。考虑到我们通过该公式得到降维数据z:
$z = U_{reduce}^T * x$
同样我们可以通过如下公式得到原数据x的近似：
$x_{approx} = U_{reduce}^T * z$
并且，原始数据和还原得到的数据的差距就是原始数据和其投影点之间的projection error

**确定降维维数k**
average squared projection error: $\frac{1}{m}\sum ||x^{(i)} - x_{approx}^{(i)}||^2$
total variation in the data: $\frac{1}{m}\sum ||x^{(i)}||^2$
我们应该选择k，使得1所占2的比例最小，即：
$\min \frac{\frac{1}{m}\sum ||x^{(i)} - x_{approx}^{(i)}||^2}{\frac{1}{m}\sum ||x^{(i)}||^2}$
实际当中，我们的做法是选择k，使得上式小于0.01、0.05、0.1，称作把数据降到k维，同时保留了99%/95%的精度。
另外，由于上式的计算比较复杂，更简单的一种方法是计算对应特征值的比，如下图：

![](/assets/2016-02-08-dimension-K.jpg)

**PCA 应用**
**加速supervised learning**
有数据集$(x^{(i)}, y^{(i)})     1<=i<=m$，对其特征部分$x^{(i)}$应用PCA进行降维，得到新的数据集$(z^{(i)}, y^{(i)})     1<=i<=m$，由于特征减少，因此能够加速训练。
注意：在运行PCA算法时，我们只应该在训练集上进行，因为此时validation set和test set对模型仍不可见；然后在进行cross validation以及test时，先用在training set上得到的Ureduce对数据进行降维，然后再进行下一步操作。

**减少文件占用内存/硬盘、可视化数据**

**bad application**
**Prevent overfitting**
克服overfitting的一种方法是减少feature数量，但最好不要通过PCA来完成，虽然可能正常工作。最好还是应用regularization。
正如前面所说，PCA可应用于加速监督学习，因此下面的ML模型建立流程可能看起来很正常：

![](/assets/2016-02-08-pipeline.jpg)

但是，一般来说，在应用PCA之前，我们应该尽量使用原始数据x，当没有其他方法时，再运用PCA，使用降维数据z


