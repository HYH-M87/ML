# Machine Learning



## 梯度下降

![image-20201011195558926](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011195558926.png)

### 学习率

要取的不大不小，过大收敛不了，过小收敛太慢

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201008234047329.png" alt="image-20201008234047329" style="zoom:67%;" />

### 局部最优解

![image-20201011200504371](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011200504371.png)

![image-20201011200513474](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011200513474.png)

### 特征缩放

使多个特征值都处在一个相近的范围，使梯度下降算法可以更快的收敛。



目标值 = （原值 - 均值）/  （max(原值) - min(原值)）

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201008233445552.png" alt="image-20201008233445552" style="zoom:67%;" />

一般缩放至-1~1

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201008233636899.png" alt="image-20201008233636899" style="zoom:60%;" />

### 多项式回归

创建新的特征，使模型简化。

将三次模型简化为线性模型：

 <img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201008234216563.png" alt="image-20201008234216563"  />新的特征值所处范围差距较大，需使用特征缩放

## 正规方程

不用进行迭代计算，一次性得到最佳解。 

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201008235335148.png" alt="image-20201008235335148" style="zoom: 80%;" />

1.创建特征矩阵，结果矩阵

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201008235354802.png" alt="image-20201008235354802" style="zoom:80%;" />

2.θ计算式

![image-20201008235441514](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201008235441514.png)

### 不可逆

当样本数小于特征数时 X * X.T 不可逆

### 正规方程和梯度下降的优缺点：

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201008235951124.png" alt="image-20201008235951124" style="zoom:80%;" />

特征量超过10000考虑梯度下降。正规方程不适用于复杂的算法



### 向量化

将循环运算，变为矩阵运算，一步到位

![image-20201010082719692](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010082719692.png)



### logistic 回归算法

分类算法。

输出值在0~1之间

![image-20201010083859757](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010083859757.png)

代价函数

对于0~1分类

![image-20201010093017910](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010093017910.png)

![image-20201010093046364](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010093046364.png)

将上述两个函数合并

![image-20201010093810391](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010093810391.png)

梯度下降，求θ

![image-20201010101049669](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010101049669.png)

正则化：

![image-20201011154831178](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011154831178.png)

高级算法：Conjugate gradient，BFGS，L-BFGS

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010101654222.png" alt="image-20201010101654222" style="zoom:67%;" />





### 决策边界  decision boundary

关于 logistics 回归算法的假设函数

对数据集进行分类的界限，决定于所取参数θ矩阵  

线性

![image-20201010085005865](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010085005865.png)

非线性

![image-20201010085434296](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010085434296.png)

### 多类别分类

拆分成多个二分类问题

![image-20201010103543017](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010103543017.png)

### 过度拟合

从左至右分别为：欠拟合，正确拟合，过拟合

![image-20201010104020155](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010104020155.png)

![image-20201010104344811](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010104344811.png)

过多的变量和过少的训练数据会导致过拟合

解决方法：

![image-20201010104847626](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201010104847626.png)

在参数多且不知道哪个参数为相关性较差的时候，可以将所有参数值缩小。在原来的代价函数后增加一项，正则项

![image-20201011080622543](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011080622543.png)

> 正则化模型：
>
> ![image-20201011080459770](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011080459770.png)

带有正则化的正规方程，当λ大于0，不需要考虑可不可逆的问题，正则化矩阵为(n+1)*(n+1)

![image-20201011083644893](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011083644893.png)



### 非线性假设（神经网络）

#### 神经元

![image-20201011162848770](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011162848770.png)

#### 激活函数

![image-20201011162815565](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011162815565.png)

#### 神经网络—前向传播

输入层—>隐藏层—>输出层  为了计算出每层的激活值

![image-20201011163927387](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011163927387.png)

![image-20201011164810966](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011164810966.png)

向量化

![image-20201011194213289](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011194213289.png)

> 例子 用神经网络实现 同或逻辑
>
> 1.首先实现AND
>
> ![image-20201017095623070](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201017095623070.png)
>
> 2.OR
>
> ![image-20201017095715792](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201017095715792.png)
>
> 3.NOT
>
> ![image-20201017095820037](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201017095820037.png)
>
> 3.XNOR
>
> ![image-20201011212822548](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011212822548.png)

### 多类别分类

 ![image-20201011214444288](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201011214444288.png)

### 代价函数：sigmode

![image-20201013080147807](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201013080147807.png)

### 反向传播法

![image-20201013081224169](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201013081224169.png)

![image-20201013122517250](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201013122517250.png)

### 梯度检测

![image-20201021200606022](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201021200606022.png)

总结：

1.隐藏层默认一层，多层时，每层的隐藏单元要一致

2.隐藏单元越多越好

3.步骤：

1）构建神经网络，权重初始化为不同的接近0的值

2）前向传播，计算出输出值

3）计算代价函数

4）反向传播

5）梯度检测

6）优化算法，更新权重

### 模型评估

73分割  data set

使用7成的数据集，训练得到参数

再用3成的数据集，去计算误差

### **模型选择**

再使用37分（训练集和测试集）得到的结果可能只是对当前数据集的最优解，无法保证它的泛化能力

可以通过增加一个交叉验证集来解决这个问题，比如622分（训练集，交叉验证集，测试集），通过交叉验证集选出最优的模型，再经过测试集验证

### 方差与偏差

当训练误差约等于交叉验证误差且都较高时，为欠拟合，高偏差

当训练误差低，交叉验证误差高时，为过拟合，高方差

![image-20201107135616964](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107135616964.png)

**正则化与方差偏差**

λ称为惩罚系数，其值越大正则化程度越大

![image-20201107161014385](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107161014385.png)

自动选择λ，使用之前的方法遍历各数值

![image-20201107161850659](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107161850659.png)

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107162155994.png" alt="image-20201107162155994" style="zoom:80%;" />

λ小->高方差->过拟合

λ大->高偏差->欠拟合

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107162553512.png" alt="image-20201107162553512" style="zoom:80%;" />

### 学习曲线

1.高偏差学习曲线

数据集增大，训练误差和交叉验证误差逐渐接近，且都较高

用更多的数据，对减小误差没多大帮助

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107163735849.png" alt="image-20201107163735849" style="zoom:80%;" />

2.高方差学习曲线

可以发现，对高方差，使用更多的数据来减少误差是有帮助的

![image-20201107164630464](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107164630464.png)

### 机器学习诊断法

当发现结果并不理想，首先需要判断是高方差还是高偏差，可能有时候比起花更多时间去收集数据，不如对特征进行筛选或增加。



![gao](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201025155742593.png)

1.获得更多数据 -> 高方差

2.减少特征值 -> 高方差

3.增加特征值 -> 高偏差

4.增加多项式特征 -> 高偏差

5.减少 λ -> 高偏差

6.增大 λ -> 高方差

### 机器学习系统设计

1.不要想着一次性设计出一个最佳的机器学习系统，先做一个能完成任务的简单系统

2.绘制学习曲线，检验误差，看看如何优化，把时间花在对的地方

3.误差分析，找出产生误差的数据，观察有无相同特征点

### 不对称性分类的误差评估（偏斜类）

对于之前的癌症例子，得癌症为1，否则为0

假设，使用a算法通过测试集得到的准确率为99%，错误率为1%

但是在一个群体中其实只有0.5%的人有癌症，此时采样以下这个算法来进行预估（不管输入是多少，都输出1，也就是都得癌症）

![image-20201107215213965](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107215213965.png)

对于上述群体来说只有0.5%的错误率，看似比a算法更好，但其实是不对的

这就是偏斜类问题

**查准率(p)和召回率(r)**

1.真阳性 ； 预测值=1 & 实际值=1

2.假阳性 ；预测值=1& 实际值=0

3.真阴性 ； 预测值=0 & 实际值=0

4.假阴性 ；预测值=0 & 实际值=1

![image-20201107220111850](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107220111850.png)

查准率= 真阳性/（真阳性+假阳性）

> 保证预测结果为1，实际也为1

召回率= 真阳性/（真阳性+假阴性）

> 保证预测结果为0，实际结果不会为1

（高查准率+低召回率 —> 查到你有病，你基本有病，查到你没病，你不一定没病）

（低查准率+高召回率 —> 查到你有病，你不一定有病，查到你没病，你基本没病）

**F1 score** 权衡查准率和召回率，越大越好，而不是用平均值

![image-20201107221433177](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107221433177.png)

![image-20201107221359462](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201107221359462.png)

### SVM

从logistic 到SVM

![image-20201108082418555](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108082418555.png)

经过中间的变换法制，得到SVM的cost	function

![image-20201108083049289](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108083049289.png)

![image-20201108083126465](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108083126465.png)

SVM——大间距分类器

比起绿色和粉色的决策边界，SVM会选择间距较大的黑色边界

![image-20201108084031593](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108084031593.png)

当C设定的很大时，SVM会由于数据异常的缘故产生很大的变化，变成粉色的边界，当C很小时不会

![image-20201108084234764](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108084234764.png)

数学原理：P72

### 使用SVM构造非线性分类器

**核函数**

高斯核函数,相似函数-》相似度



随机选取三个点，l_1,l_2,l_3

![image-20201108091510731](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108091510731.png)

![image-20201108091803251](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108091803251.png)

可以看到，当高斯核函数的值约等于1时，表示该点与指定点接近，约等于0时表示不接近

σ参数的影响

![image-20201108092325934](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108092325934.png)

σ越小，下降越快

应用案例：当点接近于l1,l2时输出1，否则输出0

![image-20201108092643897](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108092643897.png)

选点，将数据集的点作为定位点

![image-20201108093707905](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108093707905.png)

将各个F组成向量，用于SVM

![image-20201108094219793](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108094219793.png)

![image-20201108100300375](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108100300375.png)

使用软件库，需要自行选择参数：

![image-20201108100741146](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108100741146.png)

![image-20201108103723144](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201108103723144.png)

### 无监督学习

聚类算法-》对一些无标签的数据进行分组

K-Means算法

1.簇分配（cluster assignment）

2.移动聚类中心

![image-20201114142931679](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201114142931679.png)

以距离  红×（聚类中心）  和蓝× 的距离不同进行分类

![image-20201114143018780](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201114143018780.png)

将红色聚类中心移动到所有红点的均值处，蓝也是

![image-20201114143307667](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201114143307667.png)

不断重复

结果：

![image-20201114143344629](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201114143344629.png)

![image-20201114143414508](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201114143414508.png)

伪代码：

![image-20201114144355266](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201114144355266.png)

**优化函数**

![image-20201129003710594](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129003710594.png)

第一步，先保持聚类中心不变，为所有数据选择一个据类中心，使得cost function 值最小

第二步，改变聚类中心则比，使得cost function 值最小

![image-20201129004131140](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129004131140.png)

**随机初始化，避免局部最优**

聚类中心数应小于样本数据

随机选择样本作为聚类中心

![image-20201129004952182](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129004952182.png)

局部最优

![image-20201129005124292](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129005124292.png)

重复初始化，用于聚类中心较少的情况 K=2~10

重复 50~1000次

![image-20201129005336269](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129005336269.png)

**选择聚类数量**

肘部法则

![image-20201129005942244](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129005942244.png)

如左图，K=3时为最佳聚类数量，然而大多情况都如右图所示

根据实际情况制定，比如定制T恤，S、M、L三个大学，则分三个聚类中心



## **主成成分分析算法**（PCA）  

1.压缩数据，加速其他算法的运行

2.可视化2D/3D数据

**降维**，将特征减少

数据压缩，使算法运行的更快

![image-20201129011105292](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129011105292.png)

z1 为该点在绿色直线上的长度

![image-20201129011410418](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129011410418.png)

使图中蓝色线段的平方和2最小

![image-20201129013441435](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129013441435.png)

将高维降为低维，所以找投影最小的，减小与原来高维特征的差距，使得特征误差最小，找到一个降维后的最佳cost function

![image-20201129013200875](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129013200875.png)

与线性回归的区别

![image-20201129013335346](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129013335346.png)

线性回归是关于y轴的差值，PCA是关于映射的差值



**奇异值分解**

 **协方差**

![image-20201129015915945](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129015915945.png)

![image-20201129022410986](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20201129022410986.png)

主成分数量选择，使用S矩阵（对角线矩阵）

![image-20210121125859614](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121125859614.png)

## 异常检测

![image-20210121133406761](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121133406761.png)

**高斯分布**

![image-20210121135649100](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121135649100.png)

μ是数据的平均值   σ是平均方差



异常检测算法：P(x) 为各特征量的高斯概率相乘

![image-20210121141054510](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121141054510.png)

![image-20210121141858045](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121141858045.png)



**评估异常检测算法**

1.分训练集 测试集 交叉验证集

2.通过训练集得出p(x)

3.计算f1_score  查准率  召回率

4.选择ε使  f1_score 最大



##### 异常检测VS监督学习

对于负样本很少使使用 异常检测算法

负样本多时可以使用监督学习算法

![image-20210121154900827](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121154900827.png)



若数据不符合高斯分布可以试一下log，不一定是log 也可以是平方根等等…………

将Xnew = log(x) 再运用到异常检测中

[P93](https://www.bilibili.com/video/BV164411b7dx?p=93)

##### 多元高斯分布

![image-20210121161634379](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121161634379.png)

据左图可知CPU和内存占用成正比，当出现绿色叉叉这种异常情况，其在两个高斯分布的值都较高，一元高斯分布 

无法区分

> ![image-20210121162847235](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121162847235.png)

> ![image-20210121162907649](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121162907649.png)

> ![image-20210121163212105](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121163212105.png)



**多元高斯分布 异常检测算法**

![image-20210121164013195](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121164013195.png)

使用多元高斯分布后的模型

![image-20210121164214435](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121164214435.png)



相比于一元高斯   多元高斯计算代价高昂，数据数必须大于特征量

### 推荐系统

假设模型

![image-20210121193334987](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121193334987.png)



基于内容的推荐算法

![image-20210121193815483](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121193815483.png)

x1电影为浪漫片的程度  x2为动作片的程度  目的是将电影用x1  x2来表示

![image-20210121194119735](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121194119735.png)

对每个人也有一个向量去表示

最后得出Alice对电影 《Cute puppies of love》的评分预测是4.95

![image-20210121195126463](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121195126463.png)

通过对该用户其他电影的评价   通过线性回归得到θ(j)的最小值  再用θ(j)去对其他电影进行预测

### 协同过滤

日常中我们无法确知每部电影的特征

特征学习  自行学习所要使用的特征    通过调查用户的每种电影的喜好程度

![image-20210121201634000](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121201634000.png)

可以将x和θ一起最小化

![image-20210121203819419](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121203819419.png)

向量化  推荐同类产品![image-20210121221729538](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121221729538.png)

当两部电影 |x(i)  -  x(j)|足够小 可以说i和j相似

#### 均值归一化

当一个人没给任何电影打分时
![image-20210121223639307](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121223639307.png)

θ(5) 为[ 0 ; 0] 通过协同过滤算法得出他对所有的电影评分都为0

对该用户的预测取电影评分的均值

### 处理大数据集

   #### 随机梯度下降  

用于样本巨多时，针对单个样本。每处理完一个样本后就更新参数

 #### mini-batch  gradient descent

针对b个样本，分批处理b个样本 ，每处理b个样本后就更新参数

**随迭代次数的增加，减小学习率**

### 在线学习

对于连续的数据流 采用随机梯度下降 

当数据流多时   实时更新数据

当数据流小时   保存为数据集

MapReduce

### 多机/多核运行

前提是该算法可以表示为 对训练集的求和

![image-20210121235130238](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210121235130238.png)

### Photo OCR  (photo optical character recognition)  照片光学字符识别

![image-20210122001841718](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210122001841718.png)

**滑动窗口分类器**

### 人工数据合成

1.引用不同字库

2.图像拉伸

3.音频添加噪声

人工制造数据需要有针对性，代表性

众包：Amazon Mechanical Turk