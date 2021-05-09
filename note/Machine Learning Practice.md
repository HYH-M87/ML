# Machine Learning Practice

![image-20210122154111459](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210122154111459.png)

## 概念

![image-20210122161644275](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210122161644275.png)

 ###### **数据集**  

数据整体

###### 样本  

每行数据

###### 特征  

每行数据中的某个值

###### 特征向量

###### 特征空间

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210122161717696.png" alt="image-20210122161717696" style="zoom:50%;" />



 ### 任务

##### 分类任务

1.二分类

2.多分类

3.多标签分类

##### 回归任务

结果是一连串数值。 例如 ：房屋价格

##### 监督学习

K近邻

线性回归和多项式回归

逻辑回归

SVM

决策树和随机森林

##### 非监督学习

辅助监督学习   对数据进行降维

1.特征提取

2.特征压缩：PCA

3.数据可视化

4.异常检测

##### 半监督学习

一部分有标记，一部分没有

先使用无监督学习进行分类，再使用监督学习进行训练

##### 增强学习

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210122163349654.png" alt="image-20210122163349654" style="zoom:50%;" />



##### 批量学习

简单

无法长时间适应环境变化，只能定时进行批量学习

##### 在线学习

不断使用新的数据进行学习，新的数据可能产生不好的变化，只能加强对数据的监控 

##### 参数学习

一旦学习到参数，就不再需要原有的数据集

##### 非参数学习

 

## KNN

**优点：**

简单好用 

**缺点：**

效率低下  ：可使用数据结构进行小幅度优化

高度数据相关 ：对数据较敏感

对预测结果不具有可解释性 ： 

维数灾难 ：随着维度增加，”看似相近“的两个点距离越来越大，KNN高度依赖距离

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210123174541528.png" alt="image-20210123174541528" style="zoom:50%;" />

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210123174646331.png" alt="image-20210123174646331" style="zoom:50%;" />

## Liner Regression

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210123201435507.png" alt="image-20210123201435507" style="zoom:50%;" />![image-20210124095718871](C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210124095718871.png)

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210124095718871.png" alt="image-20210124095718871" style="zoom: 50%;" />

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210124095829616.png" alt="image-20210124095829616" style="zoom: 33%;" />

## Gradient Descent

<img src="C:\Users\hyh\AppData\Roaming\Typora\typora-user-images\image-20210124100405993.png" alt="image-20210124100405993" style="zoom:50%;" />