# 前言
本项目实现了常用的机器学习算法。

# 机器学习算法索引 
### 感知机算法 
#### 实现文件与测试文件 
linear_model/perceptron.py \
linear_model/tests/test_perceptron.py
##### 测试说明
数据集：mnist \
训练集数量：60000 \
测试集数量：10000 

参数设定 \
学习速率为0.0001 \
迭代次数为50
##### 测试结果
正确率: 78%（二分类） \
运行时间: 91s

### K近邻算法
#### 实现文件与测试文件
neighbors/classification.py \
neighbors/tests/test_classification.py
##### 测试说明
数据集：mnist \
训练集数量：60000 \
测试集数量：200 

参数设定1 \
邻近k数量为5 \
距离度量：欧式距离
##### 测试结果1
正确率: 98.5% \
运行时间: 167s

参数设定2 \
邻近k数量为5 \
距离度量：曼哈顿距离
##### 测试结果2
正确率: 97.5% \
运行时间: 154s

### 朴素贝叶斯算法
#### 实现文件与测试文件
linear_model/bayes.py \
linear_model/tests/test_naive_bayes.py
##### 测试说明
数据集：mnist \
训练集数量：60000 \
测试集数量：10000 

##### 测试结果
正确率: 83.3% \
运行时间: 431s

### 决策树算法
#### 实现文件与测试文件
tree/tree.py \
tree/tests/test_tree.py
##### 测试说明
数据集：mnist \
训练集数量：60000 \
测试集数量：10000 

##### 测试结果
正确率:85.89%  \
运行时间:385s

### 逻辑回归算法
#### 实现文件与测试文件
linear_model/logistic.py \
linear_model/tests/test_logistic.py
##### 测试说明
数据集：mnist \
训练集数量：60000 \
测试集数量：10000 

##### 测试结果
正确率: 90.2% \
运行时间: 291s 

### 支持向量机算法
#### 实现文件与测试文件
svm/svm.py \
svm/tests/test_svm.py
##### 测试说明
数据集：mnist \
训练集数量：1000 \
测试集数量：100 

##### 测试结果
正确率: 92% \
运行时间: 141s 


### AdaBoost算法
#### 实现文件与测试文件
ensemble/adaboost.py \
ensemble/tests/test_adaboost.py
##### 测试说明
数据集：mnist \
训练集数量：1000 \
测试集数量：100 

##### 测试结果
正确率: 97% \
运行时间: 792s

### EM算法
#### 实现文件与测试文件
em/em.py \
em/tests/test_em.py
##### 测试说明
生成混合高斯分布数据，输出高斯混合模型参数

### HMM模型
#### 实现文件与测试文件
hmm/hmm.py \
hmm/tests/test_hmm.py
##### 测试说明
1.给定隐马尔科夫模型以及观测序列，输出观测序列概率。即观测序列模型的前向算法。 \
2.给定隐马尔科夫模型以及观测序列，输出出现概率最大的路径，即维比特算法。
