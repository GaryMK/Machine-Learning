# TensorFlow

## Environment

python: 3.7

tensorflow-gpu: 2.3.1

## Note

### activation functions

#### 神经层2~3层

对于隐藏层，可以尝试任意激励函数

**默认首选激励函数：**

卷积神经网络：relu

循环神经网络：relu or tanh

#### 多层神经网络

不能随意选择激励函数，涉及梯度爆炸、梯度消失问题

reference：[Neural Network](https://tensorflow.google.cn/api_docs/python/tf/nn)

### 结果可视化

```python
import matplotlib.pyplot as plt
// plt.ion() 绘制后继续绘制动态图
// plt.show()绘制后暂替
```

### 加速神经网络训练

**传统方法：**

W += -Learning rate * dx

**Stochastic Gradient Descent(SGD)**

**Momentum**

m = b1 * m - Learning rate * dx

W += m

**NAG**

**Adagrad**

v += dx^2

W += -Learning rate * dx / sqrt(v)

**Adadelta**

**Rmsprop**

Momentum + AdaGrad

v = b1 * v + (1-b1) * dx^2

W += - Learning rate * dx / sqrt(v)

**Adam**

m = b1 * m + (1-b1) * dx

v = b2 * v + (1 - b2) * dx ^ 2

W += - Learning rate * m / sqrt(v)

### Optimizer

MomentumOptimizer: 不仅仅考虑本步的learning rate，加载了上一步的learning rate趋势

RMSPropOptimizer: 谷歌所采用的用来优化alphago的

Momentum 与 Adam 比较常用

### Tensorboard（visualization tool）



```terminal
# 生成文件
# writer一定要在sess之后
writer = tf.summary.FileWriter("logs/", sess.graph)

# terminal启动
# 法一
tensorboard --logdir logs
# 法二

tensorboard --logdir=logs
```



```python
# SCALARS
tf.summary.scalar('loss', loss)
# HISTOGRAMS
tf.summary.histogram(layer_name + '/weights', Weights)
# merged
merged = tf.summary.merge_all()
```



### Classification 分类学习

### Convolutional Neural Network(CNN)

### Recurrent Neural Network(RNN)

#### 应用：

让RNN描述照片、写学术论文、写程序脚本或作曲

#### Long Short-Term Memory(LSTM) 长短期记忆

>梯度消失（梯度弥散）：
>
>在反向传播得到的误差的时候，在每一步都会乘以一个自己的参数，如果参数小于1，参数不断乘以误差，误差传到初始时间接近于0，对于初始时刻误差消失。
>
>梯度爆炸：参数大于1，不断累成，最好可能变成无穷大。

控制器：输入控制、输出控制和忘记控制

### 自编码(Autoencoder)

非监督学习

给特征属性降维，超越PCA

### Batch Normalization

添加在全连接和激励函数之间

### api更新

```python
#增加隐藏层
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)

# 计算loss
loss = tf.losses.mean_squared_error(tf_y, output)
```

### 可视化梯度下降/公式调参

```python
from mpl_toolkits.mplot3d import Axes3D

```

尝试使用不同的初始化的参数值，进行训练，总结

### 迁移学习(Transfer Learning)

多任务学习，强化学习中的Learning to Learn；

Google Neural Machine Translation

将VGG16猫与老虎的识别模型迁移应用于猫与老虎的体长预测

## Reference
[tensorflow2学习](http://blog.nodetopo.com/2019/12/20/morvan%e5%8d%9a%e5%ae%a2-tensorflow2%e5%ad%a6%e4%b9%a0/)
