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

