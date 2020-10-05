# 忽略警告提示
import warnings
warnings.filterwarnings('ignore')

# 导入处理数据包
import numpy as np
import pandas as pd

# 导入数据
# 训练数据集
train = pd.read_csv("./train.csv")
#测试数据集
test = pd.read_csv("./test.csv")
#训练数据891条
print('训练数据集：', train.shape, "\n", '测试数据集：', test.shape,sep='')
