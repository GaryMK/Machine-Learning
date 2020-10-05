# 1.题出问题
# 什么样的人在泰坦尼克号中更容易存活？


# 2.理解数据
# 2.1 采集数据
# https://www.kaggle.com/c/titanic


# 2.2 导入数据
# 忽略警告提示
import warnings
warnings.filterwarnings('ignore')

# 导入处理数据包
import numpy as np
import pandas as pd

# 导入数据
# 训练数据集
train = pd.read_csv("./train.csv")
# 测试数据集
test = pd.read_csv("./test.csv")
# 显示所有列
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('max_colwidth', 100)

# 训练数据891条
print('训练数据集：', train.shape, "\n", '测试数据集：', test.shape, sep='')

rowNum_train = train.shape[0]
rowNum_test = test.shape[0]
print('kaggle训练数据集行数：', rowNum_train,
      '\n'
      'kaggle测试数据集行数：', rowNum_test,
      sep ='')

#合并数据集，方便同时对两个数据集进行清洗
full = train.append(test, ignore_index = True)
print('合并后的数据集：', full.shape)

'''
describe只能查看数据类型的描述统计信息，对于其他类型的数据不显示，比如字符串类型姓名（name），客舱号（Cabin）
这很好理解，因为描述统计指标是计算数值，所以需要该列的数据类型是数据
'''
# 2.3 查看数据集信息
# 查看数据
print(full.head())

# 获取数据类型列的描述统计信息
full.describe()

# 查看每一列的数据类型，和数据总数
full.info()
'''
数据总共有1309行。
其中数据类型列：年龄（Age）、船舱号（Cabin）里面有缺失数据：
1）年龄（Age）里面数据总数是1046条，缺失了1309-1046=263，缺失率263/1309=20%
2）船票价格（Fare）里面数据总数是1308条，缺失了1条数据

字符串列：
1）登船港口（Embarked）里面数据总数是1307，只缺失了2条数据，缺失比较少
2）船舱号（Cabin）里面数据总数是295，缺失了1309-295=1014，缺失率=1014/1309=77.5%，缺失比较大
这为我们下一步数据清洗指明了方向，只有知道哪些数据缺失数据，我们才能有针对性的处理。
'''

# 3.数据清洗
# 3.1数据预处理
'''
数据总共有1309行。
其中数据类型列：年龄（Age）、船舱号（Cabin）里面有缺失数据：
1）年龄（Age）里面数据总数是1046条，缺失了1309-1046=263，缺失率263/1309=20%
2）船票价格（Fare）里面数据总数是1308条，缺失了1条数据

对于数据类型，处理缺失值最简单的方法就是用平均数来填充缺失值
'''
print('处理前：')
full.info()
# 年龄（Age）
full['Age'] = full['Age'].fillna(full['Age'].mean)
# 船票价格（Fare）
full['Fare'] = full['Fare'].fillna(full['Fare'].mean)
print('处理好后：')
full.info()

# 检查数据处理是否正常
print(full.head())

'''
总数据是1309
字符串列：
1）登船港口（Embarked）里面数据总数是1307，只缺失了2条数据，缺失比较少
2）船舱号（Cabin）里面数据总数是295，缺失了1309-295=1014，缺失率=1014/1309=77.5%，缺失比较大
'''
#登船港口（Embarked）：查看里面数据长啥样
'''
出发地点：S=英国南安普顿Southampton
途径地点1：C=法国 瑟堡市Cherbourg
途径地点2：Q=爱尔兰 昆士敦Queenstown
'''
print(full['Embarked'].head())

'''
分类变量Embarked，看下最常见的类别，用其填充
'''
print(full['Embarked'].value_counts())

'''
从结果来看，S类别最常见。我们将缺失值填充为最频繁出现的值：
S=英国南安普顿Southampton
'''
full['Embarked'] = full['Embarked'].fillna( 'S' )

# 船舱号（Cabin）：查看里面数据长啥样
print(full['Cabin'].head(), '\n')

# 缺失数据比较多，船舱号（Cabin）缺失值填充为U，表示未知（Uknow）
full['Cabin'] = full['Cabin'].fillna( 'U' )

# 检查数据处理是否正常
print(full.head(), '\n')

# 查看最终缺失值处理情况，记住生成情况（Survived）这里一列是我们的标签，
# 用来做机器学习预测的，不需要处理这一列
full.info()

# 3.2 特征提取
# 3.2.1 数据分类
'''
1.数值类型：
乘客编号（PassengerId），年龄（Age），船票价格（Fare），同代直系亲属人数（SibSp），不同代直系亲属人数（Parch）
2.时间序列：无
3.分类数据：
1）有直接类别的
乘客性别（Sex）：男性male，女性female
登船港口（Embarked）：出发地点S=英国南安普顿Southampton，途径地点1：C=法国 瑟堡市Cherbourg，出发地点2：Q=爱尔兰 昆士敦Queenstown
客舱等级（Pclass）：1=1等舱，2=2等舱，3=3等舱
2）字符串类型：可能从这里面提取出特征来，也归到分类数据中
乘客姓名（Name）
客舱号（Cabin）
船票编号（Ticket）
'''
full.info()

# 3.2.1 分类数据：有直接类别的
# 乘客性别（Sex）： 男性male，女性female
# 登船港口（Embarked）：出发地点S=英国南安普顿Southampton，途径地点1：C=法国 瑟堡市Cherbourg，出发地点2：Q=爱尔兰 昆士敦Queenstown
# 客舱等级（Pclass）：1=1等舱，2=2等舱，3=3等舱

# 性别
# 查看性别数据列
print(full['Sex'].head())

'''
将性别的值映射为数值
男（male）对应数值1，女（female）对应数值0
'''
sex_mapDict = {'male' : 1,
            'female' : 0}
# map函数：对Series每个数据应用自定义的函数计算
full['Sex'] = full['Sex'].map(sex_mapDict)
print(full.head())