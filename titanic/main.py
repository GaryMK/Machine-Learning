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
pd.set_option('max_colwidth', 100)

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

# 登录港口（Embarked）
'''
登船港口(Embarked)的值是：
出发地点：S=英国南安普顿Southampton
途径地点1：C=法国 瑟堡市Cherbourg
途径地点2：Q=爱尔兰 昆士敦Queenstown
'''
print(full['Embarked'].head())
# 存放提取后的特征
embarkedDf = pd.DataFrame()
'''
使用get_dummies进行one-hot编码，产生虚拟变量（dummy variables），列名前缀是Embarked
'''
embarkedDf = pd.get_dummies(full['Embarked'], prefix='Embarked')
# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full, embarkedDf], axis = 1)
'''
因为已经使用登船港口(Embarked)进行了one-hot编码产生了它的虚拟变量（dummy variables）
所以这里把登船港口(Embarked)删掉
'''
full.drop('Embarked', axis=1, inplace=True)
print(full.head())
'''
上面drop删除某一列代码解释：
因为drop(name,axis=1)里面指定了name是哪一列，比如指定的是A这一列，axis=1表示按行操作。
那么结合起来就是把A列里面每一行删除，最终结果是删除了A这一列.
简单来说，使用drop删除某几列的方法记住这个语法就可以了：drop([列名1,列名2],axis=1)
'''

# 客舱等级（Pclass）
'''
客舱等级(Pclass):
1=1等舱，2=2等舱，3=3等舱
'''
# 存放提取后的特征
pclassDf = pd.DataFrame()

# 用get_dummies进行one-hot编码，列名前缀是Pclass
pclassDf = pd.get_dummies(full['Pclass'], prefix= 'Pclass')
print(pclassDf.head())
# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full, pclassDf], axis=1)
# 删除客舱等级（Pclass）这一列
full.drop('Pclass', axis=1, inplace=True)
print(full.head())

# 3.2.1 分类类型：字符串类型
# 从姓名中提取头衔
'''
查看姓名这一列长啥样
注意到在乘客名字（Name）中，有一个非常显著的特点：
乘客头衔每个名字当中都包含了具体的称谓或者说是头衔，将这部分信息提取出来后可以作为非常有用一个新变量，可以帮助我们进行预测。
例如：
Braund, Mr. Owen Harris
Heikkinen, Miss. Laina
Oliva y Ocana, Dona. Fermina
Peter, Master. Michael J
'''
print(full['Name'].head())
#练习从字符串中提取头衔，例如Mr
#split用于字符串分割，返回一个列表
#看到姓名中'Braund, Mr. Owen Harris'，逗号前面的是“名”，逗号后面是‘头衔. 姓’
namel = 'Braund, Mr. Owen Harris'
'''
split用于字符串按分隔符分割，返回一个列表。这里按逗号分隔字符串
也就是字符串'Braund, Mr. Owen Harris'被按分隔符,'拆分成两部分[Braund,Mr. Owen Harris]
你可以把返回的列表打印出来瞧瞧，这里获取到列表中元素序号为1的元素，也就是获取到头衔所在的那部分，即Mr. Owen Harris这部分
'''
# Mr. Owen Harris
str1 = namel.split(',')[1]
'''
继续对字符串Mr. Owen Harris按分隔符'.'拆分，得到这样一个列表[Mr, Owen Harris]
这里获取到列表中元素序号为0的元素，也就是获取到头衔所在的那部分Mr
'''
# Mr.
str2 = str1.split(',')[0]
# strip() 方法用于移除字符串头尾指定的字符（默认为空格）
str3 = str2.strip()

'''
定义函数：从姓名中获取头衔
'''
def getTitle(name):
    str1=name.split( ',' )[1] #Mr. Owen Harris
    str2=str1.split( '.' )[0]#Mr
    #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3=str2.strip()
    return str3

# 存放提取后的特征
titleDf = pd.DataFrame()
# map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = full['Name'].map(getTitle)
print(titleDf.head())

'''
定义以下几种头衔类别：
Officer政府官员
Royalty王室（皇室）
Mr已婚男士
Mrs已婚妇女
Miss年轻未婚女子
Master有技能的人/教师
'''
# 姓名中头衔字符串与定义头衔类别的映射关系
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
# map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)

# 使用get_dummies进行one-hot编码
titleDf['Title'] = titleDf['Title'].map(title_mapDict)

# 使用get_dummies 进行 one-hot编码
titleDf = pd.get_dummies(titleDf['Title'])
print(titleDf.head())

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full, titleDf], axis=1)

# 删除名字这一列
full.drop('Name', axis=1, inplace=True)
full.head()

# 从客舱号中提取客舱类别
# 补充知识：匿名函数
'''
python 使用 lambda 来创建匿名函数。
所谓匿名，意即不再使用 def 语句这样标准的形式定义一个函数，预防如下：
lambda 参数1，参数2：函数体或者表达式
'''
# 定义匿名函数：对两个数相加
sum = lambda a, b: a + b
# 调用sum函数
print("相加后的值为：", sum(10, 20))

'''
客舱号的首字母是客舱的类别
'''
# 查看客舱号的内容
print(full['Cabin'].head())

# 存放客舱号信息
cabinDf = pd.DataFrame()

'''
客场号的类别值是首字母，例如：
C85 类别映射为首字母C
'''
full['Cabin'] = full['Cabin'].map(lambda c : c[0])

# 使用get_dummies进行one-hot编码，列名前缀是Cabin
cabinDf = pd.get_dummies(full['Cabin'], prefix='Cabin')
print(cabinDf.head())

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full, cabinDf], axis=1)

# 删除客舱号这一列
full.drop('Cabin', axis=1, inplace=True)
print(full.head())

# 建立家庭人数和家庭类别
# 存放家庭信息
familyDf = pd.DataFrame()

'''
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
（因为乘客自己也是家庭成员的一个，所以这里加1）
'''
familyDf['FamilySize'] = full['Parch'] + full['SibSp'] + 1

'''
家庭类别：
小家庭Family_Single：家庭人数=1
中等家庭Family_Small: 2<=家庭人数<=4
大家庭Family_Large: 家庭人数>=5
'''
# if 条件为真的时候返回if前面内容，否则返回0
familyDf['Family_Single'] = familyDf['FamilySize'].map(lambda s : 1 if s == 1 else 0)
familyDf['Family_Small'] = familyDf['FamilySize'].map(lambda s : 1 if 2 <= s <=4 else 0)
familyDf[ 'Family_Large' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
print(familyDf.head())

# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full, familyDf], axis=1)
print(full.head())

#到现在我们已经有了这么多个特征了
print(full.shape)