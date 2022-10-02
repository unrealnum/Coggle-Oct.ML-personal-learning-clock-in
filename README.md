# 【打卡】Kaggle SpaceShip Titanic赛题

<br>

<!-- vscode-markdown-toc -->
* 1. [赛题背景](#)
	* 1.1. [赛题介绍](#-1)
	* 1.2. [数据说明](#-1)
* 2. [TASK 1：比赛报名与尝试](#TASK1)
	* 2.1. [2.1.报名成功界面](#-1)
	* 2.2. [2.2.数据初观察](#-1)
	* 2.3. [目前得到的信息：](#-1)
* 3. [TASK 2：比赛数据分析](#TASK2)
	* 3.1. [等等，先别急](#-1)
	* 3.2. [手动删除无效数据](#-1)
		* 3.2.1. [具体处理](#-1)
* 4. [TASK3：验证集划分与树模型](#TASK3)
* 5. [TASK4：特征工程入门](#TASK4)
* 6. [TASK5：特征工程进阶](#TASK5)
* 7. [TASK6：树模型进阶](#TASK6)
* 8. [TASK7：多折训练与集成](#TASK7)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

<br>

##  1. <a name=''></a>赛题背景

<br>

###  1.1. <a name='-1'></a>赛题介绍

欢迎来到 2912 年，您需要数据科学技能来解决宇宙之谜。我们收到了四光年外的信号，情况看起来不太妙。宇宙飞船泰坦尼克号是一个月前发射的星际客轮。船上有近 13,000 名乘客，这艘船开始了处女航，将太阳系的移民运送到围绕附近恒星运行的三颗新可居住的系外行星。

在绕过半人马座阿尔法星前往它的第一个目的地——炎热的巨蟹座 55 E 时，粗心的宇宙飞船泰坦尼克号与隐藏在尘埃云中的时空异常相撞。可悲的是，它遭遇了与 1000 年前同名的命运相似的命运。虽然船完好无损，但几乎有一半的乘客被运送到了异次元！

为了帮助救援人员和找回丢失的乘客，您面临的挑战是使用从飞船损坏的计算机系统中恢复的记录来预测哪些乘客被异常运送。

<br>



###  1.2. <a name='-1'></a>数据说明

在本次比赛中，您的任务是预测在泰坦尼克号飞船与时空异常相撞期间是否有乘客被运送到另一个维度。为了帮助你做出这些预测，你会得到一组从船上受损的计算机系统中恢复的个人记录。

<br>

文件说明：

- train.csv - 大约三分之二 (~8700) 乘客的个人记录，用作训练数据。
- test.csv - 剩余三分之一 (~4300) 乘客的个人记录，用作测试数据。您的任务是为该集合中的乘客预测已运输的值。
- sample_submission.csv - 格式正确的提交文件。

<br>

字段说明：

- PassengerId - 每位乘客的唯一 ID。每个 Id 采用 gggg_pp 的形式，其中 gggg 表示乘客旅行的组，pp 是他们在组中的编号。群体中的人通常是家庭成员，但并非总是如此。
- HomePlanet - 乘客离开的星球，通常是他们的永久居住星球。
- CryoSleep - 指示乘客是否选择在航行期间进入假死状态。处于低温睡眠状态的乘客被限制在他们的客舱内。
- Cabin - 乘客入住的客舱编号。采用deck/num/side 形式，其中side 可以是P 代表左舷或S 代表右舷。
- Destination - 乘客将要去的星球。
- Age - 乘客的年龄。
- VIP - 乘客在航程中是否支付了特殊的 VIP 服务费用。
- RoomService、FoodCourt、ShoppingMall、Spa、VRDeck - 乘客在泰坦尼克号宇宙飞船的众多豪华设施中所支付的金额。
- Name - 乘客的名字和姓氏。
- Transported - 乘客是否被运送到另一个维度。这是目标，您要预测的列。

<br>



##  2. <a name='TASK1'></a>TASK 1：比赛报名与尝试

<br>



###  2.1. <a name='-1'></a>报名成功界面

<br>

**如下图为报名成功的页面。**

![](https://github.com//unrealnum//Coggle-Oct.ML-personal-learning-clock-in//raw//main//图层//join-competition.png)



<br>

###  2.2. <a name='-1'></a>数据初观察



导入可能需要使用的库：<br>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
```

<br>

打印部分数据：<br>

```python
df = pd.read_csv('train.csv')
print(df.head(5))
print("----------------------------------------------------------------------------------")
print(df.tail(5))
```

<br>

结果为：<br>

```shell
  PassengerId HomePlanet CryoSleep  ... VRDeck               Name  Transported
0     0001_01     Europa     False  ...    0.0    Maham Ofracculy        False
1     0002_01      Earth     False  ...   44.0       Juanna Vines         True
2     0003_01     Europa     False  ...   49.0      Altark Susent        False
3     0003_02     Europa     False  ...  193.0       Solam Susent        False
4     0004_01      Earth     False  ...    2.0  Willy Santantines         True

[5 rows x 14 columns]
----------------------------------------------------------------------------------
     PassengerId HomePlanet CryoSleep  ...  VRDeck               Name  Transported
8688     9276_01     Europa     False  ...    74.0  Gravior Noxnuther        False
8689     9278_01      Earth      True  ...     0.0    Kurta Mondalley        False
8690     9279_01      Earth     False  ...     0.0       Fayey Connon         True
8691     9280_01     Europa     False  ...  3235.0   Celeon Hontichre        False
8692     9280_02     Europa     False  ...    12.0   Propsh Hontichre         True

[5 rows x 14 columns]
```

<br>

打印所有的特征名字：<br>

```python
print(df.columns)
```

```shell
Index(['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
       'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
       'Name', 'Transported'],
      dtype='object')
```

<br>

打印数据集形状：<br>

```python
print(df.shape)
```

```sh
(8693, 14)
```

<br>

通过Kaggle自带的数据分布及属性显示得到：<br>

* 部分特征存在缺失值，占总量的2%（217）左右
* **被传送的人数相较于未被传送的人数占比过少**
* 年龄大致呈正态分布

<br>

###  2.3. <a name='-1'></a>目前得到的信息：

* 用于训练的数据集数据总量为8693条
* 数据特征量为13
* 14个特征/标签的名字

<br>

##  3. <a name='TASK2'></a>TASK 2：比赛数据分析

<br>

针对初步观察的情况，问题为二分类问题/异常检测问题（将被传送视为异常），总的特征有14个。<br>

决定先通过人工观察删除部分无效数据，然后对异常数据（缺失值等）进行处理。<br>

###  3.1. <a name='-1'></a>等等，先别急

<br>

把`df.head()`与`df.tail()`的封印先解除。**（数据集列数不多，解除封印能更好的观察数据）**<br>

```python
df = pd.read_csv('train.csv')
pd.set_option('display.max_columns', None)  #解除最小显示列数的限制
print(df.head(5))
print("----------------------------------------------------------------------------------")
print(df.tail(5))
```

```shell
  PassengerId HomePlanet CryoSleep  Cabin  Destination   Age    VIP  \
0     0001_01     Europa     False  B/0/P  TRAPPIST-1e  39.0  False   
1     0002_01      Earth     False  F/0/S  TRAPPIST-1e  24.0  False   
2     0003_01     Europa     False  A/0/S  TRAPPIST-1e  58.0   True   
3     0003_02     Europa     False  A/0/S  TRAPPIST-1e  33.0  False   
4     0004_01      Earth     False  F/1/S  TRAPPIST-1e  16.0  False   

   RoomService  FoodCourt  ShoppingMall     Spa  VRDeck               Name  \
0          0.0        0.0           0.0     0.0     0.0    Maham Ofracculy   
1        109.0        9.0          25.0   549.0    44.0       Juanna Vines   
2         43.0     3576.0           0.0  6715.0    49.0      Altark Susent   
3          0.0     1283.0         371.0  3329.0   193.0       Solam Susent   
4        303.0       70.0         151.0   565.0     2.0  Willy Santantines   

   Transported  
0        False  
1         True  
2        False  
3        False  
4         True  
----------------------------------------------------------------------------------
     PassengerId HomePlanet CryoSleep     Cabin    Destination   Age    VIP  \
8688     9276_01     Europa     False    A/98/P    55 Cancri e  41.0   True   
8689     9278_01      Earth      True  G/1499/S  PSO J318.5-22  18.0  False   
8690     9279_01      Earth     False  G/1500/S    TRAPPIST-1e  26.0  False   
8691     9280_01     Europa     False   E/608/S    55 Cancri e  32.0  False   
8692     9280_02     Europa     False   E/608/S    TRAPPIST-1e  44.0  False   

      RoomService  FoodCourt  ShoppingMall     Spa  VRDeck               Name  \
8688          0.0     6819.0           0.0  1643.0    74.0  Gravior Noxnuther   
8689          0.0        0.0           0.0     0.0     0.0    Kurta Mondalley   
8690          0.0        0.0        1872.0     1.0     0.0       Fayey Connon   
8691          0.0     1049.0           0.0   353.0  3235.0   Celeon Hontichre   
8692        126.0     4688.0           0.0     0.0    12.0   Propsh Hontichre   

      Transported  
8688        False  
8689        False  
8690         True  
8691        False  
8692         True  
```

**现在好多了~<br>**

**顺带再把该关的关了，省的整天warning**<br>

```python
import warnings
warnings.filterwarnings('ignore')
```

<br>

###  3.2. <a name='-1'></a>手动删除无效数据

<br>

通过观察得到：<br>

* `'PassengerId'`<font color='red'>**暂时**</font>不作为特征进行处理。<br>

> - `PassengerId` - A unique Id for each passenger. Each Id takes the form `gggg_pp` where `gggg` indicates a group the passenger is travelling with and `pp` is their number within the group. People in a group are often family members, but not always.
>
> * PassengerId - 每位乘客的唯一 ID。每个 Id 采用 gggg_pp 的形式，其中 gggg 表示乘客旅行的组，pp 是他们在组中的编号。群体中的人通常是家庭成员，但并非总是如此。
>
> 这是可能作为特征的（一个组里的一起传的概率会不会更大？），但这里先不考虑。

* `name`不作为特征进行处理。<br>
* 将`transported`单独切分为待预测的标签。<br>

####  3.2.1. <a name='-1'></a>具体处理

<br>

```python
df.drop(['Name'], axis=1, inplace=True)
df.drop(['PassengerId'], axis=1, inplace=True)
print(df.shape)
print(df.head(3))
```

```shell
(8693, 12)
  HomePlanet CryoSleep  Cabin  Destination   Age    VIP  RoomService  \
0     Europa     False  B/0/P  TRAPPIST-1e  39.0  False          0.0   
1      Earth     False  F/0/S  TRAPPIST-1e  24.0  False        109.0   
2     Europa     False  A/0/S  TRAPPIST-1e  58.0   True         43.0   

   FoodCourt  ShoppingMall     Spa  VRDeck  Transported  
0        0.0           0.0     0.0     0.0        False  
1        9.0          25.0   549.0    44.0         True  
2     3576.0           0.0  6715.0    49.0        False  
```

**可以看到相关特征已经被删除了。**<br>

**这里我们对待检测集也进行上述操作：**

```python
df_predict = pd.read_csv('test.csv')
```

```python
df_predict.drop(['Name'], axis=1, inplace=True)
df_predict.drop(['PassengerId'], axis=1, inplace=True)
print(df_predict.shape)
print(df_predict.head(3))
```

```shell
(4277, 11)
  HomePlanet CryoSleep  Cabin  Destination   Age    VIP  RoomService  \
0      Earth      True  G/3/S  TRAPPIST-1e  27.0  False          0.0   
1      Earth     False  F/4/S  TRAPPIST-1e  19.0  False          0.0   
2     Europa      True  C/0/S  55 Cancri e  31.0  False          0.0   

   FoodCourt  ShoppingMall     Spa  VRDeck  
0        0.0           0.0     0.0     0.0  
1        9.0           0.0  2823.0     0.0  
2        0.0           0.0     0.0     0.0  
```

<br>



### 缺失值观察处理

#### 缺失值观察

**调用如下代码脚本进行缺失值的可视化分析：**<br>

```python
missing = df.isnull().sum().reset_index().rename(columns={0:'missNum'})
missing['missRate'] = missing['missNum']/df.shape[0]
# 按照缺失率排序显示
miss_analy = missing[missing.missRate>0].sort_values(by='missRate',ascending=False)

import pylab as pl

fig = plt.figure(figsize=(24,6))#调整画布大小（长，宽）
plt.bar(np.arange(miss_analy.shape[0]), list(miss_analy.missRate.values), align = 'center',color=['red','green','yellow','steelblue'])

plt.title('Histogram of missing value of variables')
plt.xlabel('variables names')
plt.ylabel('missing rate')
# 添加x轴标签，并旋转90度
plt.xticks(np.arange(miss_analy.shape[0]),list(miss_analy['index']))
pl.xticks(rotation=90)
# 添加数值显示
for x,y in enumerate(list(miss_analy.missRate.values)):
    plt.text(x,y+0.005,'{:.2%}'.format(y),ha='center',rotation=90)    #y+delta, default=0,可用于调整文本高度
plt.ylim([0,0.05])  #更改y值的范围, default=[0,1]

plt.show()
```

![](C:\Users\unreal_num border\Desktop\Coggle-Oct.ML-personal-learning-clock-in\图层\缺失值显示.png)

**可以看到整体的一个缺失率分布情况：其中CryoSleep的数据丢失占比最大**<br>



#### 缺失值处理

先简单的把带有缺失值的行全部drop。<br>

**花里胡哨的操作后面再搞。**<br>

```python
df.dropna(inplace=True)
print(df.shape)
```

```shell
(6764, 12)   #数据少了200多TAT
  HomePlanet CryoSleep  Cabin  Destination   Age    VIP  RoomService  \
0     Europa     False  B/0/P  TRAPPIST-1e  39.0  False          0.0   
1      Earth     False  F/0/S  TRAPPIST-1e  24.0  False        109.0   
2     Europa     False  A/0/S  TRAPPIST-1e  58.0   True         43.0   
3     Europa     False  A/0/S  TRAPPIST-1e  33.0  False          0.0   
4      Earth     False  F/1/S  TRAPPIST-1e  16.0  False        303.0   

   FoodCourt  ShoppingMall     Spa  VRDeck  Transported  
0        0.0           0.0     0.0     0.0        False  
1        9.0          25.0   549.0    44.0         True  
2     3576.0           0.0  6715.0    49.0        False  
3     1283.0         371.0  3329.0   193.0        False  
4       70.0         151.0   565.0     2.0         True 
```







#### 训练集探索性分析



#### 待检测集探索性分析



<br>

##  4. <a name='TASK3'></a>TASK 3：验证集划分与树模型

<br>

### 原始数据集切分操作

由于待检测集的标签（transported）是不存在的，所以对用于训练的数据集进行切分操作，切分为训练集与测试集。<br>

（随机种子先固定以便于后续的复现及调整）<br>

（K折交叉验证后面调参时再做）<br>

**先分离原始数据集的标签**：<br>

```python
df_label = pd.DataFrame(df['Transported'])
df.drop(['Transported'], axis=1, inplace=True)
print(df_label.shape, df.shape)
```

```shell
(8693, 1) (8693, 11)
```

**然后调用`sklearn.modelselection`中的`train_test_split`方法切分数据**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, df_label, shuffle = True, 
                                     test_size = 0.2, random_state = 114514)

print(X_train.head(5))
print("----------------------------------------------------------------------------------")
print(X_test.head(5))
print("----------------------------------------------------------------------------------")
print(y_train.head(5))
print("----------------------------------------------------------------------------------")
print(y_test.head(5))
print("----------------------------------------------------------------------------------")

```

```shell
     HomePlanet CryoSleep     Cabin    Destination   Age    VIP  RoomService  \
8369      Earth     False  G/1452/P  PSO J318.5-22  73.0  False       1806.0   
8240     Europa      True   D/266/S    TRAPPIST-1e  29.0  False          0.0   
2737      Earth     False   F/615/P  PSO J318.5-22  23.0  False          5.0   
6883     Europa      True   D/227/P    55 Cancri e  32.0  False          0.0   
6863      Earth     False  F/1502/P    TRAPPIST-1e  43.0  False        473.0   

      FoodCourt  ShoppingMall  Spa  VRDeck  
8369        0.0          47.0  0.0     0.0  
8240        0.0           0.0  0.0     0.0  
2737      308.0         363.0  7.0     0.0  
6883        0.0           0.0  0.0     0.0  
6863     1390.0           0.0  0.0     0.0  
----------------------------------------------------------------------------------
     HomePlanet CryoSleep     Cabin  Destination   Age    VIP  RoomService  \
2580       Mars     False   F/532/S  TRAPPIST-1e  27.0  False       1502.0   
3403     Europa     False   B/142/S  55 Cancri e  22.0  False          NaN   
6718      Earth     False  F/1468/P  TRAPPIST-1e  63.0  False          0.0   
2735      Earth     False   G/468/S  TRAPPIST-1e  21.0  False         15.0   
8533       Mars     False   E/598/S  TRAPPIST-1e  38.0  False       1478.0   

      FoodCourt  ShoppingMall    Spa  VRDeck  
2580        1.0           0.0    0.0     0.0  
3403      171.0        2233.0    1.0   800.0  
6718      742.0           0.0  125.0     3.0  
2735       30.0          11.0    2.0   388.0  
8533        0.0         631.0    0.0     0.0  
----------------------------------------------------------------------------------
      Transported
8369        False
8240         True
2737         True
6883         True
6863         True
----------------------------------------------------------------------------------
      Transported
2580        False
3403         True
6718        False
2735        False
8533        False
----------------------------------------------------------------------------------
```

<br>



##  5. <a name='TASK4'></a>TASK 4：特征工程入门

<br>



##  6. <a name='TASK5'></a>TASK 5：特征工程进阶

<br>



##  7. <a name='TASK6'></a>TASK 6：树模型进阶

<br>



##  8. <a name='TASK7'></a>TASK 7：多折训练与集成

<br>

