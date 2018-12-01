# NeNe帮助文档

## 1. 关于NeNe

NeNe主要基于numpy完成。NeNe是一个在调用方法上模仿keras的神经网络库，提供了方便的接口，能够快捷的搭建自己的神经网络，并在已有的类基础上设计自己的激活函数或是层函数。

## 2. 快速开始NeNe

### 2.1 搭建网络和显示

使用类似以下的语句能够快速搭建神经网络。

```py
    import NeNe
    from NeNe.Activation import *
    from NeNe.Loss import *
    from NeNe.Layer import *
    from NeNe.LRD import *

    model = NeNe.NeNe()
    model.add(InputLayer(num=num_input))
    model.add(Layer(num=500, activation=Tanh(),init_seed='norm'))
    model.add(Layer(num=500, activation=Tanh(),init_seed='norm'))
    model.add(Layer(num=500, activation=Tanh(),init_seed='norm'))
    model.add(Layer(num=500, activation=Tanh(),init_seed='norm'))
    model.add(Layer(num=num_output, activation=SoftMax(),init_seed='norm'))
```

对于已声明的NeNe.NeNe对象，可以使用add()方法添加层。add方法接受一个NeNe.Layer.Layer对象。  
初始化一个NeNe.Layer对象需要三个参数：num,activation,init_seed。第一个参数代表该隐藏层的神经元数量；第二个参数代表该层使用的激活函数，接受一个NeNe.Activation.Basic_Activation对象或继承自该类的预置或自定义激活函数，默认为线性(linear())。第三个参数代表对层偏置(bias)参数的初始化方法，接受字符串作为关键字：'norm'代表正态分布的数值，'zero'代表全为0。  
特别的，在添加第一层（输入层）的时候，需要使用NeNe.Layer.InputLayer对象。该对象只用指明输入的特征数量num即可。

在构建网络完成之后，可以使用

`model.summary()`


打印该网络的相关信息。以上文代码所示，其将显示如下内容：

```
---------------NeNe neural network model summary------------------  
Basic Information:  
Depth:6  
Input shape:9  
Output shape:5  
Network structure:
           Name     Neural_Num      Param_Num     Activation
------------------------------------------------------------------
  Layer0(Input)              9              0         linear

         Layer1            500           5000           tanh

         Layer2            500         250500           tanh

         Layer3            500         250500           tanh

         Layer4            500         250500           tanh

         Layer5              5           2505        SoftMax

        Summary           2014        5488145         ------
```

### 2.2 训练

在处理好输入和输出的数据集后，可以进行训练。数据应该是m（样本数）*n（特征数）形式的。使用.fit()方法即可以进行训练。其使用例如下：

```py
model.fit(
        train_data=(x_train, y_train),
        valid_data=(x_valid, y_valid),
        epoch=20,
        lrf=LRD_cooldown(lr=1.0, epoch_wait=3, decay_rate=0.1),
        batch_size=5,
        loss=MSE(),
        accu_echo=True)
```

其中，train_data和valid_data两个参数均接受由输入和输出数据组成的元组或列表。  
epoch为训练的轮数。  
lrf代表学习率的更新函数，其必须是NeNe.LRD.LRC对象或者其子类。LRC()代表学习率在学习过程中恒定；LRD_expdecay代表按照指数形式衰减；LRD_cooldown则代表指定(由 epoch_wait 参数指定)回数内损失函数不下降是衰减学习率。这些类的详细会在LRD模块中详细说明。  
batch_size为分批训练时的批大小。指定batch_size=1时，相当于使用随机梯度下降法(SGD)；指定batch_size为训练集大小时,相当于使用传统的梯度下降法(GD)。  
loss代表损失函数，目前接受两种由NeNe.Loss提供的损失函数类：平均交叉熵损失函数(CEL)和均方差误差损失函数(MSE)。前者用于分类问题，后者在回归问题应用较多，可以，但不推荐在分类问题中使用。  
accu_echo代表是否在训练中显示准确率。建议仅在分类问题时设置为True,否则返回值可能不具有实际意义。

### 2.3 测试

使用如下函数可以快速的进行模型在测试集的表现比对。

```py
loss, accu = model.predict(x_test, y_test, loss=MSE(), get_accu=True)
```

不指定测试集的输出时，该函数将返回其预测的输出；否则返回给定样本的给定损失函数下的损失和准确率。



## 3. NeNe.NeNe
用于搭建和训练模型的类。

### 3.1 add方法

add方法用于在既有的模型上添加层。接受一个NeNe.Layer.Layer对象或其继承类型。

对于添加的第一层（输入层），必须是NeNe.Layer.InputLayer。

使用例：

```py
model.add(InputLayer(num=num_input))
model.add(Layer(num=500, activation=Tanh(),init_seed='norm'))
```

### 3.2 summary方法

将打印网络的结构和参数。
可以用于检查和确认网络结构。  
直接调用即可：

```py
model.summary()
```

### 3.3 fit方法

fit()方法用以进行指定数据的训练。其使用例如下：

```py
model.fit(
        train_data=(x_train, y_train),
        valid_data=(x_valid, y_valid),
        epoch=20,
        lrf=LRD_cooldown(lr=1.0, epoch_wait=3, decay_rate=0.1),
        batch_size=5,
        loss=MSE(),
        accu_echo=True)
```

其中，train_data和valid_data两个参数均接受由输入和输出数据组成的元组或列表。  
epoch为训练的轮数。  
lrf代表学习率的更新函数，其必须是NeNe.LRD.LRC对象或者其子类。
batch_size为分批训练时的批大小。指定batch_size=1时，相当于使用随机梯度下降法(SGD)；指定batch_size为训练集大小时,相当于使用传统的梯度下降法(GD)。  
loss代表损失函数，目前接受两种由NeNe.Loss提供的损失函数类：平均交叉熵损失函数(CEL)和均方差误差损失函数(MSE)。 
accu_echo代表是否在训练中显示准确率。建议仅在分类问题时设置为True,否则返回值可能不具有实际意义。

### 3.4 predict 方法

使用该函数可以快速的进行模型在测试集的表现比对。

```py
loss, accu = model.predict(x_test, y_test, loss=MSE(), get_accu=True)
```

第一个参数为输入数据。  
第二个参数为输出数据，可不指定，不指定测试集的输出时，该函数将返回其预测的输出；否则返回给定样本的给定损失函数下的损失和准确率。  
第三个参数为计算损失时所使用的损失函数，默认为均方根误差(MSE)。  
第四个参数为是否返回准确率。建议仅在分类问题时设置为True,否则返回值可能不具有实际意义。  

## 4. NeNe.Activation

Activation模块提供了部分常用的激活函数，以快速的进行激活映射和反向传播时的求导。
> Relu函数有些问题，还没调整好，因此没有提供。

### 4.1 Base_Activation

激活函数的基类。其激活方式与线性相同。

### 4.2 linear

线性激活函数。其输入与输出相同。
$$ y=x $$

### 4.3 sigmoid

即logstic激活函数:

$$ y=\frac{1}{1+e^x} $$

### 4.4 tanh

双曲激活函数其曲线趋势和sigmoid函数相同。不同的是tanh将结果映射到[-1,1]的区间。

$$ y=tanh(x) $$

### 4.5 softmax

softmax函数多用于分类网络的输出层，用于将神经元的输出归一化，表示各个种类的可能的概率。该函数与交叉熵函数组合时在反向传播具有衰减低的优势。

$$ y_i=\frac{e^i}{\sum_{m=1}^n{e^m}}$$

使用这些激活函数类时，直接创建实例即可。一个使用例如下：

```py
Layer(num=10,activation=NeNe.Activation.Softmax)
```

## 5 NeNe.Layer

NeNe.Layer模块中的Layer对象是构成NeNe.NeNe神经网络的主要对象，后者将根据层的大小属性建立起层之间连接的权重矩阵。

### 5.1 Layer

#### 5.1.1 初始化Layer对象

构建Layer对象需要三个参数：：num,activation,init_seed。第一个参数代表该隐藏层的神经元数量；第二个参数代表该层使用的激活函数，接受一个NeNe.Activation.Basic_Activation对象或继承自该类的预置或自定义激活函数，默认为线性(linear())。第三个参数代表对层偏置(bias)参数的初始化方法，接受字符串作为关键字：'norm'代表正态分布的数值，'zero'代表全为0。 

```py
def __init__(self, num, activation=None, init_seed='norm'):
```

#### 5.1.2 可调用的类属性：neural_num和acti

二者将分别返回该层的神经元数量和激活函数名称。

#### 5.1.3 前向传播/反向传播 forwardPropagation/backwardPropagation

用于进行结果的计算和反向传播更新网络。一般不用进行直接的调用。

### 5.2 InputLayer

作为输入层用，为Layer的子类。  
初始化InputLayer只需要提供特征的数目num即可。
InputLayer不使用偏置和激活（使用线性激活函数）。

## 6 NeNe.Loss

该模块内提供了两个常用的损失函数：平均交叉熵函数(CEL)和均方差损失函数（MSE）。前者用于分类问题，后者在回归问题应用较多，可以，但不推荐在分类问题中使用。  

### 6.1 平均交叉熵函数(CEL)

平均交叉熵的计算公式如下：
$$ 
CEL=\sum_{i=1}^{n}{log(a_i)y_i} \\
ACE=\frac{CEL}{n}
$$

>CEL的loss显示有问题，但是能够进行正常的训练。待修正。

### 6.2 均方根误差函数(MSE)

均方根误差的计算公式如下:
$$J(\theta) = \frac{1}{2m}\sum_{i = 0} ^m(y^i - h_\theta (x^i))^2$$

### 6.3 成员函数

该两类均有get_loss，get_loss_deriv两个成员函数（实际被编写为了类函数形式，不需要对象）
前者用于返回损失率，函数头如下：

```py
@classmethod
def get_loss(cls, y_output, y_target, return_accu=True):
```

接受至少两个参数：实际输出和目标输出。
第三个参数代表是否希望返回准确率。

后者用于返回损失率的倒数，用于反向更新：

```py
@classmethod
def get_loss_deriv(cls, y_output, y_target):
```

只接受实际输出和目标输出两个参数。

## 7. NeNe.LRD

LRD模块提供了常见的学习率调整函数类，这些类用于在训练中更新学习率。

由于NeNe.fit()函数的lrf参数接受且仅接受NeNe.LRD中的对象，因此即使不调整学习率，也需要使用LRD.LRC对象创建恒定的学习率对象。这些类均具有update(self,epoch,err)成员函数，NeNe.NeNe在每轮训练结束时使用该函数完成学习率的更新。可以据此编写自己的学习率更新函数。

### 7.1 NeNe.LRC

该类提供恒定的学习率。
创建该类需要提供学习率(lr):

```py
def __init__(self, lr=0.01):
```

### 7.2 NeNe.LRD_expdecay

该类能够以指数衰减的形式更新学习率。

```py
def __init__(self, lr=0.01, decay_rate=0.9, decay_epoch=1,staircase=False):
```

需要4个可选参数：初始的学习率(lr)，衰减率（[0, 1]之间的浮点数），特征轮数（int）,学习率是否平台式恒定staircase。

指定staircase=True时：
$$lr_{epoch}=lr_{init}*decayRate^{int(\frac{epoch}{ decayEpoch})}$$
其更新是不连续的。为False时：
$$lr_{epoch}=lr_{init}*decayRate^{\frac{epoch}{ decayEpoch}}$$

### 7.3 NeNe.LRD_cooldown

该类在学习率不下降时降低学习率，使得模型更好收敛。

```py
def __init__(self, lr, epoch_wait=1, decay_rate=0.9):
```

接受三个可选参数：初始的学习率(lr)，最大不下降轮数epoch_wait，衰减率（[0, 1]之间的浮点数）
在经过epoch_wait轮而不下降的训练时，学习率将按照如下公式更新：
$$ lr_{epoch}=lr_{now}*decayRate$$
