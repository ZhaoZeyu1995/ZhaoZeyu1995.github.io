---
title:      TensorFlow和Keras简介
date:       2019-11-07
---
# TensorFlow和Keras 简介
- - - -
## 本文的主要目的
介绍Tensorflow以及Keras的基本使用方法，主要介绍作为高级API的Keras的主要使用方法，包括
1. Keras的基本逻辑
2. Sequential API
3. Functional API
4. 模型的保存和读取
5. Callbacks实用模块
6. 自定义
- - - -
## TensorFlow 和Keras
### TensorFlow
[TensorFlow](https://www.tensorflow.org/)是Google公司所开发的一套深度学习框架。使用TensorFlow我们可以搭建几乎任意的深度神经网络结构并进行训练、预测和测试。截止到目前为止(2019年11月6日)，TensorFlow的最新版本为2.0，这个版本相比于之前的版本(1.x)在许多方面进行了改进。不过，虽然如此，本文还是以鄙人较为熟悉的1.14版本来进行介绍，不过不用太担心，在基本的使用方法上，TensorFlow2.0并没有进行太大程度上的改变，所以本文还是具有一定的参考意义的。
### Keras
[Keras](https://keras.io/)原本是Google旗下的一个小团队开发的一个高级的API，它实际上只是一个使用友好的API，并不涉及神经网络中的核心计算内容。最初，它支持CNTK、Theano以及TensorFlow。后来，随着Keras的使用热度不断增加，现在Kera已经被TensorFlow集成，成为了一个专属于TensorFlow的高级API。通过使用Keras，可以将建立、训练、预测以及测试神经网络的过程变得非常之简单。这也是目前TensorFlow官网强推的高级API。Keras的灵活性和易用性结合得非常之好。事实上，灵活性和易用性这二者在某种意义上来说是存在着矛盾的，为何？灵活性往往意味着功能强大，功能强大则不易于掌握其全部用法；反之，易用性往往意味着功能精简，功能精简则易于学习掌握。Keras可谓面面俱到，对于不同程度的使用者，其有不同的API相适用。
- - - -
## 深度神经网络简介
这一部分的内容我就不在这篇文档中介绍了，因为大家可以在很多材料中学习到相关的内容。关于神经网络，我会在课上进行非常简要的介绍。
- - - -
## TensorFlow的安装
详细信息可以查看TensorFlow的官方网站
```python
pip install tensorflow # CPU version
# Or
pip install tensorflow-gpu # GPU version
```
- - - -
## Keras基本逻辑
Keras的基本逻辑如下，欲刻画一个神经网络，我们需知道，神经网络或者所谓之模型(Model)的基本单位是神经元，然而，事实上神经元并非完全没有结构，这个结构是指若干个神经元所排列成的层(Layer)，即模型由层所组成。为了刻画层和层之间的连接关系，需要定义每一个层的输入和输出节点，有了这些节点就可以将层和层之间联系起来从而组成模型，而这些层的输入和输出无一例外都是张量(Tensor)。这样的想法是十分直接的，Keras也是按照这样的逻辑进行设计的。以下分别介绍之。
### Tensor
`tf.Tensor`是所有Tensor的基类，使用Keras API的时候，我们并不会经常的用到它，最常用的应该是类似于这样的一个用法`inputs = tf.keras.Input(shape=(100, 64))`。 我们可以检测一下看看`inputs`是什么，`type(inputs)`它会返回一大长串的东西，不同环境之下或许返回的东西还不一样。不过，我们可以使用`isinstance(inputs, tf.Tensor)`来验证它是不是`tf.Tensor`，这将会返回`True`，从而说明了`inputs`确实是一个`tf.Tensor`的实例。
### Layer
在Keras API之下所有的Layer都应该是`tf.keras.layers.Layer`的子类，我们可以简单的实验一下。定义一个全联接层
```python
import tensorflow as tf
from tensorflow import keras
layer = keras.layers.Dense(10)
isinstance(layer, keras.layers.Layer)
# >>> True
```
### Model
在Keras API之下，所有的Model都是`tf.keras.models.Model`或者`tf.keras.Model`的子类。我们可以尝试定义一个Model。
```python
model = keras.Sequential()
isinstance(model, tf.keras.models.Model)
# >>> True
isinstance(model, tf.keras.Model)
# >>> True 
```
**注：Sequential API下面将要介绍，这里仅仅是一个示例**
- - - -
## Keras Sequential API
或许，我们能想到的一个最简单的神经网络可能就是一些全联接层的叠加了吧？想要创建一个Sequential模型，可以按照如下的方法。
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape
# >>> (60000, 28, 28)
y_train.shape
# >>> (60000,)
x_test.shape
# >>> (10000, 28, 28)
y_test.shape
# >>> (10000,)

# Model
model = keras.Sequential()
model.add(layers.Dense(32, activation=‘relu’, input_shape=(784,)))
model.add(layers.Dense(10, activation=’softmax’))
model.summary()
‘’’
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_2 (Dense)              (None, 32)                25120
_________________________________________________________________
dense_3 (Dense)              (None, 10)                330
=================================================================
Total params: 25,450
Trainable params: 25,450
Non-trainable params: 0
_________________________________________________________________
‘’’
model.compile(optimizer=‘Adam’, loss=‘sparse_categorical_crossentropy’, metrics=[‘acc’])

# Train
x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))
model.fit(x_train, y_train, batch_size=32, validation_split=0.1, epochs=5)
‘’’
Train on 54000 samples, validate on 6000 samples
Epoch 1/5
54000/54000 [==============================] - 2s 30us/sample - loss: 0.2926 - acc: 0.9258 - val_loss: 0.2537 - val_acc: 0.9370
Epoch 2/5
54000/54000 [==============================] - 2s 30us/sample - loss: 0.2840 - acc: 0.9278 - val_loss: 0.2505 - val_acc: 0.9417
Epoch 3/5
54000/54000 [==============================] - 2s 29us/sample - loss: 0.2651 - acc: 0.9323 - val_loss: 0.2359 - val_acc: 0.9437
Epoch 4/5
54000/54000 [==============================] - 2s 30us/sample - loss: 0.2598 - acc: 0.9340 - val_loss: 0.2526 - val_acc: 0.9430
Epoch 5/5
54000/54000 [==============================] - 2s 30us/sample - loss: 0.2570 - acc: 0.9334 - val_loss: 0.2770 - val_acc: 0.9367
‘’’
# Evaluation
model.evaluate(x_test, y_test)
‘’’
10000/10000 [==============================] - 0s 13us/sample - loss: 0.3257 - acc: 0.9280
[0.32569682545661927, 0.928]
‘’’
```
以上便是使用Keras的一个最基本的示例。可以分几个基本的步骤
1.  数据准备——事实上，在我的日产工作中这部分的代码占了绝大部分的时间……
2. 定义模型——以上示例使用了Sequential API，实际上，在之后将会看到Functional API只是在定义模型的过程上与其不同
3. 编译模型——`model.compile(optimizer=…, loss=…)`是最为核心和简单的一步，这一步中定义了loss以及optimizer
4. 训练模型——`model.fit(x_train, y_train, validation_split=0.1, batch_size=32, epochs=5)`
5. 测试模型——`model.evaluate(x_teset, y_test)`
- - - -
## Keras Functional API
以下将会介绍使用Functional API的方法，最核心的差别在于，使用Functional API可以定义更加灵活的结构，特别是多输入多输出的模型。
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist

# Data
# Omit

# Functional API
inputs = keras.Input((28, 28))
x = layers.Flatten()(inputs)
x = layers.Dense(32, activation=‘relu’)(x)
x = layers.Dense(10, activation=‘softmax’)(x)
model = Model(inputs, x)
model.summary()

# Model Compiling
model.compile(optimizer=‘Adam’, loss=‘sparse_categorical_crossentropy’, metrics=[‘acc’])

# Model Fitting
model.fit(x_train, y_train, …)

# Model Evaluation 

model.evaluate(x_test, y_test)
```
我们可以看到，由于Functional API的引入，我们可以实现两个Sequential API所不能实现的功能。
1. 多输入多输出
2. 层共享
同时，至于其他的操作，包括编译、训练和测试，都是完全不变的。
- - - -
## 更多的一些示例
以下给出两个示例，包括CNN和RNN两种基本模型的建立和训练，分别使用了MNIST和IMDB数据集，其中前者是多分类任务，后者是二分类任务。
### CNN
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist

# Data
# Omit 

# Model
inputs = keras.Input((28, 28))
x = layers.Conv2D(32, 3, activation=‘relu’)(inputs)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation=‘relu’)(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation=‘relu’)(x)
x = layers.Dense(10, activation=‘softmax’)(x)
model = Model(inputs, x)
model.summary()

model.compile(optimizer=‘adma’, loss=‘sparse_categorical_crossentropy’, metrics=[‘acc’])

model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=32)

model.evaluate(x_test, y_test)
```
### RNN
以下给出一个RNN的例子
```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Data
(x_train, y_train), (x_test, y_test) = imdv.load_data(num_words=10000)
x_train = pad_sequences(x_train)
x_test = pad_sequences(x_test)

# Model
inputs = keras.Input((None,))
x = layers.LSTM(32)(inputs)
x = layers.Dense(1, activation=‘sigmoid’)(x)
model = Model(inputs, x)
model.summary()

model.compile(optimizer=‘rmsprop’, loss=‘binary_crossentropy’, metrics=[‘acc’])

model.fit(x_train, y_train, batch_size=32, validation_split=0.1, epochs=5)

model.evaluate(x_test, y_test)
```
- - - -
## 模型的保存和加载
### 保存模型
```python
model.save(‘path/to/save/’) # .hdf5 file the whole model
model.save_weights(‘path/to/save’) # .hdf5 file only the weights 
```
### 模型加载
```python
from tensorflow.keras.models import load_model
model = load_model(‘path/to/save’)

# Or from weights
model = create_model(…)
model.load_weights(‘path/to/save’)
```
- - - -
## Keras Callbacks
Callbacks（回调函数） `keras.Callbacks`是在fit模型的时候可选的输入，可以实现某些比较使用的功能`model.fit(callbacks=[Callbacks])`
### CSVLogger
`Callbacks.CSVLogger(path)`
可以生成csv日志文件，默认为在每一个epoch之后记录训练的情况，包括各种loss以及metrics。
### Tensorboard
`Callbacks.Tensorboard(path)`
可以调用强大的Tensorboard来可视化模型以及查看模型的权重分布等。
### ReduceLROnPlateau
`Callbacks.ReduceLROnPlateau(factor, monitor, patience)`
可以在训练效果持续没有增进的情况下自动减小学习率
### ModelCheckpoint
`Callbacks.ModelCheckpoint(filepath, save_best_only, save_weights_only)`
可以在训练的过程中自动保存模型
- - - -
## 自定义——灵活性的体现
### 自定义层
```python
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_variable(“kernel”,
                                    shape=[int(input_shape[-1]),
                                           self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)

inputs = keras.Input((64,))
outputs = layer(inputs)
model = Model(inputs, outputs)
model.summary()
```
### 类继承方式定义模型
```python
class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, inputs_shape, kernel_size, filters):
    super(ResnetIdentityBlock, self).__init__(name=‘’)
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding=‘same’)
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()
    super(ResnetIdentityBlock, self).build(inputs_shape=inputs_shape)

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)

block = ResnetIdentityBlock((None, 28, 28, 1), 1, [1, 2, 3])
block.summary()
```
### 自定义loss和metrics
```python
def myloss(y_true, y_pred):
	return 
def mymetric(y_true, y_pred):
	return 
```
- - - -
## 总结
本文介绍了Tensorflow的Keras API的基本使用方法，包括Sequential API——可以建立最简单的层叠模型；Functional API——可以建立复杂的多输入输出模型以及共享层权重的模型；子类继承方式建立模型——难度比较大，但灵活性最高。另外，在Keras中Layer也是可以通过子类继承的方式来进行自定义的。在训练的过程中，Callback是一个很有用的工具。
- - - -
## 结语
* Deep learning with python, Francois Chollet 这本书是非常全面的介绍Keras和深度学习基础的书，我非常推荐大家去阅读。
* 大家需要学会去阅读官网上的API的说明，要能够读懂，这实际上需要一种对Keras基本逻辑的认识。
* 大家请多多阅读源代码，会使你对这个框架的理解上升一个层次

