---
title:      Tensorflow安装以及GPU支持 				# 标题
subtitle:   Tensorflow CUDA CuDNN #副标题
date:       2019-03-13 				# 时间
---

# Tensorflow安装以及GPU支持
----
本文如题，介绍Tensorflow的安装以及添加GPU支持。并且假设你不具有root权限。

以下为本人工作环境的一些简单罗列
* CentOS 7 这个是操作系统，本文只涉及Linux系统
* anaconda3 Anaconda是一个非常棒的环境管理和配置系统，其使用方法可以自行Google
* nvidia-driver 410.79 这个是CUDA以及CuDNN的依赖
* glibc 2.17 事实上，在我安装的过程中出现过一些问题与之相关

## 预先准备
* Anaconda3新建一个环境，名字可以随意，比如我这里叫做tf2.0，可以使用如下命令
```shell
conda create --name tf2.0 python=3.7
```
* 了解好自己要安装的Tensorflow的版本，它支持什么版本的python。
比如，这里即将安装的tf-nightly-gpu-2.0-preview，它可以支持python3.7；一些旧版本的Tensorflow，例如1.12或者之前的版本，我记得貌似只支持到python3.6。这些信息可以到[Tensorflow官网](https://www.tensorflow.org/)上去查看，也可以到[PyPI](https://pypi.org/)上去查看
* 如果需要GPU支持的话，你还需要了解你所安装的Tensorflow其所对应的CUDA版本以及CuDNN版本

## 安装Tensorflow
1. 首先进入你刚刚创建的环境
```shell
conda activate tf2.0
```
2. 安装某一个版本的Tensorflow，这里我选择安装了tf-nightly-gpu-2.0-preview
```shell
pip install tf-nightly-gpu-2.0-preview
```
如果只需要安装CPU版本的话，就执行
```shell
pip install tf-nightly-2.0-preview
```
至此，Tensorflow的安装完毕了，如果想安装不同版本的Tensorflow的话，只需更改`pip install`后面的内容，例如
```shell
pip install tensorflow-gpu==1.12
```
表示安装1.12版本的Tensorflow，且支持GPU
```shell
pip install tensorflow==1.12
```
表示安装1.12版本的Tensorflow，不支持GPU

## 添加GPU支持
如果安装了GPU版本的Tensorflow的话，则必须添加GPU的相关支持才能够正常运行。事实上，你只需要安装两样东西，一个是CUDA，另一个是CuDNN。

### 安装CUDA
首先，还是那句话，你需要知道你所安装的Tensorflow对应着什么版本的CUDA，这个十分重要，而且大多数的时候并不是向下兼容的，即新版本的CUDA不一定会支持旧版本的Tensorflow，所以CUDA的版本不是越新越好，而是要“恰到好处”。这些信息你可以从[Tensorflow官网](Tensorflow官网)获得。

这里，tf-nightly-gpu-2.0-preview需要的是CUDA10.0，可以从[CUDA官网](https://developer.nvidia.com/cuda-downloads)下载所需要的CUDA版本的安装文件，并且按照网站上的说明进行操作即可，这是将非常简单的过程。

在安装完CUDA之后，还需要做的一件事情是修改添加CUDA的路径到环境变量，你可以直接修改.bashrc文件，但更为明智的选择是修改当前conda环境的环境变量，具体做法如下。

1. 先切入到当前conda环境的路径之下，在我这里是~/anaconda3/envs/tf2.0/
```shell
cd ~/anaconda3/envs/tf2.0
```
2. 创建并进入如下的路径
```shell
mkdir -p etc/conda/activate.d
cd etc/conda/activate.d
```
3. 在此路径下建立一个新的文件`env_var.sh`并写入如下内容
```shell
#!/bin/sh
# CUDA10.0
export PATH=~/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=~/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
```
**注意：这里，我的cuda10.0安装在了`~/local/cuda-10.0`这个路径之下，你只需要模仿我这样的方式配置就可以了**

#### 关于nvidia-driver版本一些说明
每一个CUDA版本对应着一个nvidia-driver的下限，即你的nvidia-driver至少要达到一个最低标准，才能保证成功安装并运行你所需要的CUDA版本，如果nvidia-dirver的版本不够的话，是需要进行升级的，而升级nvidia-driver需要root权限。

所以，如果你发现你的nvidia-driver版本过低的话，你联系管理员求助吧。

#### 关于glibc的一些问题
在我安装Tensorflow2.0的过程当中，出现了一个问题
```
ImportError: /lib64/libm.so.6: version `GLIBC_2.23' not found
```
这说明我的系统的glibc的版本不够高，可以使用
```
ldd --version
```
来查看glibc的版本
```
ldd (GNU libc) 2.17
Copyright (C) 2012 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
由 Roland McGrath 和 Ulrich Drepper 编写。
```
可以看到，我的glibc版本是2.17，而所需要的是2.23，为了解决这个问题，一般来说，我们是需要升级glibc的，然而这个往往需要root权限才能够完成。

对于此事，我们可以去[glibc官网](https://ftp.gnu.org/gnu/glibc/)下载所需要版本的glibc源代码压缩包，解压之后进行编译，至于如何进行编译可以自行Google，这个我相信会有很多的资料，在此就不赘述了。

对于编译之后的结果，我们可以在编译完成之后的路径当中看到`lib/`这个路径，在这个路径当中复制四个我们需要的文件，包括
`libm.so.6`、`libm-2.23.so`、`libm.so`、`libm.a`，到环境的库路径`~/anaconda3/envs/tf2.0/lib`下即可。
### 安装CuDNN
安装CuDNN相比于安装CUDA将会是一个更加简单的过程，因为conda允许你一键安装某一个版本的CuDNN，具体操作如下。

进入你的环境，
```shell
conda activate tf2.0
```

在conda中搜索cudnn
```shell
conda search cudnn
```
你应该会看到类似于下面的结果
![conda search cudnn](https://ws4.sinaimg.cn/large/006tKfTcly1g118xtvjvej30vz0u0doe.jpg)

选择你所需要的CuDNN的Version和Build，这里我需要安装的是Version为7.3.1以及Build为cuda10.0_0的cudnn
```shell
conda install cudnn=7.3.1=cuda10.0_0
```

至此，cudnn便安装完毕了。
