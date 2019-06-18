---
title:      Installation of Tensorflow and GPU support
date:       2019-06-18
---
In this post, you will see how to install tensorflow and GPU support. I assume that you don't have a root permision.

Of course, this post is just a test file for my Github Pages. You can find a lot of articles like this from Google.

Here is some details about my OS and environment.
* CentOS 7 (BTW, here I only talk about linux system.)
* anaconda3 Anaconda is a goog tool to manage your enviroment. You can get more information in its [offcial website](https://www.anaconda.com/)
* nvidia-driver 410.79 Dependence for [CUDA](https://developer.nvidia.com/cuda-toolkit) and [CuDNN](https://developer.nvidia.com/cudnn)

## Create a new enviroment
* I suppose that you have already had anaconda3 installed. Then, you can run the command below to create an enviroment, where the name of it is tensorflow and version of python we use is 3.7.
```shell
conda create --name tensorflow python=3.7
```
After that you have created a new enviroment. Then you can activate it and do something.

## Install Tensorflow
1. First, activate the enviroment created before by runing this command in your terminal.
```shell
conda activate tensorflow
```
2. Second, use `pip` to install tensorflow for CPU only.
```shell
pip install tensorflow
```
And now, if you only want to use CPU for tensorflow, you have finished.

If you want to use a GPU, just change it to this.
```shell
pip install tensorflow-gpu
```

## GPU support
There is something a little troublesome that you need CUDA and CuDNN to make GPU support for your tensorflow-gpu.

### Install CUDA
Make sure that the versions of Tensorflow and CUDA are compatible, otherwise you will have much trouble. To check if the version of CUDA you have is campatible with your Tensorflow, go to the offical website of [Tensorflow](https://www.tensorflow.org/) for more information.

As for the installation of CUDA, you will get installation file from [offcial website](https://developer.nvidia.com/cuda-toolkit). Just run it and follow the guidance.

After CUDA has been installed, remember to add the directory of CUDA to your PATH and LD_LIBRARY_PATH. Of course, you can change .bashrc file in your HOME directory. But do you remember anaconda? It will be a better choice that you do this by anaconda.

Here are the details.

1. Fisrt, change the directory to tensorflow enviroment. (Here, my anaconda3 is justed in the HOME directory.)

```shell
cd ~/anaconda3/envs/tensorflow
```

2. Second, make some directory and go into it.
```shell
mkdir -p etc/conda/activate.d
cd etc/conda/activate.d
```

3. Here, create a file `env_var.sh` and write something as below.
```shell
#!/bin/sh
# CUDA10.0
export PATH=~/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=~/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
```
**Note: you should remember where you install your CUDA. Here, for me, it's in `~/local/cuda-10.0`. Don't copy completely.**

#### Some explanation about nvidia-driver
Every version of CUDA needs a specific lowest version of nvidia-driver. To install a new nvidia-driver, you need a root permission. So, if you find that the version of nvidia-driver is not high enough, just contact your administrators for help.

### Install CuDNN
Installation of CuDNN is much easier than that of CUDA because anaconda allows you to do it by using just one command.

1. First, activate your enviroment.
```shell
conda activate tensorflow
```

2. Second, search cudnn using conda.
```shell
conda search cudnn
```

3. Finally, choose the version and build you need and install it.
```shell
conda install cudnn=7.3.1=cuda10.0_0
```

Now, CuDNN has been installed.

## Have a test
Make sure the tensorflow enviroment has been activated and then run python in your termial.

```shell
python
```

In python console, run

```python
import tensorflow as tf
tf.test.is_gpu_available()
```

If it returns True, congratulations, everying's fine!
