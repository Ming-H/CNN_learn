# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:18:42 2017

@author: HM-PC

数据:训练集为55000张图片；验证集为5000张图片；测试集为10000张图片；每张图片大小为28x28；
每张图片精处理为784的一维数组（28x28）；像素矩阵的值为[0,1]之间，0表示白色背景，1表示黑色前景
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.chdir("F:/project/tensorflow_learn")
import numpy as np
import cnn_train
import cnn_test

if __name__ == '__main__':
    mnist = input_data.read_data_sets("F:/datasets/MNIST_data", one_hot=True)
    cnn_train.train(model, mnist)
    cnn_test.evaluate(model,mnist)