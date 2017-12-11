# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:18:42 2017

@author: HM-PC

数据:训练集为55000张图片；验证集为5000张图片；测试集为10000张图片；每张图片大小为28x28；
每张图片精处理为784的一维数组（28x28）；像素矩阵的值为[0,1]之间，0表示白色背景，1表示黑色前景
"""

import tensorflow as tf


def lenet5(input_tensor, train, regularizer):
    #输入为32x32，过滤器边长为5，深度为32，移动步长为1且使用全0填充，输出为28x28x32
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME') 
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    
    #输入为28x28x32，过滤器为全0填充且移动步长为2，输出为14x14x32
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME") 
        
    #输入为14x14x32，过滤器边长为5，深度为64，移动步长为1且使用全0填充，输出为14x14x64
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    #输入为14x14x64，输出为7x7x64
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
   
    #输入为3136的向量；输出为512的向量
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: 
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: 
            fc1 = tf.nn.dropout(fc1, 0.5)
    
    #输入为512的向量；输出为10的向量
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: 
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


