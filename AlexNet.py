# -*-coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:18:42 2017

@author: HM-PC

数据:训练集为55000张图片；验证集为5000张图片；测试集为10000张图片；每张图片大小为28x28；
每张图片精处理为784的一维数组（28x28）；像素矩阵的值为[0,1]之间，0表示白色背景，1表示黑色前景
"""
import tensorflow as tf


def alexnet(input_tensor, train, regularizer):
    # conv1  
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [11, 11, 3, 96], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [96], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 4, 4, 1], padding='SAME') 
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
    #dropout1 = tf.nn.dropout(norm1, 0.5)
    

    # conv2
    with tf.name_scope('conv2') as scope:
        conv2_weights = tf.get_variable("weight", [5, 5, 96, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(norm1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') 
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    
    # conv3
    with tf.name_scope('conv3') as scope:
        conv3_weights = tf.get_variable("weight", [3, 3, 256, 384], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [384], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME') 
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    
     # conv4
    with tf.name_scope('conv4') as scope:
        conv4_weights = tf.get_variable("weight", [3, 3, 384, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME') 
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        
     # conv5
    with tf.name_scope('conv5') as scope:
        conv5_weights = tf.get_variable("weight", [3, 3, 384, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(relu4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME') 
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
    pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    
    
    # fc1
    with tf.name_scope('fc1') as scope:        
        fc1 = tf.reshape(pool5, [-1, tf.Variable(tf.random_normal([4*4*256, 1024])).get_shape().as_list()[0]])
        fc1 = tf.nn.relu(tf.matmul(fc1, tf.Variable(tf.random_normal([4*4*256, 1024])) + tf.Variable(tf.random_normal([1024]), name=scope))) 

    # fc2  
    with tf.name_scope('fc2') as scope:        
        fc2 = tf.nn.relu(tf.matmul(fc1, tf.Variable(tf.random_normal([1024, 1024])) + tf.Variable(tf.random_normal([1024]), name=scope))) 
    
    # fc3  
    with tf.name_scope('fc3') as scope:        
        fc3 = tf.nn.relu(tf.matmul(fc2, tf.Variable(tf.random_normal([1024, 10])) + tf.Variable(tf.random_normal([OUTPUT_NODE]), name=scope)))
    return fc3
   
    
