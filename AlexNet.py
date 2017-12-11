# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:18:42 2017

@author: HM-PC

数据:ILSVRC-2010 

"""

import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 1000
IMAGE_SIZE = 227
NUM_CHANNELS = 3

def inference(input_tensor, train, regularizer):
    # conv1  
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [11, 11, 3, 96], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [96], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 4, 4, 1], padding='SAME') 
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    norm1 = tf.nn.lrn(relu1, 5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    #dropout1 = tf.nn.dropout(norm1, 0.5)
    

    # conv2
    with tf.name_scope('conv2') as scope:
        conv2_weights = tf.get_variable("weight", [5, 5, 96, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') 
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    norm2 = tf.nn.lrn(relu2, 5, bias=1.0, alpha=0.0001, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    
    
    # conv3
    with tf.name_scope('conv3') as scope:
        conv3_weights = tf.get_variable("weight", [3, 3, 256, 384], initializer=tf.truncated_normal_initializer(stddev=0.1),)
        conv3_biases = tf.get_variable("bias", [384], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME') 
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    
     # conv4
    with tf.name_scope('conv4') as scope:
        conv4_weights = tf.get_variable("weight", [3, 3, 384, 384], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [384], initializer=tf.constant_initializer(0.0))
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
   
    
