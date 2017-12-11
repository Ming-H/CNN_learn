# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:18:42 2017

@author: HM-PC

数据:训练集为55000张图片；验证集为5000张图片；测试集为10000张图片；每张图片大小为28x28；
每张图片精处理为784的一维数组（28x28）；像素矩阵的值为[0,1]之间，0表示白色背景，1表示黑色前景
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99


def train(model, data, model_name, model_save_path="/model"):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [BATCH_SIZE, model.IMAGE_SIZE, model.IMAGE_SIZE, model.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, model.OUTPUT_NODE], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = model.inference(x,False,regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, data.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = data.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (BATCH_SIZE, model.IMAGE_SIZE, model.IMAGE_SIZE, model.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value)) 
                #saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)

                
