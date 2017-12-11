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


def evaluate(model, data):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}  #导入测试数据
        y = inference(x, None)
       
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
   
                
                


