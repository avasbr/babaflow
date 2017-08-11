from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops


def inference(inputs, is_training, keep_prob=0.8):
    endpoints = {}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.95}):

        inputs_reshaped = tf.reshape(inputs, [-1, 28, 28, 1])

        net = slim.conv2d(inputs_reshaped, 16, [5, 5], scope='conv1')
        endpoints['conv1'] = net

        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        endpoints['pool1'] = net

        net = slim.conv2d(net, 32, [5, 5], scope='conv2')
        endpoints['conv2'] = net

        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        endpoints['pool2'] = net

        net = slim.flatten(net)

        net = slim.fully_connected(net, 1024, scope='fc1')
        endpoints['fc1'] = net

        net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training,
                           scope='dropout')
        endpoints['dropout'] = net

        logits = slim.fully_connected(
            net, 10, activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='logits')
        return logits, endpoints


def create_metrics_ops(logits, labels, num_classes=10):

    with tf.name_scope('xent'):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        xent_mean = tf.reduce_mean(xent)

    with tf.name_scope('accuracy'):
        acc = compute_accuracy(logits, labels)

    return xent_mean, acc


def compute_accuracy(logits, labels):
    labels = tf.cast(labels, tf.int32)
    correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), labels)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc


def main():

    # construct graph
    inputs_placeholder = tf.placeholder(tf.float32, [None, 784])
    labels_placeholder = tf.placeholder(tf.int32, [None])
    phase_train_placeholder = tf.placeholder(tf.bool)

    # create the training ops
    logits, _ = inference(inputs_placeholder, phase_train_placeholder)
    xent, acc = create_metrics_ops(logits, labels_placeholder)
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0.0), trainable=False)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=0.1)
    train_op = slim.learning.create_train_op(
        xent, optimizer, global_step=global_step)

    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # if update_ops:
    #    updates = tf.group(*update_ops)
    #    xent = control_flow_ops.with_dependencies([updates], xent)

    # add summaries for BN variables
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('cross_entropy', xent)
    for v in tf.all_variables():
        if v.name.startswith('conv1/Batch') or v.name.startswith('conv2/Batch') or \
                v.name.startswith('fc1/Batch') or v.name.startswith('logits/Batch'):
            print(v.name)
            tf.summary.histogram(v.name, v)
    merged_summary_op = tf.summary.merge_all()

    # create initializer ops
    global_init_op = tf.global_variables_initializer()

    # load data
    mnist = input_data.read_data_sets(
        '/Users/babasarala/Desktop/mnist', one_hot=False)

    # create saver
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=300)

    with tf.Session() as sess:
        sess.run(global_init_op)

        # establish training and validation
        train_writer = tf.summary.FileWriter(
            '/Users/babasarala/Desktop/mnist/events/train', sess.graph)

        for i in range(10000):
            batch_inputs, batch_labels = mnist.train.next_batch(128)
            feed_dict = {inputs_placeholder: batch_inputs,
                         labels_placeholder: batch_labels,
                         phase_train_placeholder: True}
            _, xent_value, acc_value, global_step_value, summary_value = sess.run(
                [train_op, xent, acc, global_step, merged_summary_op], feed_dict=feed_dict)

            print('Step: %i, Cross entropy: %.4f, Training accuracy: %.4f' %
                  (global_step_value, xent_value, acc_value))

            # for each batch, write summary statistics
            train_writer.add_summary(summary_value, global_step_value)

            # every so often, check the training accuracy for inference
            if global_step_value % 50 == 0:
                batch_inputs, batch_labels = mnist.validation.next_batch(128)
                feed_dict = {inputs_placeholder: batch_inputs,
                             labels_placeholder: batch_labels,
                             phase_train_placeholder: False}
                xent_value, acc_value = sess.run(
                    [xent, acc], feed_dict=feed_dict)
                print('Cross entropy: %.4f, Validation accuracy: %.4f' %
                      (xent_value, acc_value))
                saver.save(sess, '/Users/babasarala/Desktop/mnist/models')


if __name__ == '__main__':
    main()
