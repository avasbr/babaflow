from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from tensorflow.python.ops import control_flow_ops
import pickle as pkl
import Dumbnet


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


def save_metagraph(sess, saver, model_dirpath, model_name):
    metagraph_filename = os.path.join(
        model_dirpath, 'model-%s.meta' % model_name)
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        saver.export_meta_graph(metagraph_filename)


def save_checkpoint_filepath(sess, saver, model_dirpath, model_name, step):
    print('Saving checkpoint file')
    checkpoint_path = os.path.join(
        model_dirpath, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)


def main(args):

    # construct graph
    inputs_placeholder = tf.placeholder(tf.float32, [None, 784], name='input')
    labels_placeholder = tf.placeholder(tf.int32, [None], name='label')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    # create the training ops
    logits, _ = Dumbnet.inference(inputs_placeholder, phase_train_placeholder)
    xent, acc = create_metrics_ops(logits, labels_placeholder)
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    # optimizer = tf.train.AdamOptimizer(
    #    learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=0.1)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    train_op = slim.learning.create_train_op(
        xent, optimizer, global_step=global_step)

    # add summaries for BN variables
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('cross_entropy', xent)
    for v in tf.global_variables():
        if v.name.startswith('conv1/Batch') or v.name.startswith('conv2/Batch') or \
                v.name.startswith('fc1/Batch') or v.name.startswith('logits/Batch'):
            print(v.name)
            tf.summary.histogram(v.name, v)
    merged_summary_op = tf.summary.merge_all()

    # create initializer ops
    global_init_op = tf.global_variables_initializer()

    # load data
    mnist = input_data.read_data_sets(
        '/Users/babasarala/Datasets/MNIST', one_hot=False)

    # create saver
    saver = tf.train.Saver(max_to_keep=50)
    model_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    with tf.Session() as sess:
        save_metagraph(sess, saver, args.model_dirpath, model_name)
        sess.run(global_init_op)

        pre_training_batch_norm_vals = {v.name: sess.run(
            v) for v in tf.global_variables() if v.name.startswith('fc1/Batch')}

        # establish training and validation
        train_writer = tf.summary.FileWriter(
            args.event_log_dirpath, sess.graph)

        for i in range(200):
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
                print('Saving checkpoint filepath...')
                save_checkpoint_filepath(
                    sess, saver, args.model_dirpath, model_name, global_step)
                print('Finished!')

        post_training_batch_norm_vals = {v.name: sess.run(
            v) for v in tf.global_variables() if v.name.startswith('fc1/Batch')}

    pkl.dump((pre_training_batch_norm_vals, post_training_batch_norm_vals), open(
        '/Users/babasarala/Datasets/MNIST/train_batch_norm_values.pkl', 'wb'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dirpath', default=None,
                        help='Directory to write models')
    parser.add_argument('--event_log_dirpath', default=None,
                        help='Directory to write models for tensorboard')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
