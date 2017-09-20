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
import pickle as pkl
from tensorflow.python.ops import control_flow_ops


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


def main(args):
    
    # load the graph
    print('Loading the model...')
    saver = tf.train.import_meta_graph('%s/%s'%(args.model_dirpath, args.meta_filename))
    print('Finished!')

    # get placeholders needed specifically for inference
    inputs_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
    labels_placeholder = tf.get_default_graph().get_tensor_by_name('label:0')
    
    # get the logits 
    logits = tf.get_default_graph().get_tensor_by_name('logits/BiasAdd:0')
    xent, acc = create_metrics_ops(logits, labels_placeholder)
    
    # create initializer ops
    global_init_op = tf.global_variables_initializer()

    # load data
    mnist = input_data.read_data_sets(
        '/Users/babasarala/Datasets/MNIST', one_hot=False)

    with tf.Session() as sess:
        sess.run(global_init_op)
        pre_loading_batch_norm_vals = {v.name: sess.run(v) for v in tf.global_variables() if v.name.startswith('fc1/Batch')} 
        saver.restore(sess, '%s/%s'%(args.model_dirpath, args.checkpoint_filename))
        post_loading_batch_norm_vals = {v.name: sess.run(v) for v in tf.global_variables() if v.name.startswith('fc1/Batch')} 

        pkl.dump((pre_loading_batch_norm_vals, post_loading_batch_norm_vals), open('/Users/babasarala/Datasets/MNIST/validation_batch_norm_vals.pkl', 'wb'))


        for i in range(50):
            batch_inputs, batch_labels = mnist.validation.next_batch(128)
            feed_dict = {inputs_placeholder: batch_inputs,
                         labels_placeholder: batch_labels,
                         phase_train_placeholder: False}
            xent_value, acc_value = sess.run([xent, acc], feed_dict=feed_dict)

            print('Cross entropy: %.4f, Validation accuracy: %.4f' %(xent_value, acc_value))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dirpath', default=None, help='Directory to write models')
    parser.add_argument('--meta_filename', default=None, help='Directory to write models for tensorboard')
    parser.add_argument('--checkpoint_filename', default=None, help='Directory to write checkpoint file for tensorboard')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
