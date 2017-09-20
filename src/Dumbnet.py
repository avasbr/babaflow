from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.slim as slim


def inference(inputs, num_classes, is_training, keep_prob=0.5, weight_decay=5e-3, decay_term=0.95):
    endpoints = {}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_params={'is_training': is_training,
                                           'decay': decay_term}):

        inputs_reshaped = tf.reshape(inputs, [-1, 28, 28, 1])

        net = slim.conv2d(inputs, 16, [5, 5], scope='conv1')
        endpoints['conv1'] = net

        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        endpoints['pool1'] = net

        net = slim.conv2d(net, 32, [5, 5], scope='conv2')
        endpoints['conv2'] = net

        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        endpoints['pool2'] = net

        net = slim.flatten(net)

        net = slim.fully_connected(net, 128, scope='bottleneck')
        endpoints['bottleneck'] = net

        net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training,
                           scope='dropout')
        endpoints['dropout'] = net

        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='logits')

        return logits, endpoints