from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(images, n_hid_1, n_hid_2):
    '''Build the MNIST model up to where it may be used for inference

    Parameters
    ----------
    images: images placeholder, from inputs()
    n_hid_1: size of the first hidden layer
    n_hid_2: size of the second hidden layer

    Returns
    -------
    softmax_linear: output tensor with the computed logits
    '''

    def fully_connected_layer(X, W_shape, b_shape):
        W = tf.get_variable(name='weights',
                            shape=W_shape,
                            initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                        stddev=1.0/math.sqrt(float(W_shape[0]))))
        b = tf.get_variable(name='biases',
                                 shape=b_shape,
                                 initializer=tf.zeros_initializer())
        return tf.matmul(X, W) + b

    # Hidden 1
    with tf.variable_scope('hidden_1'):
        pre_activations = fully_connected_layer(
            images, [IMAGE_PIXELS, n_hid_1], [n_hid_1])
        hidden_1 = tf.nn.relu(pre_activations)
    # Hidden 2
    with tf.variable_scope('hidden_2'):
        pre_activations = fully_connected_layer(
            hidden_1, [n_hid_1, n_hid_2], [n_hid_2])
        hidden_2 = tf.nn.relu(pre_activations)

    with tf.variable_scope('softmax_linear'):
        logits = fully_connected_layer(
            hidden_2, [n_hid_2, NUM_CLASSES], [NUM_CLASSES])

    return logits

def old_inference(images, n_hid_1, n_hid_2):
    '''Build the MNIST model up to where it may be used for inference

    Parameters
    ----------
    images: images placeholder, from inputs()
    n_hid_1: size of the first hidden layer
    n_hid_2: size of the second hidden layer

    Returns
    -------
    softmax_linear: output tensor with the computed logits
    '''

    # Hidden 1
    with tf.name_scope('hidden_1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, n_hid_1],
                                stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),
            name='weights')

        biases = tf.Variable(tf.zeros([n_hid_1]), name='biases')
        hidden_1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    # Hidden 2
    with tf.name_scope('hidden_2'):
        weights = tf.Variable(
            tf.truncated_normal([n_hid_1, n_hid_2],
                                stddev=1.0/math.sqrt(float(n_hid_1))),
            name='weights')
        biases = tf.Variable(tf.zeros([n_hid_2]), name='biases')
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, weights) + biases)

    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([n_hid_2, NUM_CLASSES],
                                stddev=1.0/math.sqrt(float(n_hid_2))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden_2, weights) + biases

    return logits


def loss(logits, labels):

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):

    tf.summary.scalar('training_loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

# def evaluation(logits, labels):
#    eval_loss = loss(logits, labels)
#    tf.summary.scalar('evaluation_loss', loss)
#    return


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
