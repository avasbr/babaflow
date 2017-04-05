from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import os.path
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import mnist_input
import mnist

FLAGS = tf.app.flags.FLAGS

# Basic model parameters
tf.app.flags.DEFINE_integer(
    'batch_size', 128, '''Number of images to process in a batch.''')
tf.app.flags.DEFINE_string(
    'data_dirpath', '/Users/babasarala/repos/babaflow/data/mnist_data',
    '''Path to the MNIST dataset''')
tf.app.flags.DEFINE_integer(
    'hidden_1', 100, '''Number of nodes in the first hidden layer''')
tf.app.flags.DEFINE_integer(
    'hidden_2', 50,  '''Number of nodes in the second hidden layer''')
tf.app.flags.DEFINE_float(
    'learning_rate', 0.01, '''Learning rate for SGD''')
tf.app.flags.DEFINE_string(
    'log_dirpath', '/Users/babasarala/repos/babaflow/logging/mnist_logging',
    '''Path to logs directory''')
tf.app.flags.DEFINE_integer(
    'max_steps', 5000, '''Maximum number of iterations''')


def placeholder_inputs(batch_size):
    ''' Generate placeholder variables to represent the input tensors '''
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    return images_placeholder, labels_placeholder


def fill_feed_dict(dataset, images_pl, labels_pl):
    ''' Fills the feed_dict for training the given step '''
    images_feed, labels_feed = dataset.next_batch(FLAGS.batch_size)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
    }
    return feed_dict


def do_eval(sess, eval_correct, images_pl, labels_pl, dataset):
    true_count = 0  # counts the number of predictions
    steps_per_epoch = dataset.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(dataset, images_pl, labels_pl)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    ''' Train MNIST for a number of steps '''
    dataset = mnist_input.read_data_sets(FLAGS.data_dirpath)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Generate placeholders for the images and labels
            images_placeholder, labels_placeholder = placeholder_inputs(
                FLAGS.batch_size)

            # Add forward pass op
            logits = mnist.inference(
                images_placeholder, FLAGS.hidden_1, FLAGS.hidden_2)

            # Add loss function op
            loss = mnist.loss(logits, labels_placeholder)

            # Add in the training operation
            train_op = mnist.training(loss, FLAGS.learning_rate)

            # Add op to compare logics to the labels
            eval_correct = mnist.evaluation(logits, labels_placeholder)

            # Build the summary Tensor based on the TF collection of Summaries
            summary = tf.summary.merge_all()

            # Add the variable initializer
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints
            saver = tf.train.Saver()

            # Instantiate a SummaryWriter to output summaries and the Graph
            summary_writer = tf.summary.FileWriter(
                FLAGS.log_dirpath, sess.graph)

            # Run the Op to initailze the variables
            sess.run(init)

            for step in xrange(FLAGS.max_steps):

                start_time = time.time()

                # update the next batch of images to feed in, and train
                # off of these examples
                feed_dict = fill_feed_dict(
                    dataset.train, images_placeholder, labels_placeholder)
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict=feed_dict)

                duration = time.time() - start_time

                # Write the summaries and print an overfiew

                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.2f sec)' %
                          (step, loss_value, duration))
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically
                if (step + 1) % 5000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_file = os.path.join(
                        FLAGS.log_dirpath, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    print('Training Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            dataset.train)
                    print('Validation Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            dataset.validation)
                    print('Test Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            dataset.test)


def main():
    if tf.gfile.Exists(FLAGS.log_dirpath):
        tf.gfile.DeleteRecursively(FLAGS.log_dirpath)
    tf.gfile.MakeDirs(FLAGS.log_dirpath)
    run_training()

if __name__ == '__main__':
    main()
