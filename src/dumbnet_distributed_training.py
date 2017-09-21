import tensorflow as tf
from tensorflow.contrib import slim
import Dumbnet
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys


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

    # define the parameter and worker servers and the cluster
    parameter_servers = ['localhost:2222']
    workers = ['localhost:2223',
               'localhost:2224']
    cluster = tf.train.ClusterSpec({'ps': parameter_servers,
                                    'worker': workers})
    server = tf.train.Server(cluster,
                             job_name=args.job_name,
                             task_index=args.task_index)

    if args.job_name == 'ps':
        server.join()

    elif args.job_name == 'worker':
        with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % args.task_index,
                                                      cluster=cluster)):
            # Building the model...
            global_step = tf.contrib.framework.get_or_create_global_step()

            # set up the data
            mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
            num_classes = 10
            input_placeholder = tf.placeholder(
                tf.float32, shape=[None, 784], name='input')
            labels_placeholder = tf.placeholder(
                tf.int32, shape=[None], name='output')
            phase_train_placeholder = tf.placeholder_with_default(
                tf.constant(True), shape=(), name='phase_train')

            logits, _ = Dumbnet.inference(
                input_placeholder, num_classes, is_training=phase_train_placeholder, keep_prob=0.5, weight_decay=5e-3, decay_term=0.95)
            opt = tf.train.AdamOptimizer(
                1e-2, beta1=0.9, beta2=0.999, epsilon=0.1)
            loss_op, acc_op = create_metrics_ops(logits, labels_placeholder)
            train_op = slim.learning.create_train_op(
                loss_op, opt, global_step=global_step)
            init_op = tf.global_variables_initializer()

            hooks = [tf.train.StopAtStepHook(last_step=args.num_steps)]
            print 'Starting the training...'
            # , log_device_placement=False,
            config = tf.ConfigProto(allow_soft_placement=True)
            # device_filters=["/job:ps", "/job:worker/task:%d" %
            # args.task_index])
            step = 0
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(
                                                       args.task_index == 0),
                                                   checkpoint_dir=args.model_dirpath,
                                                   save_checkpoint_secs=30,
                                                   log_step_count_steps=10,
                                                   config=config,
                                                   hooks=hooks) as sess:
                while not sess.should_stop():
                    sess.run(init_op)
                    batch_x, batch_y = mnist.train.next_batch(args.batch_size)
                    _, acc, loss, step = sess.run([train_op, acc_op, loss_op, global_step],
                                                  feed_dict={input_placeholder: batch_x, labels_placeholder: batch_y})
                    if step % args.print_every == 0:
                        print 'Worker : {}, Step: {}, Loss: {}, Accuracy: {}'.format(args.task_index, step, loss, acc)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', default='', help='"ps", "worker"')
    parser.add_argument('--task_index', type=int,
                        default=0, help='Index of task')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Size of the batch for minibatch training')
    parser.add_argument('--num_steps', type=int, default=5000,
                        help='Number of training steps')
    parser.add_argument('--model_dirpath', default='/home/bhargav/models',
                        help='Directory where models live')
    parser.add_argument('--print_every', type=int, default=50)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
