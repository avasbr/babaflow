import tensorflow as tf
from tensorflow.contrib import slim
from everface.utils import train_utils as tu
import Dumbnet
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys


def main(args):

    # define the parameter and worker servers and the cluster
    parameter_servers = ['localhost:2222']
    workers = ['localhost:2223',
               'localhost:2224',
               'localhost:2225']
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
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            num_classes = 10

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                          trainable=False)

            x_ = tf.placeholder(tf.float32, shape=[None, 784], name='input')
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name='output')
            is_training = tf.placeholder(tf.bool, name='phase_train')

            logits, _ = Dumbnet.inference(
                x_, num_classes, is_training, keep_prob=0.5, weight_decay=5e-3, decay_term=0.95)
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))

            train_op = tf.train.AdamOptimizer(
                1e-4).minimize(loss_op, global_step=global_step)
            acc_op = tu.compute_accuracy(logits, y_)

            init_op = tf.global_variables_initializer()

            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(args.task_index == 0),
                                                   save_checkpoint_secs=30,
                                                   log_step_count_steps=10,
                                                   config=config,
                                                   checkpoint_dir=args.model_dirpath) as sess:
                step = 0
                while not sess.should_stop() and step <= args.num_steps:
                    sess.run(init_op)
                    batch_x, batch_y = mnist.train.next_batch(args.batch_size)
                    _, acc, loss, step = sess.run([train_op, acc_op, loss_op, global_step],
                                                  feed_dict={x_: batch_x, y_: batch_y})
                    if step % args.print_every == 0:
                        print 'Worker : {}, Step: {}, Loss: {}, Accuracy: {}'.format(args.task_index, step, loss, acc)

                # Add test set accuracy


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', default='', help='"ps", "worker"')
    parser.add_arguemnt('--task_index', type=int,
                        default=0, help='Index of task')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Size of the batch for minibatch training')
    parser.add_argument('--num_steps', type=int, default=5000,
                        help='Number of training steps')
    parser.add_argument('--model_dirpath', default='/ml/mnist_models',
                        help='Directory where models live')
    parser.add_argument('--print_every', type=int, default=100)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
