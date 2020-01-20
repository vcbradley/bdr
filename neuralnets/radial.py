# adapted from https://github.com/hcllaw/bdr

from __future__ import division

import numpy as np
import tensorflow as tf

from neuralnets.base import Network


def _rbf_kernel(X, Y, log_bw):
    with tf.name_scope('rbf_kernel'):
        X_sqnorms_row = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
        Y_sqnorms_col = tf.expand_dims(tf.reduce_sum(tf.square(Y), 1), 0)
        XY = tf.matmul(X, Y, transpose_b=True)
        gamma = 1 / (2 * tf.exp(2 * log_bw))
        return tf.exp(-gamma * (-2 * XY + X_sqnorms_row + Y_sqnorms_col))


def build_radial_net(net, landmarks, bw, reg_out,
                     reg_out_bias=0, scale_reg_by_n=False,
                     dtype=tf.float32,
                     init_out=None,
                     init_out_bias=None,
                     opt_landmarks=False,  #whether or not to optimize the landmarks
                     outcome_type='binary'
                     ):

    inputs = net.inputs
    params = net.params

    # Model parameters
    params['landmarks'] = tf.Variable(tf.constant(landmarks, dtype=dtype),
                                      trainable=opt_landmarks, name = 'landmarks')

    params['log_bw'] = tf.Variable(tf.constant(np.log(bw), dtype=dtype),
                                   trainable=True,  # we always want to train the bandwitdth param, not only when we're optimizing landmarks
                                   name = 'log_bw')

    # initialize the outcome coefs with ridge regression (if we performed it) and randomly otherwise
    if init_out is None:
        out = tf.random.normal([n_land, 1], dtype=dtype)
    else:
        assert np.size(init_out) == n_land
        out = tf.constant(np.resize(init_out, [n_land, 1]), dtype=dtype)
    params['out'] = tf.Variable(out, name = 'out')

    # initialize the intercept weight randomly, or with ridge output if we have it
    if init_out_bias is None:
        out_bias = tf.random.normal([1], dtype=dtype)
    else:
        out_bias = tf.constant(init_out_bias, shape=(), dtype=dtype)
    params['out_bias'] = tf.Variable(out_bias, name = 'out_bias')


    # Compute kernels to landmark points: shape (n_X, n_land)
    kernel_layer = _rbf_kernel(inputs['X'], params['landmarks'], params['log_bw'])

    # Pool bags: shape (n_bags, n_land)
    layer_pool = net.bag_pool_layer(kernel_layer)

    # Output
    out_layer = tf.squeeze(tf.matmul(layer_pool, params['out']) + params['out_bias'])

    if outcome_type == 'binary':
        net.output = tf.sigmoid(out_layer)
    else:
        net.output = out_layer

    # Loss
    net.early_stopper = tf.reduce_mean(tf.square(net.output - inputs['y']))

    if scale_reg_by_n:
        n = tf.cast(tf.squeeze(tf.shape(net.output), [0]), dtype)
        reg_out /= n
        reg_out_bias /= n

    net.loss = (
        net.early_stopper   #early stopping penalty contribution for regularization
        + reg_out * tf.nn.l2_loss(params['out'])
        + reg_out_bias * tf.nn.l2_loss(params['out_bias'])
    )

    return net
