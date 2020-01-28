#!/usr/bin/env python3

# adapted from https://github.com/hcllaw/bdr

from __future__ import division

import numpy as np
import tensorflow as tf

from .base import Network


def _rbf_kernel(X, Y, log_bw):
    with tf.name_scope('rbf_kernel'):
        X_sqnorms_row = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
        Y_sqnorms_col = tf.expand_dims(tf.reduce_sum(tf.square(Y), 1), 0)
        XY = tf.matmul(X, Y, transpose_b=True)
        gamma = 1 / (2 * tf.exp(2 * log_bw))
        return tf.exp(-gamma * (-2 * XY + X_sqnorms_row + Y_sqnorms_col))

def _rbf_kernel_nonstat(X, Y, log_ell_X, log_ell_Y):
    with tf.name_scope('rbf_kernel_nonstat'):
        X_sqnorms_row = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
        Y_sqnorms_col = tf.expand_dims(tf.reduce_sum(tf.square(Y), 1), 0)

        XY = tf.matmul(X, Y, transpose_b=True)

        log_ell_X_col = tf.expand_dims(log_ell_X, 1)
        log_ell_Y_row = tf.expand_dims(log_ell_Y, 0)

        denom = tf.exp(2 * log_ell_X_col) + tf.exp(2 * log_ell_Y_row)
        num = 2 * tf.exp(log_ell_X_col) * tf.exp(log_ell_Y_row)

        return tf.math.multiply(tf.sqrt(num / denom), tf.exp(-1 / denom * (-2 * XY + X_sqnorms_row + Y_sqnorms_col)))


def build_simple_rbf(x_dim, landmarks, bw, reg_out, y_dim = 1,
                     reg_out_bias=0, scale_reg_by_n=False,
                     dtype=tf.float32,
                     init_out=None,
                     init_out_bias=None,
                     opt_landmarks=False,  #whether or not to optimize the landmarks
                     outcome_type='binary'
                     ):

    n_coef = landmarks.shape[0]
    net = Network(x_dim, n_coef, dtype=dtype, y_dim = y_dim)

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
        out = tf.random.normal([n_coef, net.y_dim], dtype=dtype)
    else:
        assert np.size(init_out) == n_coef
        out = tf.constant(np.resize(init_out, [n_coef, net.y_dim]), dtype=dtype)
    params['out'] = tf.Variable(out, name = 'out')

    # initialize the intercept weight randomly, or with ridge output if we have it
    if init_out_bias is None:
        out_bias = tf.random.normal([net.y_dim], dtype=dtype)
    else:
        out_bias = tf.constant(np.resize(init_out_bias, [net.y_dim]), dtype=dtype)
    params['out_bias'] = tf.Variable(out_bias, name = 'out_bias')


    # Compute kernels to landmark points: shape (n_X, n_land)
    kernel_layer = _rbf_kernel(inputs['X'], params['landmarks'], params['log_bw'])

    # Pool bags: shape (n_bags, n_land)
    layer_pool = net.bag_pool_layer(kernel_layer)

    # Output
    out_layer = tf.squeeze(tf.matmul(layer_pool, params['out']) + params['out_bias'])

    if outcome_type == 'binary':
        net.output = tf.sigmoid(out_layer)
    elif outcome_type == 'categorical':
        net.output = tf.nn.softmax(out_layer)
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



def build_spatsep_rbf(x_dim, landmarks, bw, reg_out, y_dim = 1,
                     reg_out_bias=0, scale_reg_by_n=False,
                     dtype=tf.float32,
                     init_out=None,
                     init_out_bias=None,
                     opt_landmarks=False,  #whether or not to optimize the landmarks
                     outcome_type='binary',
                     feat_types=None,
                      feat_agg_type = 'concat'
                     ):

    # Make network
    n_coef = landmarks.shape[0] * (2 if feat_agg_type == 'concat' else 1)  # separate coefs for spatial landmarks and demo / scores
    net = Network(x_dim, n_coef, dtype=dtype, y_dim = y_dim)

    inputs = net.inputs
    params = net.params

    # get inputs
    spat_ind = np.where(feat_types == 'latlong')[0]
    nonspat_ind = np.where(feat_types != 'latlong')[0]

    inputs_spat = tf.slice(inputs['X'], begin = [0,np.min(spat_ind)], size = [-1,len(spat_ind)])
    inputs_nonspat = tf.slice(inputs['X'], begin=[0,np.max(spat_ind) + 1], size = [-1,-1])


    ### Model parameters
    params['landmarks_spat'] = tf.Variable(tf.constant(landmarks[:,spat_ind], dtype=dtype),
                                      trainable=opt_landmarks, name = 'landmarks_spat')
    params['landmarks_nonspat'] = tf.Variable(tf.constant(landmarks[:,nonspat_ind], dtype=dtype),
                                           trainable=opt_landmarks, name='landmarks_nonspat')

    params['log_bw_spat'] = tf.Variable(tf.constant(np.log(bw), dtype=dtype),
                                   trainable=True,
                                   name = 'log_bw_spat')
    params['log_bw_nonspat'] = tf.Variable(tf.constant(np.log(bw), dtype=dtype),
                                   trainable=True,
                                   name='log_bw_nonspat')

    # initialize coefs with ridge coefs
    if init_out is None:
        out = tf.random.normal([n_coef, net.y_dim], dtype=dtype)
    else:
        assert np.size(init_out) == n_coef
        out = tf.constant(np.resize(init_out, [n_coef, net.y_dim]), dtype=dtype)
    params['out'] = tf.Variable(out, name = 'out')

    if init_out_bias is None:
        out_bias = tf.random.normal([net.y_dim], dtype=dtype)
    else:
        out_bias = tf.constant(np.resize(init_out_bias, [net.y_dim]), dtype=dtype)
    params['out_bias'] = tf.Variable(out_bias, name = 'out_bias')


    #### Put layers together
    spat_kernel_layer = _rbf_kernel(inputs_spat, params['landmarks_spat'], params['log_bw_spat'])
    nonspat_kernel_layer = _rbf_kernel(inputs_nonspat, params['landmarks_nonspat'], params['log_bw_nonspat'])

    # combine spat and non-spat
    if feat_agg_type == 'concat':
        kernel_layer = tf.concat([spat_kernel_layer, nonspat_kernel_layer], axis = 1)
    else:
        kernel_layer = tf.add(spat_kernel_layer, nonspat_kernel_layer)


    # Pool bags: shape (n_bags, n_coef)
    layer_pool = net.bag_pool_layer(kernel_layer)

    # Output
    out_layer = tf.squeeze(tf.matmul(layer_pool, params['out']) + params['out_bias'])

    if outcome_type == 'binary':
        net.output = tf.sigmoid(out_layer)
    else:
        net.output = out_layer

    ##### Loss
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



def build_nonstat_rbf(x_dim, landmarks, bw, reg_out, y_dim = 1,
                     reg_out_bias=0, reg_ell_weights = 0,
                      scale_reg_by_n=False,
                     dtype=tf.float32,
                     init_out=None,
                     init_out_bias=None,
                     opt_landmarks=False,  #whether or not to optimize the landmarks
                     outcome_type='binary',
                     feat_types=None,
                      feat_agg_type = 'concat'
                     ):
    # landmarks = args['landmarks']
    # x_dim = train.dim
    n_coef = landmarks.shape[0] * (2 if feat_agg_type == 'concat' else 1)  # separate coefs for spatial landmarks and demo / scores
    net = Network(x_dim, n_coef, dtype=dtype, y_dim = y_dim)

    inputs = net.inputs
    params = net.params

    spat_ind = np.where(feat_types == 'latlong')[0]
    nonspat_ind = np.where(feat_types != 'latlong')[0]

    inputs_spat = tf.slice(inputs['X'], begin = [0,np.min(spat_ind)], size = [-1,len(spat_ind)])
    inputs_nonspat = tf.slice(inputs['X'], begin=[0,np.max(spat_ind) + 1], size = [-1,-1])

    # Model parameters
    params['landmarks_spat'] = tf.Variable(tf.constant(landmarks[:,spat_ind], dtype=dtype),
                                      trainable=opt_landmarks, name = 'landmarks_spat')

    params['landmarks_nonspat'] = tf.Variable(tf.constant(landmarks[:, nonspat_ind], dtype=dtype),
                                              trainable=opt_landmarks, name='landmarks_nonspat')

    params['log_bw_spat'] = tf.Variable(tf.constant(np.log(bw), dtype=dtype),
                                           trainable=True,
                                           name='log_bw_spat')
    params['log_bw_nonspat'] = tf.Variable(tf.constant(np.log(bw), dtype=dtype),
                                           trainable=True,
                                           name='log_bw_nonspat')

    params['ell_weights'] = tf.Variable(tf.random.normal([n_coef, 1], dtype=dtype),
                                        name='ell_weights',
                                        trainable=True)

    # initialize the outcome coefs with ridge regression (if we performed it) and randomly otherwise
    if init_out is None:
        out = tf.random.normal([n_coef, net.y_dim], dtype=dtype)
    else:
        assert np.size(init_out) == n_coef
        out = tf.constant(np.resize(init_out, [n_coef, net.y_dim]), dtype=dtype)
    params['out'] = tf.Variable(out, name = 'out')

    # initialize the intercept weight randomly, or with ridge output if we have it
    if init_out_bias is None:
        out_bias = tf.random.normal([net.y_dim], dtype=dtype)
    else:
        out_bias = tf.constant(np.resize(init_out_bias, [net.y_dim]), dtype=dtype)
    params['out_bias'] = tf.Variable(out_bias, name = 'out_bias')



    #### Put layers together
    K_ell_X = _rbf_kernel(inputs_spat, params['landmarks_spat'], params['log_bw_spat'])
    K_ell_ldmks = _rbf_kernel(params['landmarks_spat'], params['landmarks_spat'], params['log_bw_spat'])

    log_ell_X = tf.squeeze(tf.math.log(tf.matmul(K_ell_X, params['ell_weights'])))
    log_ell_ldmks = tf.squeeze(tf.math.log(tf.matmul(K_ell_ldmks, params['ell_weights'])))

    spat_kernel_layer = _rbf_kernel_nonstat(inputs_spat, params['landmarks_spat'], log_ell_X, log_ell_ldmks)
    nonspat_kernel_layer = _rbf_kernel(inputs_nonspat, params['landmarks_nonspat'], params['log_bw_nonspat'])


    #concatenate or add spatial and non-spatial kernels together
    if feat_agg_type == 'concat':
        kernel_layer = tf.concat([spat_kernel_layer, nonspat_kernel_layer], axis = 1)
    else:
        kernel_layer = tf.add(spat_kernel_layer, nonspat_kernel_layer)

    # Pool bags: shape (n_bags, n_coef)
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
        + reg_ell_weights * tf.nn.l2_loss(params['ell_weights'])
    )

    return net