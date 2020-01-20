from __future__ import division, print_function
from functools import partial
import os
import sys

import numpy as np
import pandas as pd
import importlib as imp
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.utils import check_random_state

from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf

from neuralnets.features import Features
from neuralnets.base import Network
from neuralnets.radial import build_radial_net
from neuralnets.train import eval_network, train_network
from neuralnets.utils import get_median_sqdist, tf_session
from neuralnets.utils import loop_batches

from rpy2.robjects import r, pandas2ri
pandas2ri.activate()




def _split_feats(args, feats, labels=None, groups=None):
    if groups is None:
        ss = ShuffleSplit
    else:
        ss = GroupShuffleSplit

    rs = check_random_state(args['split_seed'])
    test_splitter = ss(
        1, train_size=args['trainval_size'], test_size=args['test_size'],
        random_state=rs)
    (trainval, test), = test_splitter.split(feats, None, None)

    val_splitter = ss(
        1, train_size=args['train_estop_size'], test_size=args['val_size'],
        random_state=rs)
    X_v = feats[trainval]
    y_v = None if labels is None else labels[trainval]
    g_v = None if groups is None else groups[trainval]
    (train_estop, val), = val_splitter.split(X_v, y_v, g_v)

    estop_splitter = ss(
        1, train_size=args['train_size'], test_size=args['estop_size'],
        random_state=rs)
    X = X_v[train_estop]
    y = None if labels is None else y_v[train_estop]
    g = None if groups is None else g_v[train_estop]
    (train, estop), = estop_splitter.split(X, y, g)
    return X[train], X[estop], X_v[val], feats[test]

def pick_landmarks(args, train):
    train.make_stacked()
    rs = check_random_state(args.landmark_seed)
    if args.kmeans_landmarks:
        kmeans = KMeans(n_clusters=args.n_landmarks, random_state=rs, n_jobs=1)
        kmeans.fit(train.stacked_features)
        return kmeans.cluster_centers_
    else:
        w = rs.choice(train.total_points, args.n_landmarks, replace=False)
        return train.stacked_features[w]


def make_network(args, train):
    kw = {'in_dim': train.dim
        , 'reg_out': args['reg_out']
        , 'reg_out_bias': args['reg_out_bias']
        , 'scale_reg_by_n': args['scale_reg_by_n']
        , 'dtype': tf.float64 if args['dtype_double'] else tf.float32
          }

    kw['bw'] = np.sqrt(get_median_sqdist(train) / 2) #* args['bw_scale']

    # get landmarks
    #kw['landmarks'] = landmarks = pick_landmarks(args, train)
    kw['landmarks'] = args['landmarks']
    kw['opt_landmarks'] = args['opt_landmarks']

    if args['type'] == 'radial':
        net = build_radial_net(**kw)
    else:
        raise Exception(args['type'] + ' not recognized type of network')

    return net



def train_net(sess, args, net, train, val):
    optimizer = {
        'adam': tf.compat.v1.train.AdamOptimizer,
        'sgd': tf.compat.v1.train.GradientDescentOptimizer,
    }[args['optimizer']]
    train_network(sess, net, train, val,
                  os.path.join(args['out_dir'], 'checkpoints/model'),
                  batch_pts=args['batch_pts'], batch_bags=args['batch_bags'],
                  eval_batch_pts=args['eval_batch_pts'],
                  eval_batch_bags=args['eval_batch_bags'],
                  max_epochs=args['max_epochs'],
                  first_early_stop_epoch=args['first_early_stop_epoch'],
                  optimizer=optimizer,
                  lr=args['learning_rate'])

def eval_network(sess, net, test_f, batch_pts, batch_bags=np.inf, do_var=False):
    preds = np.zeros_like(test_f.y)
    if do_var:
        pred_vars = np.zeros_like(test_f.y)
    i = 0
    for batch in loop_batches(test_f, max_pts=batch_pts, max_bags=batch_bags,
                              stack=True, shuffle=False):
        d = net.feed_dict(batch, training=False)
        if do_var:
            preds[i:i + len(batch)], pred_vars[i:i + len(batch)] = sess.run(
                [net.output, net.output_var], feed_dict=d)
        else:
            preds[i:i + len(batch)] = net.output.eval(feed_dict=d)
        i += len(batch)
    return (preds, pred_vars) if do_var else preds


###############################

if __name__ == '__main__':
    data_path = '~/Documents/LibDems/data/projection_data'

    #### GET DATA

    ## covariate data
    X_latlong = pd.read_pickle(data_path + '/X_latlong.pkl')
    X_scores = pd.read_pickle(data_path + '/X_scores.pkl')
    X_demo = pd.read_pickle(data_path + '/X_demo.pkl')

    ## outcome data
    outcome_data = pd.read_pickle(data_path + '/outcome_data.pkl')
    outcome_data['constituency_lower'] = outcome_data['constituency'].str.lower()
    outcome_data.set_index(['code', 'constituency'], inplace=True)

    # add code num to outcome data
    temp = X_demo[['constituency', 'code_num']]
    temp.insert(2, "constituency_lower", temp['constituency'].str.lower())
    temp = temp.groupby(['code_num', 'constituency_lower']).size().reset_index()

    outcome_data = pd.merge(temp, outcome_data, left_on='constituency_lower', right_on='constituency_lower',
                            how='inner')

    ## landmark data
    r['load']('~/github/libdems/turnout/turnout_ldmks_300_data.RData')
    turnout_ldmks = r.turnout_ldmks
    landmark_ind = turnout_ldmks[0].astype(int)

    #### MAKE FEATURES
    # bags = X_latlong.to_numpy()[:, 2:]
    # n_pts = X_latlong.groupby('code_num')['code_num'].count().to_numpy()

    i = X_latlong['code_num'].values
    ukeys, slices = np.unique(i, True)
    X_latlong_array = np.split(X_latlong.iloc[:, 2:].values, slices[1:])

    feats = Features(bags=X_latlong_array, copy=False, bare=False, y=outcome_data['pct_turnout_ge2017'])
    feats.make_stacked()
    # feats.stacked

    #### SET ARGS
    args = {'reg_out': 0  # regularization param for regression coefs
        , 'reg_out_bias': 0  # regularisation param for regression intercept
        , 'scale_reg_by_n': False
        , 'dtype_double': False
        , 'type': 'radial'  # type pf network to use
        , 'init_from_ridge': False  # use ridge regression to initialize regression coefs
        , 'landmarks': feats.stacked_features[landmark_ind]
        , 'opt_landmarks': False  # whether or not to optimize landmarks too

        , 'optimizer': 'adam'
        , 'out_dir': '/users/valeriebradley/github/bdr/neuralnets/results/'
        , 'batch_pts': np.inf
        , 'batch_bags': 30
        , 'eval_batch_pts': np.inf
        , 'eval_batch_bags': 100
        , 'max_epochs': 10
        , 'first_early_stop_epoch': 10
        , 'learning_rate': 0.01

            # , 'n_estop':50
        , 'test_size': 0.2
        , 'trainval_size': None
        , 'val_size': 0.1875
        , 'train_estop_size': None
        , 'estop_size': 0.23
        , 'train_size': None
        , 'split_seed': np.random.randint(2 ** 32)
            }

    # split data
    train, estop, val, test = _split_feats(args, feats)

    # build network
    net = make_network(args, train)

    # initialize outcome vars
    d = {'args': args}

    # train and evaluate
    with tf_session(n_cpus=4) as sess:
        train_net(sess, args, net, train, estop)

        # save weights
        for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES):
            d[v.name] = v.eval()

        # whether we want to evaluate the variance as well (which we ignore)
        do_var = False

        for name, ds in [('val', val), ('test', test)]:
            print()
            preds = eval_network(sess, net, ds
                                 , batch_pts=args['eval_batch_pts']
                                 , batch_bags=args['eval_batch_bags']
                                 , do_var=do_var)
            if do_var:
                preds, preds_var = preds
                d[name + '_preds_var'] = preds_var

            d[name + '_y'] = y = ds.y
            d[name + '_preds'] = preds

            d[name + '_mse'] = mse = mean_squared_error(y, preds)
            print('{} MSE: {}'.format(name, mse))

            d[name + '_r2'] = r2 = r2_score(y, preds)
            print('{} R2: {}'.format(name, r2))

            if do_var:
                liks = stats.norm.pdf(y, preds, np.sqrt(preds_var))
                d[name + '_nll'] = nll = -np.log(liks).mean()
                print('{} NLL: {}'.format(name, nll))

                cdfs = stats.norm.cdf(y, preds, np.sqrt(preds_var))
                coverage = np.mean((cdfs > .025) & (cdfs < .975))
                d[name + '_coverage'] = coverage
                print('{} coverage at 95%: {:.1%}'.format(name, coverage))

    d['log_bw:0']
    d['out_bias:0']
    d['out:0']  # regression coefficients
    d['out:0'].shape
    # d['landmarks:0']

    