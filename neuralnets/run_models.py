
# usage example
# python run_models.py --type 'simple' --outcome-name 'pct_turnout_ge2017' --out-dir 'experiments/simple-rbf'
#  python run_models.py --type simple --outcome-type categorical --outcome-name mrpdec_supp_con mrpdec_supp_lab mrpdec_supp_ld --out-dir experiments/simple-rbf --max-epoch 10

from __future__ import division, print_function
import os
print(os.getcwd())
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf

from bdr.features import Features
from bdr.networks.radial import build_simple_rbf, build_spatsep_rbf
from bdr.train import eval_network, train_network
from bdr.utils import get_median_sqdist, tf_session

from rpy2.robjects import r, pandas2ri
pandas2ri.activate()


network_types = {
    'simple': build_simple_rbf,
    'spatial_sep': build_spatsep_rbf
}


def get_parsed():
    parser = argparse.ArgumentParser(description="Run network")

    # network parameters
    parser.add_argument('--type', required=True, choices=network_types)
    # g = parser.add_mutually_exclusive_group()
    # g.add_argument('--ldmks', default = None)
    # g.add_argument('--ldmk_path', default = '/Users/valeriebradley/github/libdems/turnout/turnout_ldmks_300_data.RData')

    parser.add_argument('--opt-landmarks', action='store_true', default=True,
                         help = 'whether to optimize the landmarks')
    parser.add_argument('--bw-scale', type=float, default=1)
    parser.add_argument('--reg-out', type=float, default=0,
                         help = 'regularization coefficient for regression weights')
    parser.add_argument('--reg-out-bias', type=float, default=0,
                         help = 'regularization coefficient for regression intercept weight')
    parser.add_argument('--scale-reg-by-n', action='store_true', default=False)
    parser.add_argument('--init-from-ridge', action='store_true', default=False,
                         help = 'initialize coefficients from ridge regression')
    parser.add_argument('--dtype-double', action='store_true', default=False)
    parser.add_argument('--dtype-single', action='store_false', dest='dtype_double')

    # data parameters
    parser.add_argument('--test-size', default=0.2)
    parser.add_argument('--val-size', default=None)
    parser.add_argument('--estop-size', default=0.23)
    parser.add_argument('--split-seed', default=np.random.randint(2 ** 32))
    parser.add_argument('--outcome-name', required=True, nargs = "*")
    parser.add_argument('--outcome-type', default = 'binary')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--feat-types', default=None)
    parser.add_argument('--pred-disagg', action='store_true', default=False)

    # training parameters
    int_inf = lambda x: np.inf if x.lower() in {'inf', 'none'} else int(x)
    parser.add_argument('--optimizer', default = 'adam')
    parser.add_argument('--batch-pts', type=int_inf, default='inf')
    parser.add_argument('--batch-bags', default = 30)
    parser.add_argument('--eval-batch-pts', type=int_inf, default='inf')
    parser.add_argument('--eval-batch-bags', default = 30)
    parser.add_argument('--max-epochs', default = 200, type = int)
    parser.add_argument('--first-early-stop-epoch', type=int, default = None, help="Default: MAX_EPOCHS / 3.")
    parser.add_argument('--learning-rate', default = 0.01)

    args = parser.parse_args()
    return args


def _split_feats(args, feats, labels=None, groups=None):
    if groups is None:
        ss = ShuffleSplit
    else:
        ss = GroupShuffleSplit

    rs = check_random_state(args['split_seed'])

    test_splitter = ss(1,
                       train_size=args['test_size'],
                       random_state=rs)
    (test_ind, train_estop_val_ind), = test_splitter.split(feats, None, None)

    X_v = feats[train_estop_val_ind]
    y_v = None if labels is None else labels[train_estop_val_ind]
    g_v = None if groups is None else groups[train_estop_val_ind]

    if args['val_size'] is not None:
        val_splitter = ss(1,
                          train_size=args['val_size'],
                          random_state=rs)
        (val_ind, train_estop_ind), = val_splitter.split(X_v, y_v, g_v)
        val = X_v[val_ind]
    else:
        val = None
        train_estop_ind = list(range(0,len(train_estop_val_ind)))


    estop_splitter = ss(1,
                        train_size=args['estop_size'],
                        random_state=rs)
    X = X_v[train_estop_ind]
    y = None if labels is None else y_v[train_estop_ind]
    g = None if groups is None else g_v[train_estop_ind]
    (estop_ind, train_ind), = estop_splitter.split(X, y, g)

    return X[train_ind], X[estop_ind], val, feats[test_ind]

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

    kw = {'x_dim':train.dim
          , 'y_dim':train.y.shape[1]
        , 'reg_out': args['reg_out']
        , 'reg_out_bias': args['reg_out_bias']
        , 'scale_reg_by_n': args['scale_reg_by_n']
        , 'dtype': tf.float64 if args['dtype_double'] else tf.float32
        , 'outcome_type':args['outcome_type']
          }

    kw['bw'] = np.sqrt(get_median_sqdist(train) / 2)

    # get landmarks
    #kw['landmarks'] = landmarks = pick_landmarks(args, train)
    kw['landmarks'] = args['landmarks']
    kw['opt_landmarks'] = args['opt_landmarks']

    # if args['type'] != 'simple':
    #     raise Exception(args['type'] + ' not recognized type of network')

    if args['type'] != 'simple':
        kw['feat_types'] = args['feat_types']

    return network_types[args['type']](**kw)



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



def plot_agg_results(d, save_file = None):
    y_dim = d['test_preds'].shape[1]

    if y_dim > 1:
        x = np.linspace(0., 1., 100)
    else:
        x = np.linspace(min(d['test_preds']), max(d['test_preds']), 100)

    c = np.ceil(np.sqrt(y_dim))
    r = np.ceil(y_dim / c)

    # plt.plot(d['train_y'], d['train_preds'], 'o', label='training set')
    # plt.plot(d['test_y'], d['test_preds'], 'o', label = 'test set')
    # plt.plot(d['estop_y'], d['estop_preds'], 'o', label = 'estop set')

    fig = plt.figure()
    #plt.title("Test set performance")
    #plt.xlabel("Actual")
    #plt.ylabel("Predicted")

    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # add points
    for i in range(1, y_dim + 1):
        ind = i - 1
        ax = fig.add_subplot(r, c, i)

        # add x = y
        ax.plot(x, x, linestyle='--', color='black')

        # add points
        ax.plot(d['train_y'][:,ind], d['train_preds'][:,ind], 'o', label='training set', alpha = 0.5)
        ax.plot(d['test_y'][:,ind], d['test_preds'][:,ind], 'o', label='test set', alpha = 0.5)

        title = d['args']['outcome_name'][i-1]
        ax.set_title(title)

    #fig.legend()
    #plt.show()

    if save_file is not None:
        fig.savefig(save_file)

    return fig


###############################

if __name__ == '__main__':

    args = vars(get_parsed())
    print(args)

    data_path = '/Users/valeriebradley/Documents/LibDems/data/projection_data'

    ## covariate data
    X_latlong = pd.read_pickle(data_path + '/X_latlong.pkl')
    X_scores = pd.read_pickle(data_path + '/X_scores.pkl')
    X_demo = pd.read_pickle(data_path + '/X_demo.pkl')

    X_all = np.concatenate((X_latlong.iloc[:, 2:],
                            X_scores.iloc[:, 2:],
                            X_demo.iloc[:, 2:]), axis=1)

    var_cat = np.concatenate((np.repeat('latlong', X_latlong.shape[1] - 2),
                              np.repeat('scores', X_scores.shape[1] - 2),
                              np.repeat('demo', X_demo.shape[1] - 2)))
    args['feat_types'] = var_cat

    ## outcome data
    outcome_data = pd.read_pickle(data_path + '/outcome_data.pkl')

    #### MAKE FEATURES
    i = X_demo['code_num'].values
    ukeys, slices = np.unique(i, True)
    X_all_array = np.split(X_all, slices[1:])

    feats = Features(bags=X_all_array, copy=False, bare=False, y=outcome_data[args['outcome_name']])
    feats.make_stacked()

    #individ feats
    if args['pred_disagg']:
        feats_individ = Features(bags=feats.stacked_features, n_pts = np.repeat(1,feats.total_points), copy=False, bare=False)
        feats_individ.make_stacked()
        (train_i_ind, test_i_ind), = ShuffleSplit(1,
                           test_size=0.025,
                           random_state=check_random_state(args['split_seed'])).split(feats_individ, None, None)
        test_individ = feats_individ[test_i_ind]
    else:
        test_individ = None

    ## landmark data
    r['load']('~/github/libdems/turnout/turnout_ldmks_300_data.RData')
    turnout_ldmks = r.turnout_ldmks
    landmark_ind = turnout_ldmks[0].astype(int)

    # add landmarks to args
    args['landmarks'] = feats.stacked_features[landmark_ind]

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

        pred_list = [('train', train), ('estop', estop), ('test', test)]

        if args['pred_disagg']:
            pred_list.append(('test_individ', test_individ))

        print(pred_list)

        for name, ds in pred_list:
            print()
            preds = eval_network(sess, net, ds
                                 , batch_pts=args['eval_batch_pts']
                                 , batch_bags=args['eval_batch_bags']
                                 , do_var=do_var)
            if do_var:
                preds, preds_var = preds
                d[name + '_preds_var'] = preds_var

            d[name + '_preds'] = preds

            if hasattr(ds, 'y'):
                d[name + '_y'] = y = ds.y
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

    plot_agg_results(d, save_file=args['out_dir'] + '/ploterror.png')

    pd.to_pickle(d, args['out_dir'] + '/net_params.pkl')
