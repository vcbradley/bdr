
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
from bdr.networks.radial import build_simple_rbf, build_spatsep_rbf, build_nonstat_rbf
from bdr.train import eval_network, train_network
from bdr.utils import get_median_sqdist, tf_session

from rpy2.robjects import r, pandas2ri
pandas2ri.activate()


network_types = {
    'simple': build_simple_rbf,
    'spatial_sep': build_spatsep_rbf,
    'nonstationary': build_nonstat_rbf
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
    parser.add_argument('--reg-ell-weights', type=float, default=0,
                        help='regularization coefficient for nonstat spatial kernel reg')
    parser.add_argument('--scale-reg-by-n', action='store_true', default=False)
    parser.add_argument('--init-from-ridge', action='store_true', default=False,
                         help = 'initialize coefficients from ridge regression')
    parser.add_argument('--dtype-double', action='store_true', default=False)
    parser.add_argument('--dtype-single', action='store_false', dest='dtype_double')
    parser.add_argument('--feat-agg-type', default='concat')

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

    remaining_ind=np.arange(len(feats))

    test_splitter = ss(1,
                       train_size=args['test_size'],
                       random_state=rs)
    (i, j), = test_splitter.split(remaining_ind, None, None)
    test_ind = remaining_ind[i]
    remaining_ind = remaining_ind[j]

    y_v = None if labels is None else labels[remaining_ind]
    g_v = None if groups is None else groups[remaining_ind]

    if args['val_size'] is not None:
        val_splitter = ss(1,
                          train_size=args['val_size'],
                          random_state=rs)
        (i, j), = val_splitter.split(remaining_ind, y_v, g_v)
        val_ind = remaining_ind[i]
        remaining_ind = remaining_ind[j]
    else:
        val_ind = None

    estop_splitter = ss(1,
                        train_size=args['estop_size'],
                        random_state=rs)

    y = None if labels is None else y_v[remaining_ind]
    g = None if groups is None else g_v[remaining_ind]
    (i, j), = estop_splitter.split(remaining_ind, y, g)

    estop_ind = remaining_ind[i]
    train_ind = remaining_ind[j]

    return train_ind, estop_ind, val_ind, test_ind

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
          , 'feat_agg_type':args['feat_agg_type']
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
    y_dim = d['preds'].shape[1]

    if y_dim > 1:
        x = np.linspace(0., 1., 100)
    else:
        x = np.linspace(min(d['preds']), max(d['preds']), 100)

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
        train_ind = d['args']['split_ind']['train']
        other_ind = [i for i in np.arange(d['preds'].shape[0]) if i not in train_ind]
        ax.plot(d['y'][train_ind, ind], d['preds'][train_ind, ind], 'o', label='training set', alpha = 0.5)
        ax.plot(d['y'][other_ind, ind], d['preds'][other_ind, ind], 'o', label='non-training set', alpha=0.5)

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

    ## landmark data
    r['load']('~/github/libdems/turnout/turnout_ldmks_300_data.RData')
    turnout_ldmks = r.turnout_ldmks
    landmark_ind = turnout_ldmks[0].astype(int)

    # add landmarks to args
    args['landmarks'] = feats.stacked_features[landmark_ind]

    # split data
    train_ind, estop_ind, val_ind, test_ind = _split_feats(args, feats)
    train = None if train_ind is None else feats[train_ind]
    estop = None if estop_ind is None else feats[estop_ind]
    val = None if val_ind is None else feats[val_ind]
    test = None if test_ind is None else feats[test_ind]
    args['split_ind'] = {
        'train' : train_ind,
        'estop' : estop_ind,
        'val':val_ind,
        'test':test_ind
    }

    #individ feats
    if args['pred_disagg']:
        feats_individ = Features(bags=feats.stacked_features, n_pts = np.repeat(1,feats.total_points), copy=False, bare=False)
        feats_individ.make_stacked()
    else:
        feats_individ = None

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
        # do all predictions
        preds = eval_network(sess, net, feats
                             , batch_pts=args['eval_batch_pts']
                             , batch_bags=args['eval_batch_bags']
                             , do_var=do_var)
        d['y'] = feats.y

        if do_var:
            preds, preds_var = preds
            d['preds_var'] = preds_var

        d['preds'] = preds

        # individ predictions
        if args['pred_disagg']:
            print("Predicting disaggregated observations")
            preds_individ = eval_network(sess, net, feats_individ
                                 , batch_pts=args['eval_batch_pts']
                                 , batch_bags=args['eval_batch_bags']
                                 , do_var=do_var)
            print('Done')


        pred_list = ['train', 'estop', 'test']
        for name in pred_list:

            y = feats.y[args['split_ind'][name]]
            y_hat = preds[args['split_ind'][name]]

            d[name + '_mse'] = mse = mean_squared_error(y, y_hat)
            print('{} MSE: {}'.format(name, mse))

            d[name + '_r2'] = r2 = r2_score(y, y_hat)
            print('{} R2: {}'.format(name, r2))

            if do_var:
                var_hat = preds_var[args['split_ind'][name]]

                liks = stats.norm.pdf(y, y_hat, np.sqrt(var_hat))
                d[name + '_nll'] = nll = -np.log(liks).mean()
                print('{} NLL: {}'.format(name, nll))

                cdfs = stats.norm.cdf(y, preds, np.sqrt(var_hat))
                coverage = np.mean((cdfs > .025) & (cdfs < .975))
                d[name + '_coverage'] = coverage
                print('{} coverage at 95%: {:.1%}'.format(name, coverage))

    plot_agg_results(d, save_file=args['out_dir'] + '/ploterror.png')

    pd.to_pickle(d, args['out_dir'] + '/net_params.pkl')
