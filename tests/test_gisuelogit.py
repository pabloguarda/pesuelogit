from gisuelogit import __version__

import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import tensorflow as tf
import isuelogit as isl

# Internal modules
from src.gisuelogit.visualizations import plot_predictive_performance, plot_convergence_estimates
from src.gisuelogit.models import UtilityParameters, GISUELOGIT, BPRParameters, ODParameters, compute_rr
from src.gisuelogit.networks import load_k_shortest_paths, build_tntp_network, Equilibrator
from src.gisuelogit.etl import get_design_tensor, get_y_tensor, simulate_suelogit_data, add_period_id, simulate_features
from src.gisuelogit.descriptive_statistics import mse

# Seed for reproducibility
_SEED = 2022


def test_version():
    assert __version__ == '0.1.0'

def simulate_data(n_days = 10):

    np.random.seed(_SEED)
    random.seed(_SEED)
    tf.random.set_seed(_SEED)

    # Path management
    main_dir = str(Path(os.path.abspath('')).parents[1])
    os.chdir(main_dir)
    print('main dir:', main_dir)

    ## Build network
    network_name = 'SiouxFalls'
    tntp_network = build_tntp_network(network_name=network_name)

    ## Read OD matrix
    Q = isl.reader.read_tntp_od(network_name=network_name)
    tntp_network.load_OD(Q=Q)

    # Paths
    load_k_shortest_paths(network=tntp_network, k=2, update_incidence_matrices=True)
    # features_Z = []

    # Simulate counts
    n_links = len(tntp_network.links)
    features_Z = ['tt_sd', 's']

    n_sparse_features = 0
    features_sparse = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

    utility_function = UtilityParameters(features_Y=['tt'],
                                         features_Z=features_Z,
                                         true_values={'tt': -1, 'tt_sd': -1.3, 's': -3}
                                         #  true_values={'tt': -1, 'c': -6, 's': -3}
                                         )

    utility_function.add_sparse_features(Z=features_sparse)

    equilibrator = Equilibrator(network=tntp_network,
                                utility_function=utility_function,
                                uncongested_mode=False,
                                max_iters=100,
                                method='fw',
                                accuracy=1e-20,
                                iters_fw=100,
                                search_fw='grid')

    exogenous_features = simulate_features(links=tntp_network.links,
                                           features_Z=features_Z + features_sparse,
                                           option='continuous',
                                           daytoday_variation=False,
                                           range=(0, 1),
                                           n_days=n_days)

    # Generate data from multiple days. The value of the exogenous attributes varies between links but not between days (note: sd_x is the standard deviation relative to the true mean of traffic counts)

    df = simulate_suelogit_data(
        days=list(exogenous_features.period.unique()),
        features_data=exogenous_features,
        equilibrator=equilibrator,
        sd_x=0,
        sd_t=0,
        network=tntp_network)

    output_file = tntp_network.key + '-link-data.csv'
    output_dir = Path('tests/output/network-data/' + tntp_network.key + '/links')

    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.reset_index()

    # Write data to csv
    df.to_csv(output_dir / output_file, index=False)

    return df, tntp_network, features_Z

def test_simulate_data():

    df, tntp_network, features_Z = simulate_data()

    # REad synthethic data which was generated under the assumption of path sets of size 2.
    df_read = pd.read_csv(
        'tests/output/network-data/' + tntp_network.key + '/links/' + tntp_network.key + '-link-data.csv')

    selected_columns = [i for i in df.columns if i != 'link_key']

    assert_frame_equal(df[selected_columns],df_read[selected_columns])

def tensorize_data():

    # Read synthethic data which was generated under the assumption of path sets of size 2.
    df, tntp_network, features_Z = simulate_data()

    # Prepare the training and validation dataset.

    n_timepoints = len(df.period.unique())
    n_links = len(tntp_network.links)

    # Add free flow travel times
    df['tt_ff'] = np.tile([link.bpr.tf for link in tntp_network.links], n_timepoints)

    traveltime_data = get_design_tensor(y=df['traveltime'], n_links=n_links, n_timepoints=n_timepoints)
    flow_data = get_y_tensor(y=df[['counts']], n_links=n_links, n_timepoints=n_timepoints)

    df['hour'] = 0
    df.loc[df['period'] <= 50, 'hour'] = 1
    df = add_period_id(df, period_feature='hour')
    # df = add_period_id(df, period_feature='period')

    Y = tf.concat([traveltime_data, flow_data], axis=2)
    X = get_design_tensor(Z=df[features_Z + ['period_id']], n_links=n_links, n_timepoints=n_timepoints)

    X_train, X_test, Y_train, Y_test = train_test_split(X.numpy(), Y.numpy(), test_size=0.2, random_state=_SEED)

    X_train, X_test, Y_train, Y_test = [tf.constant(i) for i in [X_train, X_test, Y_train, Y_test]]

    return X_train, X_test, Y_train, Y_test, tntp_network, features_Z


def test_zero_error():

    X_train, X_test, Y_train, Y_test, tntp_network, features_Z = tensorize_data()

    # _EPOCHS = {'learning': 1000, 'equilibrium': 0}
    _EPOCHS = {'learning': 500, 'equilibrium': 0}
    _LR = 1#e-1
    _RELATIVE_GAP = 1e-15
    _XTICKS_SPACING = 50
    # To reduce variability in estimates of experiments, it is better to not use batches
    # _BATCH_SIZE = None
    _BATCH_SIZE = 16
    _MOMENTUM_EQUILIBRIUM = 1

    # loss_metric = mnrmse
    _LOSS_METRIC = mse

    _EPOCHS_PRINT_INTERVAL = 5

    _LOSS_WEIGHTS = {'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 0}

    # Models
    list_models = ['equilibrium', 'lue', 'ode', 'lpe', 'odlue', 'odlulpe', 'tvodlulpe']

    # run_model = dict.fromkeys(list_models,True)
    run_model = dict.fromkeys(list_models, False)

    run_model['ode'] = True

    train_results_dfs = {}
    val_results_dfs = {}

    if run_model['ode']:
        print('\n ODE: OD estimation with historic OD')

        # _RELATIVE_GAP = 1e-6

        optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)
        # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)

        utility_parameters = UtilityParameters(features_Y=['tt'],
                                               features_Z=features_Z,
                                               initial_values={'tt': -1, 'tt_sd': -1.3, 's': -3, 'psc_factor': 0,
                                                               'fixed_effect': np.zeros_like(tntp_network.links)},
                                               trainables={'psc_factor': False, 'fixed_effect': False
                                                   , 'tt': False, 'tt_sd': False, 's': False},
                                               )

        bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                       initial_values={'alpha': 0.15, 'beta': 4},
                                       trainables=dict.fromkeys(['alpha', 'beta'], False),
                                       )

        # Q_historic = isl.factory.random_disturbance_Q(tntp_network.Q, sd=np.mean(tntp_network.Q) * 0).copy()
        Q_historic = tntp_network.Q.copy()

        od_parameters = ODParameters(key='od',
                                     # initial_values=tntp_network.q.flatten(),
                                     initial_values=isl.networks.denseQ(Q_historic).flatten(),
                                     # initial_values=np.ones_like(tntp_network.q.flatten()),
                                     historic_values={1: isl.networks.denseQ(Q_historic).flatten()},
                                     trainable=True)

        equilibrator = Equilibrator(
            network=tntp_network,
            # paths_generator=paths_generator,
            utility=utility_parameters,
            max_iters=100,
            method='fw',
            iters_fw=50,
            accuracy=1e-4,
        )

        ode = GISUELOGIT(
            key='ode',
            network=tntp_network,
            dtype=tf.float64,
            equilibrator=equilibrator,
            # column_generator=column_generator,
            utility=utility_parameters,
            bpr=bpr_parameters,
            od=od_parameters,
            # n_periods=len(np.unique(X_train[:, :, -1].numpy().flatten()))
        )

        train_results_dfs['ode'], val_results_dfs['ode'] = ode.train(
            X_train, Y_train, X_test, Y_test,
            optimizer=optimizer,
            batch_size=_BATCH_SIZE,
            loss_metric=_LOSS_METRIC,
            # generalization_error={'train': False, 'validation': True},
            loss_weights=dict(_LOSS_WEIGHTS, od=1),
            momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
            threshold_relative_gap=_RELATIVE_GAP,
            epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
            epochs=_EPOCHS)

        plot_predictive_performance(train_losses=train_results_dfs['ode'], val_losses=val_results_dfs['ode'],
                                    xticks_spacing=250)

        plt.show()

        print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(ode.theta.numpy())))}")
        print(f"alpha = {ode.alpha: 0.2f}, beta  = {ode.beta: 0.2f}")
        print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(ode.q - tntp_network.q.flatten())): 0.2f}")

        assert np.allclose(pd.to_numeric(train_results_dfs['ode'].iloc[-1][['loss_flow', 'loss_tt']].values),0.0,atol = 1e-10)

def test_equilibrium(_RELATIVE_GAP = 1e-8):
    '''
    Check if equilibrium achieves small relative gap. A larger number of epochs may be necessary to achieve the target gap
    '''

    X_train, X_test, Y_train, Y_test, tntp_network, features_Z = tensorize_data()

    _EPOCHS = {'learning': 2000, 'equilibrium': 0}
    # _EPOCHS = {'learning': 10, 'equilibrium': 2}
    _LR = 5e-1#e-1

    _XTICKS_SPACING = 50
    # To reduce variability in estimates of experiments, it is better to not use batches
    # _BATCH_SIZE = None
    _BATCH_SIZE = 16
    _MOMENTUM_EQUILIBRIUM = 1

    # loss_metric = mnrmse
    _LOSS_METRIC = mse

    _EPOCHS_PRINT_INTERVAL = 5

    # Models
    list_models = ['equilibrium', 'lue', 'ode', 'lpe', 'odlue', 'odlulpe', 'tvodlulpe']

    # run_model = dict.fromkeys(list_models,True)
    run_model = dict.fromkeys(list_models, False)

    run_model['equilibrium'] = True

    train_results_dfs = {}
    val_results_dfs = {}

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    print('ISUELOGIT: Equilibrium')

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': -1, 'tt_sd': -1.3, 's': -3, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': False, 'tt_sd': False, 's': False},
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 1, 'beta': 1},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 # initial_values=0.6 * tntp_network.q.flatten(),
                                 initial_values=tntp_network.q.flatten(),
                                 true_values=tntp_network.q.flatten(),
                                 historic_values={1: tntp_network.q.flatten()},
                                 trainable=False)

    len(tntp_network.q.flatten())

    # q_historic = isl.networks.denseQ(isl.factory.random_disturbance_Q(tntp_network.Q, sd=np.mean(tntp_network.Q) * 0.05).copy())

    q_historic = tntp_network.q

    equilibrator = Equilibrator(
        network=tntp_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    suelogit = GISUELOGIT(
        key='suelogit',
        # endogenous_flows=True,
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        # column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters
    )

    train_results_dfs['suelogit'], val_results_dfs['suelogit'] = suelogit.train(
        X_train, Y_train, X_test, Y_test,
        # generalization_error={'train': False, 'validation': True},
        # loss_metric = mse,
        loss_metric=_LOSS_METRIC,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 0, 'theta': 0, 'tt': 0, 'flow': 0, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    train_results_estimates, train_results_losses = suelogit.split_results(results=train_results_dfs['suelogit'])
    val_results_estimates, val_results_losses = suelogit.split_results(results=val_results_dfs['suelogit'])

    plot_predictive_performance(train_losses=train_results_dfs['suelogit'], val_losses=val_results_dfs['suelogit'],
                                xticks_spacing=250)

    fig, ax = plot_convergence_estimates(
        estimates=train_results_losses.drop([0]).assign(
            relative_gap=np.abs(train_results_losses['relative_gap']))[['epoch', 'relative_gap']])

    ax.set_yscale('log')
    ax.set_ylabel('relative gap (log scale)')

    plt.show()

    relative_gaps = np.abs(train_results_losses['relative_gap']).values

    assert relative_gaps[-1]<_RELATIVE_GAP

