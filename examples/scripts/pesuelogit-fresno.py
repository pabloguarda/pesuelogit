'''
'''

import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
from datetime import datetime

import numpy as np
import pandas as pd
import random
# import cudf as pd
import tensorflow as tf
import isuelogit as isl
import glob

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:', main_dir)
isl.config.dirs['read_network_data'] = "input/network-data/fresno/"

# Internal modules
from src.pesuelogit.models import UtilityParameters, BPRParameters, ODParameters, PESUELOGIT, NGD, compute_rr
from src.pesuelogit.visualizations import plot_predictive_performance, plot_convergence_estimates, plot_utility_parameters_periods, plot_top_od_flows_periods, plot_rr_by_period, plot_rr_by_period_models, plot_total_trips_models
from src.pesuelogit.networks import load_k_shortest_paths, read_paths, build_fresno_network, \
    Equilibrator, sparsify_OD, ColumnGenerator, read_OD
from src.pesuelogit.etl import get_design_tensor, get_y_tensor, data_curation, temporal_split, add_period_id, get_tensors_by_year
from src.pesuelogit.descriptive_statistics import mse, btcg_mse, nrmse, mnrmse

# Seed for reproducibility
_SEED = 2023
np.random.seed(_SEED)
random.seed(_SEED)
tf.random.set_seed(_SEED)

# To write estimation report and set seed for all algorithms involving some randomness
# estimation_reporter = isl.writer.Reporter(
#     folderpath=isl.config.dirs['output_folder'] + 'estimations/' + 'Fresno', seed=_SEED)

## Build Fresno network
fresno_network = build_fresno_network()

# [(link.bpr.tf,link.link_type) for link in fresno_network.links if link.link_type != 'LWRLK']

## Read OD matrix
# TODO: option to specify path to read OD matrix
read_OD(network=fresno_network, sparse=True)

# np.sum(fresno_network.Q)

# Read paths
# read_paths(network=fresno_network, update_incidence_matrices=True, filename='paths-fresno.csv')
# read_paths(network=fresno_network, update_incidence_matrices=True, filename = 'paths-full-model-fresno.csv')

# For quick testing (do not need to read_paths before)
Q = fresno_network.load_OD(sparsify_OD(fresno_network.Q, prop_od_pairs=0.99))
load_k_shortest_paths(network=fresno_network, k=2, update_incidence_matrices=True)

## Read spatiotemporal data
folderpath = isl.config.dirs['read_network_data'] + 'links/spatiotemporal-data/'
df = pd.concat([pd.read_csv(file) for file in glob.glob(folderpath + "*link-data*")], axis=0)
df.hour.unique()

# TODO: Check why there are missing dates, e.g. October 1, 2019
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Select data from Tuesday to Thursday
df = df[df['date'].dt.dayofweek.between(1, 3)]
# df = df[df['date'].dt.year == 2019]
# df['date'].dt.dayofweek.unique()
# len(sorted(df['date']).unique())
df['period'] = df['date'].astype(str) + '-' + df['hour'].astype(str)
# df['period'] = df.period.map(hash)

# # Consolidate all csv files into two single files per year. Read from output folder
# read_folderpath = 'output/network-data/fresno/links/'
# df = pd.concat([pd.read_csv(file) for file in glob.glob(read_folderpath + "*fresno-link-data*")], axis=0)
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# df['year'] = df.date.dt.year
# df['period'] = df['date'].astype(str) + '-' + df['hour'].astype(str)
#
# # Only select data from Tuesday to Thursday
# df = df[df['date'].dt.dayofweek.between(1, 3)]
#
# # Write data in input folder
# write_folderpath = isl.config.dirs['read_network_data'] + 'links/spatiotemporal-data/'
#
# for year in sorted(df['year'].unique()):
#     df_year = df[df['year'] == year].sort_values('period')
#
#     filename = f'fresno-spatiotemporal-link-data-{year}.csv.gz'
#     filepath = f"{os.getcwd()}/{write_folderpath}{filename}"
#
#     df_year.to_csv(filepath, index=False, compression="gzip")
#
#     print(f'consolidated file {filename} has been written')


# Add id for period and respecting the temporal order
# periods_keys = dict(zip(sorted(df['period'].unique()), range(len(sorted(df['period'].unique())))))

# - By hour
period_feature = 'hour'
df = add_period_id(df, period_feature=period_feature)
period_keys = df[[period_feature,'period_id']].drop_duplicates().reset_index().drop('index',axis =1).sort_values('hour')
print(period_keys)

# - By hour

# df1 = pd.read_csv(main_dir + '/input/network-data/' + fresno_network.key + '/links/2019-10-01-fresno-link-data.csv')
# df1['date'] = "2019-10-01"
# df1['period'] = 0
#
# df2 = pd.read_csv(main_dir + '/input/network-data/' + fresno_network.key + '/links/2020-10-06-fresno-link-data.csv')
# df2['date'] = "2020-10-06"
# df2['period'] = 1
#
# df = pd.concat([df1, df2], axis=0)

## Data curation

# df['tt_ff'] = np.tile([link.bpr.tf for link in fresno_network.links],len(df.date.unique())*len(df.hour.unique()))
# df['tt_ff'] = df['tf_inrix']

df['tt_ff'] = np.where(df['link_type'] != 'LWRLK', 0,df['length']/df['speed_ref_avg'])
df.loc[(df.link_type == "LWRLK") & (df.speed_ref_avg == 0),'tt_ff'] = float('nan')

df['tt_avg'] = np.where(df['link_type'] != 'LWRLK', 0,df['length']/df['speed_hist_avg'])
df.loc[(df.link_type == "LWRLK") & (df.speed_hist_avg == 0),'tt_avg'] = float('nan')

tt_sd_adj = df.groupby(['period_id','link_key'])[['tt_avg']].std().reset_index().rename(columns = {'tt_avg': 'tt_sd_adj'})

df = df.merge(tt_sd_adj, on = ['period_id','link_key'])

df = data_curation(df)

df['tt_sd'] = df['tt_sd_adj']

## Utility function

features_Z = ['tt_sd', 'median_inc', 'incidents', 'bus_stops', 'intersections']
# features_Z = ['tt_sd']
# features_Z = []

# utility_parameters.constant_initializer(0)

## Data processing

n_links = len(fresno_network.links)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['year'] = df.date.dt.year
df.hour.unique()

# Select only dates used for previous paper
# df = df.query('date == "2019-10-01"  | date == "2020-10-06"')
# df = df.query('date == "2019-10-01"')
# df = df.query('hour == 16')
# df = df.query('hour == 17')
# df = df.query('hour == 16 | hour == 17')
# df = df.query('hour == 17')

print(df.query('year == 2019')[['counts', 'tt_ff', 'tt_avg', 'tf_inrix']].describe())

print(df.query('year == 2020')[['counts', 'tt_ff', 'tt_avg', 'tf_inrix']].describe())

# Normalization of features to range [0,1]

# (TODO: may enable normalization in get_design_tensor method. See if tensorflow have it)
# TODO: exclude test data from transform to avoid data leakage
# df[features_Z + ['tt_avg'] + ['tt_ff']] \
#     = preprocessing.MaxAbsScaler().fit_transform(df[features_Z + ['tt_avg'] + ['tt_ff']])

# Set free flow travel times
# tt_ff_links = df.query('link_type == "LWRLK"').groupby('link_key')['tt_ff'].min()
tt_ff_links = df.groupby('link_key')['tt_ff'].min()
# [(link.bpr.tf,link.link_type) for link in fresno_network.links if link.link_type == "LWRLK"]
for link in fresno_network.links:
    fresno_network.links_dict[link.key].performance_function.tf = float(tt_ff_links[tt_ff_links.index==str(link.key)])

tt_ff_links.mean()
df[['tt_avg','tt_ff','tf_inrix']].mean()

df['tt_ff'] = df.groupby('link_key')['tt_ff'].transform(lambda x: x.min())


# EDA

obs_date = df.groupby('date')['hour'].count()

df.groupby('date')[['speed_sd','speed_avg', 'counts']].mean().assign(total_obs = obs_date)
#
# eda_df = df.copy()
# eda_df['date'] = eda_df['date'].astype(str)
#
# # Transform to monthly income
# eda_df['median_inc'] = eda_df['median_inc']/12
#
# sns.lineplot(x= 'date', y = 'counts', data =eda_df.groupby('date')[['counts']].mean().reset_index())
# plt.tight_layout()
# plt.xticks(rotation=90)
# plt.show()
#
#
# sns.lineplot(x= 'date', y = 'value', hue = 'variable', data =pd.melt(eda_df.groupby('date')[features_Z].mean().reset_index(),id_vars= ['date']))
# plt.tight_layout()
# plt.xticks(rotation=90)
# # plt.show()
#
# sns.lineplot(x= 'date', y = 'speed_avg', data =eda_df.groupby('date')[['speed_avg']].mean().reset_index())
# plt.tight_layout()
# plt.xticks(rotation=90)
# # plt.show()
# plt.show()
#
# print(eda_df.groupby('date')[features_Z].mean())
#
# print(df.groupby('date')[['tt_avg', 'tt_sd', 'tt_ff']].mean())

# Link flows by hour
# eda_df = df.copy()
# eda_df['date'] = eda_df['date'].astype(str)
#
# link_keys = eda_df[(eda_df.counts>0) & (eda_df.speed_avg>0)].link_key.unique()
# link_keys = link_keys[0:20]
#
# sns.lineplot(x= 'hour', y = 'counts', hue = 'link_key',
#              data =eda_df[eda_df.link_key.isin(link_keys)].groupby(['hour','link_key'])[['counts']].mean().reset_index())
# plt.show()
#
# sns.lineplot(x= 'hour', y = 'counts', data =eda_df.groupby(['hour','link_key'])[['counts']].mean().reset_index())
# plt.show()
#
#
# sns.lineplot(x= 'hour', y = 'speed_avg', hue = 'link_key',
#              data =eda_df[eda_df.link_key.isin(link_keys)].groupby(['hour','link_key'])[['speed_avg']].mean().reset_index())
# plt.show()
#
# sns.lineplot(x= 'hour', y = 'speed_avg',
#              data =eda_df.groupby(['hour','link_key'])[['speed_avg']].mean().reset_index())
# plt.show()
#
# sns.lineplot(x= 'hour', y = 'speed_sd',
#              data =eda_df.groupby(['hour','link_key'])[['speed_sd']].mean().reset_index())
# plt.show()

 # plt.hist(df_year['tt_avg'])
#
# bins = np.linspace(-10, 10, 100)

# plt.hist(df_year['tt_ff'], alpha=0.5, label='free flow travel times')
# plt.hist(df_year['tt_avg'], alpha=0.5, label='average travel times')
# plt.legend(loc='upper right')
# plt.show()
#
# plt.scatter(df_year['tt_ff'],df_year['tt_avg'])
# plt.legend(loc='upper right')
# plt.show()

## Training and validation sets

# Include only data between 4pm and 5pm
X, Y = get_tensors_by_year(df[df.hour == 16], features_Z = features_Z, network = fresno_network)
# Include hourly data between 6AM and 8PM (15 hour intervals)
# XT, YT = get_tensors_by_year(df, features_Z = features_Z)
# XT, YT = get_tensors_by_year(df[df.hour.isin(range(14,18))], features_Z = features_Z, network = fresno_network)
XT, YT = get_tensors_by_year(df[df.hour.isin([6,7,8, 15,16,17])], features_Z = features_Z, network = fresno_network)
# Split to comply with temporal ordering
# X_train, X_test, Y_train, Y_test = temporal_split(X[2019].numpy(), Y[2019].numpy(), n_days = X[2019].shape[0])

# X_train, X_test, Y_train, Y_test = X[2020], X[2019], Y[2020], Y[2019]
X_train, X_test, Y_train, Y_test = X[2019], X[2020], Y[2019], Y[2020]
XT_train, XT_test, YT_train, YT_test = XT[2019], XT[2020], YT[2019], YT[2020]

# Remove validation set to reduce computation costs
X_test, Y_test = None, None
XT_test, YT_test = None, None

#Models
# run_model = dict.fromkeys(['equilibrium', 'lue', 'ode', 'odlue', 'odlulpe-1','odlulpe', 'tvodlulpe'], True)
run_model = dict.fromkeys(['equilibrium', 'lue', 'ode', 'odlue', 'odlulpe-1','odlulpe', 'tvodlulpe', 'test_tvodlulpe'], False)

# run_model.update(dict.fromkeys(['lue', 'odlue', 'odlulpe'], True))
# run_model = dict.fromkeys( for i in ['lue', 'odlue', 'odlulpe'], True)
# run_model['equilibrium'] = True
# run_model['lue'] = True
# run_model['ode'] = True
# run_model['odlue'] = True
# run_model['odlulpe-1'] = True
# run_model['odlulpe'] = True
run_model['tvodlulpe'] = True
# run_model['test_tvodlulpe'] = True

train_results_dfs = {}
test_results_dfs = {}

# For testing
# _EPOCHS = {'learning': 10, 'equilibrium': 1}
_EPOCHS = {'learning': 10, 'equilibrium': 3}
_BATCH_SIZE = 16
_LR = 1e-1
_RELATIVE_GAP = 1e-5
_XTICKS_SPACING = 50
_EPOCHS_PRINT_INTERVAL = 1

# _LOSS_METRIC  = mnrmse
_LOSS_METRIC  = nrmse

# Excluding historic OD gives more freedom for the model to find an equilibria and minimize reconstruction error
_LOSS_WEIGHTS ={'od': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1, 'prop_od': 1, 'ntrips': 1}
_MOMENTUM_EQUILIBRIUM = 0.99
#_MOMENTUM_EQUILIBRIUM = 1

# Including historic OD matrix
# _LOSS_WEIGHTS ={'od': 1, 'tt': 1, 'flow': 1, 'eq_flow': 1}
# _MOMENTUM_EQUILIBRIUM = 0.99

# _LOSS_METRIC = mse
# _LOSS_WEIGHTS ={'od': 1, 'theta': 0, 'tt': 1e10, 'flow': 1, 'eq_flow': 1}

#_LOSS_METRIC  = btcg_mse
#_LOSS_METRIC  = mnrmse


print(f"Relative gap threshold: {_RELATIVE_GAP}, "
      f"Learning rate: {_LR}, "
      f"Batch size: {_BATCH_SIZE}")

optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

if run_model['equilibrium']:

    print('Equilibrium computation')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': -1, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': False,
                                                       'tt': False, 'tt_sd': False, 'median_inc': False,
                                                       'incidents': False,
                                                       'bus_stops': False, 'intersections': False
                                                       },
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 # initial_values=0.6 * tntp_network.q.flatten(),
                                 initial_values=fresno_network.q.flatten(),
                                 true_values=fresno_network.q.flatten(),
                                 historic_values={10: fresno_network.q.flatten()},
                                 trainable=False)

    equilibrator = Equilibrator(
        network=fresno_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    column_generator = ColumnGenerator(equilibrator=equilibrator,
                                       utility=utility_parameters,
                                       n_paths=0,
                                       ods_coverage=0.1,
                                       ods_sampling='sequential',
                                       # ods_sampling='demand',
                                       )

    print("\nSUELOGIT equilibrium")

    suelogit = PESUELOGIT(
        key='suelogit',
        # endogenous_flows=True,
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters
    )
    # X_train.shape
    # Y_train.shape
    train_results_dfs['suelogit'], test_results_dfs['suelogit'] = suelogit.train(
        X_train, Y_train, X_test, Y_test,
        # generalization_error={'train': False, 'validation': True},
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 0, 'theta': 0, 'tt': 0, 'flow': 0, 'eq_flow': 1},
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    # print("\nTravel time based autoencoder")
    # model_0b = AETSUELOGIT(
    #     key='model_0b',
    #     # endogenous_traveltimes=True,
    #     network=fresno_network,
    #     dtype=tf.float64,
    #     equilibrator=equilibrator,
    #     column_generator=column_generator,
    #     utility=utility_parameters,
    #     bpr=bpr_parameters,
    #     od=od_parameters
    # )
    #
    # train_results_dfs['model_0b'], test_results_dfs['model_0b'] = model_0b.train(
    #     X_train, Y_train, X_test, Y_test,
    #     # generalization_error={'train': False, 'validation': True},
    #     optimizer=optimizer,
    #     batch_size=_BATCH_SIZE,
    #     loss_weights={'od': 0, 'theta': 0, 'tt': 0, 'flow': 0, 'bpr': 0, 'eq_tt': 1e5},
    #     epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['suelogit'], val_losses=test_results_dfs['suelogit'])

if run_model['ode']:

    print('\nODE: OD estimation with historic OD')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'psc_factor': 0, 'tt':0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': False,
                                                       'tt': False, 'tt_sd': False, 'median_inc': False,
                                                       'incidents': False,
                                                       'bus_stops': False, 'intersections': False
                                                       },
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 initial_values=fresno_network.q.flatten(),
                                 historic_values={10: fresno_network.q.flatten()},
                                 total_trips={0: 1e5},
                                 trainable=True)

    equilibrator = Equilibrator(
        network=fresno_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    ode = PESUELOGIT(
        key='ode',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['ode'], test_results_dfs['ode'] = ode.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        # generalization_error={'train': False, 'validation': True},
        loss_weights= dict(_LOSS_WEIGHTS, od = 0),
        # loss_weights=dict(_LOSS_WEIGHTS, **{'tt': 0, 'od': 0}),
        loss_metric=_LOSS_METRIC,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['ode'], val_losses=test_results_dfs['ode'],
                                xticks_spacing = 50)

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(ode.theta.numpy())))}")
    print(f"alpha = {ode.alpha: 0.2f}, beta  = {ode.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(ode.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['lue']:
    print('\nLUE: Benchmark of pesuelogit and isuelogit (utility only)')

    # _RELATIVE_GAP = 1e-4

    # optimizer = NGD(learning_rate=_LR)
    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': True,
                                                       'tt': True, 'tt_sd': True, 'median_inc': True, 'incidents': True,
                                                              'bus_stops': True, 'intersections': True
                                                       },
                                           )

    utility_parameters.constant_initializer(0)

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 # initial_values=0.6 * tntp_network.q.flatten(),
                                 initial_values=fresno_network.q.flatten(),
                                 true_values=fresno_network.q.flatten(),
                                 historic_values={10: fresno_network.q.flatten()},
                                 trainable=False)

    equilibrator = Equilibrator(
        network=fresno_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    column_generator = ColumnGenerator(equilibrator=equilibrator,
                                       utility=utility_parameters,
                                       n_paths=0,
                                       ods_coverage=0.1,
                                       ods_sampling='sequential',
                                       # ods_sampling='demand',
                                       )

    lue = PESUELOGIT(
        key='lue',
        network=fresno_network,
        # endogenous_flows=False,
        # endogenous_traveltimes=False,
        dtype=tf.float64,
        equilibrator = equilibrator,
        column_generator = column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters
    )

    train_results_dfs['lue'], test_results_dfs['lue'] = lue.train(
        X_train, Y_train, X_test, Y_test,
        # generalization_error={'train': False, 'validation': True},
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        # loss_weights={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        # loss_weights={'od': 0, 'theta': 0, 'tt': 1e10, 'flow': 1, 'eq_flow': 1},
        loss_weights=dict(_LOSS_WEIGHTS, od=0),
        loss_metric=_LOSS_METRIC,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    sns.displot(pd.DataFrame({'fixed_effect':np.array(lue.fixed_effect)}),
            x="fixed_effect", multiple="stack", kind="kde", alpha = 0.8)

    plot_convergence_estimates(estimates=train_results_dfs['lue'].\
                       assign(rr = train_results_dfs['lue']['tt_sd']/train_results_dfs['lue']['tt'])[['epoch','rr']],
                           xticks_spacing = _XTICKS_SPACING)

    plot_predictive_performance(train_losses=train_results_dfs['lue'], val_losses=test_results_dfs['lue'],
                                xticks_spacing = _XTICKS_SPACING)


    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(lue.theta.numpy())))}")
    print(f"alpha = {lue.alpha: 0.2f}, beta  = {lue.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(lue.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['odlue']:

    print('\nODLUE: OD + utility estimation with historic OD')

    # _RELATIVE_GAP = 1e-4\
    # _XTICKS_SPACING = 50

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)
    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': True,
                                                       'tt': True, 'tt_sd': True, 'median_inc': True,
                                                       'incidents': True,
                                                       'bus_stops': True, 'intersections': True
                                                       },
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 initial_values=fresno_network.q.flatten(),
                                 historic_values={10: fresno_network.q.flatten()},
                                 trainable=True)

    equilibrator = Equilibrator(
        network=fresno_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    odlue = PESUELOGIT(
        key='odlue',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['odlue'], test_results_dfs['odlue'] = odlue.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        # generalization_error={'train': False, 'validation': True},
        loss_weights= _LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['odlue'], val_losses=test_results_dfs['odlue'],
                                xticks_spacing = _XTICKS_SPACING)

    plot_convergence_estimates(estimates=train_results_dfs['odlue'].\
                           assign(rr = train_results_dfs['odlue']['tt_sd']/train_results_dfs['odlue']['tt'])[['epoch','rr']],
                               xticks_spacing = _XTICKS_SPACING)

    sns.displot(pd.DataFrame({'fixed_effect':np.array(odlue.fixed_effect)}),
            x="fixed_effect", multiple="stack", kind="kde", alpha = 0.8)

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(odlue.theta.numpy())))}")
    print(f"alpha = {odlue.alpha: 0.2f}, beta  = {odlue.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(odlue.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['odlulpe-1']:

    print('\nODLULPE: ODLUE + link performance parameters with historic OD matrix (link specifics alpha)')

    # _MOMENTUM_EQUILIBRIUM = 0.99

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    # _LR = 1e-2
    # _RELATIVE_GAP = 1e-5
    # _XTICKS_SPACING = 50

    # Some initializations of the bpr parameters, makes the optimization to fail (e.g. alpha =1, beta = 1). Using a common
    # alpha but different betas for every link make the estimation more stable but there is overfitting after a certain amount of iterations

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   # initial_values={'alpha': 0.15*np.ones_like(fresno_network.links,dtype = np.float32),
                                   #                 'beta': 4*np.ones_like(fresno_network.links,dtype = np.float32)},
                                   initial_values={'alpha': 0.15*np.ones_like(fresno_network.links,dtype = np.float32),
                                                   'beta': 4},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 0.15,
                                   #                 'beta': 4 * np.ones_like(fresno_network.links, dtype=np.float32)},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables={'alpha': True, 'beta':False},
                                   # trainables={'alpha': True, 'beta': True},
                                   # trainables={'alpha': False, 'beta': True},
                                   )

    od_parameters = ODParameters(key='od',
                                 initial_values=fresno_network.q.flatten(),
                                 historic_values={10: fresno_network.q.flatten()},
                                 trainable=True)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'psc_factor': 0, 'tt':0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': _FIXED_EFFECT,
                                                       'tt': True, 'tt_sd': True, 'median_inc': True,
                                                       'incidents': True,
                                                       'bus_stops': True, 'intersections': True
                                                       },
                                           time_varying = True
                                           )

    equilibrator = Equilibrator(
        network=fresno_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    column_generator = ColumnGenerator(equilibrator=equilibrator,
                                       utility=utility_parameters,
                                       n_paths=0,
                                       ods_coverage=0.1,
                                       ods_sampling='sequential',
                                       # ods_sampling='demand',
                                       )

    odlulpe_1 = PESUELOGIT(
        key='odlulpe-1',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
        n_periods = len(np.unique(X_train[:,:,-1].numpy().flatten()))
    )

    train_results_dfs['odlulpe-1'], test_results_dfs['odlulpe-1'] = odlulpe_1.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        loss_weights=_LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        momentum_equilibrium = _MOMENTUM_EQUILIBRIUM['odlulpe-1'],
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['odlulpe-1'], val_losses=test_results_dfs['odlulpe-1'],
                                xticks_spacing = _XTICKS_SPACING)

    plot_convergence_estimates(estimates=train_results_dfs['odlulpe-1'][['epoch','alpha']],
                                xticks_spacing = _XTICKS_SPACING)

    sns.displot(pd.DataFrame({'alpha':odlulpe_1.alpha}),
            x="alpha", multiple="stack", kind="kde", alpha = 0.8)

    # Compute utility parameters over time (heatmap) and value of travel time reliability (lineplot)
    theta_df = plot_utility_parameters_periods(odlulpe_1, df, period_feature='hour')
    print(theta_df)

    rr_df = theta_df.apply(compute_rr, axis=1).reset_index().rename(columns={'index': 'hour', 0: 'rr'})
    sns.lineplot(data=rr_df, x='hour', y="rr")

    plot_convergence_estimates(estimates=train_results_dfs['odlulpe-1'].\
                   assign(rr = train_results_dfs['odlulpe-1']['tt_sd']/train_results_dfs['odlulpe-1']['tt'])[['epoch','rr']],
                       xticks_spacing = _XTICKS_SPACING)

    plt.ylabel('average reliability ratio')

    sns.displot(pd.DataFrame({'fixed_effect':np.array(odlulpe_1.fixed_effect)}),
                x="fixed_effect", multiple="stack", kind="kde", alpha = 0.8)

    #print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(odlulpe_1.theta.numpy())))}")
    print(f"theta:\n\n {theta_df}")
    print(f"alpha = {np.mean(odlulpe_1.alpha): 0.2f}, beta  = {np.mean(odlulpe_1.beta): 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(odlulpe_1.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['odlulpe']:

    print('\nODLULPE: ODLUE + link performance parameters with historic OD matrix (link specifics alphas and betas)')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    # _LR = 5e-1
    # _RELATIVE_GAP = 1e-5

    # Some initializations of the bpr parameters, makes the optimization to fail (e.g. alpha =1, beta = 1). Using a common
    # alpha but different betas for every link make the estimation more stable but there is overfitting after a certain amount of iterations

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15*np.ones_like(fresno_network.links,dtype = np.float32),
                                                   'beta': 4*np.ones_like(fresno_network.links,dtype = np.float32)},
                                   # initial_values={'alpha': 0.15*np.ones_like(fresno_network.links,dtype = np.float32),
                                   #                 'beta': 4},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 0.15,
                                   #                 'beta': 4 * np.ones_like(fresno_network.links, dtype=np.float32)},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   # trainables={'alpha': True, 'beta':False},
                                   trainables={'alpha': True, 'beta': True},
                                   # trainables={'alpha': False, 'beta': True},
                                   )

    od_parameters = ODParameters(key='od',
                                 initial_values=fresno_network.q.flatten(),
                                 historic_values={10: fresno_network.q.flatten()},
                                 trainable=True)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'psc_factor': 0, 'tt':0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': True,
                                                       'tt': True, 'tt_sd': True, 'median_inc': True,
                                                       'incidents': True,
                                                       'bus_stops': True, 'intersections': True
                                                       },
                                           )

    equilibrator = Equilibrator(
        network=fresno_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    column_generator = ColumnGenerator(equilibrator=equilibrator,
                                       utility=utility_parameters,
                                       n_paths=0,
                                       ods_coverage=0.1,
                                       ods_sampling='sequential',
                                       # ods_sampling='demand',
                                       )

    odlulpe = PESUELOGIT(
        key='odlulpe',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['odlulpe'], test_results_dfs['odlulpe'] = odlulpe.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        # loss_weights={'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        loss_weights= _LOSS_WEIGHTS,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        loss_metric=_LOSS_METRIC,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['odlulpe'], val_losses=test_results_dfs['odlulpe'],
                                xticks_spacing = _XTICKS_SPACING)

    plot_convergence_estimates(estimates=train_results_dfs['odlulpe'][['epoch','alpha','beta']],
                                xticks_spacing = _XTICKS_SPACING)

    sns.displot(pd.melt(pd.DataFrame({'alpha':odlulpe.alpha, 'beta': odlulpe.beta}), var_name = 'parameters'),
                x="value", hue="parameters", multiple="stack", kind="kde", alpha = 0.8)

    plot_convergence_estimates(estimates=train_results_dfs['odlulpe'].\
                           assign(rr = train_results_dfs['odlulpe']['tt_sd']/train_results_dfs['odlulpe']['tt'])[['epoch','rr']],
                               xticks_spacing = _XTICKS_SPACING)

    sns.displot(pd.DataFrame({'fixed_effect':np.array(odlulpe.fixed_effect)}),
                x="fixed_effect", multiple="stack", kind="kde", alpha = 0.8)

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(odlulpe.theta.numpy())))}")
    print(f"alpha = {np.mean(odlulpe.alpha): 0.2f}, beta  = {np.mean(odlulpe.beta): 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(odlulpe.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['tvodlulpe']:
    print('\ntvodlulpe: Time specific utility and OD, link performance parameters, no historic OD')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': -1, 'tt_sd': -1, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           # trainables={'psc_factor': False, 'fixed_effect': True},
                                           trainables={'psc_factor': False, 'fixed_effect': True,
                                                       'tt': True, 'tt_sd': True,
                                                       'median_inc': False, 'incidents': False,
                                                       'bus_stops': False, 'intersections': False
                                                       },
                                           time_varying=True,
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15*np.ones_like(fresno_network.links,dtype = np.float32),
                                                   'beta': 4*np.ones_like(fresno_network.links,dtype = np.float32)},
                                   trainables={'alpha': True, 'beta': True},
                                   )

    od_parameters = ODParameters(key='od',
                                 initial_values=fresno_network.q.flatten(),
                                 true_values=fresno_network.q.flatten(),
                                 historic_values={10: fresno_network.q.flatten()},
                                 total_trips={0: 1e5, 1: 1e5, 2: 1e5, 9: 1e5, 10: 1e5, 11: 1e5},
                                 time_varying=True,
                                 trainable=True)

    tvodlulpe = PESUELOGIT(
        key='tvodlulpe',
        network=fresno_network,
        dtype=tf.float64,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
        n_periods = len(np.unique(XT_train[:,:,-1].numpy().flatten()))
    )

    train_results_dfs['tvodlulpe'], test_results_dfs['tvodlulpe'] = tvodlulpe.train(
        XT_train, YT_train, XT_test, YT_test,
        optimizer=optimizer,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        loss_weights= _LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    # Plot heatmap with flows of top od pairs
    plot_top_od_flows_periods(tvodlulpe,
                              historic_od= fresno_network.q.flatten(),
                              period_keys = period_keys,
                              period_feature='hour', top_k=20)

    plot_predictive_performance(train_losses=train_results_dfs['tvodlulpe'], val_losses=test_results_dfs['tvodlulpe'],
                               xticks_spacing=_XTICKS_SPACING)

    plot_convergence_estimates(estimates=train_results_dfs['tvodlulpe'][['epoch', 'alpha', 'beta']],
                               xticks_spacing=_XTICKS_SPACING)

    sns.displot(pd.melt(pd.DataFrame({'alpha':tvodlulpe.alpha, 'beta': tvodlulpe.beta}), var_name = 'parameters'),
                x="value", hue="parameters", multiple="stack", kind="kde", alpha = 0.8)

    plt.show()

    # Compute utility parameters over time (heatmap) and value of travel time reliability (lineplot)
    theta_df = plot_utility_parameters_periods(tvodlulpe, period_keys = period_keys, period_feature='hour')

    plot_rr_by_period(tvodlulpe, period_keys, period_feature='hour')

    sns.displot(pd.DataFrame({'fixed_effect': np.array(tvodlulpe.fixed_effect)}),
                x="fixed_effect", multiple="stack", kind="kde", alpha=0.8)

    plt.show()
    
    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(np.mean(tvodlulpe.theta.numpy(),axis = 0))))}")
    print(f"alpha = {np.mean(tvodlulpe.alpha): 0.2f}, beta  = {np.mean(tvodlulpe.beta): 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(tvodlulpe.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['test_tvodlulpe']:
    print('\ntvodlulpe: Time specific utility and OD, link performance parameters, no historic OD')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': 0, 'tt_sd': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'tt_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           # trainables={'psc_factor': False, 'fixed_effect': True},
                                           trainables={'psc_factor': False, 'fixed_effect': True,
                                                       'tt': True, 'tt_sd': True,
                                                       'median_inc': False, 'incidents': False,
                                                       'bus_stops': False, 'intersections': False
                                                       },
                                           time_varying=True,
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15 * np.ones_like(fresno_network.links, dtype=np.float32),
                                                   'beta': 4 * np.ones_like(fresno_network.links, dtype=np.float32)},
                                   trainables={'alpha': True, 'beta': True},
                                   )

    od_parameters = ODParameters(key='od',
                                 initial_values=fresno_network.q.flatten(),
                                 true_values=fresno_network.q.flatten(),
                                 historic_values={10: fresno_network.q.flatten()},
                                 time_varying=True,
                                 trainable=True)

    test_tvodlulpe = PESUELOGIT(
        key='test_tvodlulpe',
        network=fresno_network,
        dtype=tf.float64,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
        n_periods=1)

    random_period = np.random.choice(range(XT_train.shape[0]))

    train_results_dfs['test_tvodlulpe'], test_results_dfs['test_tvodlulpe'] = test_tvodlulpe.train(
        tf.expand_dims(XT_train[random_period,:,:],0), tf.expand_dims(YT_train[random_period,:,:],0), XT_test, YT_test,
        optimizer=optimizer,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        loss_weights=_LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    # Plot heatmap with flows of top od pairs
    plot_top_od_flows_periods(test_tvodlulpe,
                              historic_od= fresno_network.q.flatten(),
                              period_feature='hour', top_k=20)

    plot_predictive_performance(train_losses=train_results_dfs['test_tvodlulpe'], val_losses=test_results_dfs['test_tvodlulpe'],
                                xticks_spacing=_XTICKS_SPACING)

    plot_convergence_estimates(estimates=train_results_dfs['test_tvodlulpe'][['epoch', 'alpha', 'beta']],
                               xticks_spacing=_XTICKS_SPACING)

    sns.displot(pd.melt(pd.DataFrame({'alpha': test_tvodlulpe.alpha, 'beta': test_tvodlulpe.beta}), var_name='parameters'),
                x="value", hue="parameters", multiple="stack", kind="kde", alpha=0.8)

    plt.show()

    # Compute utility parameters over time (heatmap) and value of travel time reliability (lineplot)
    theta_df = plot_utility_parameters_periods(test_tvodlulpe, period_feature='hour')

    plot_rr_by_period(theta_df)

    sns.displot(pd.DataFrame({'fixed_effect': np.array(test_tvodlulpe.fixed_effect)}),
                x="fixed_effect", multiple="stack", kind="kde", alpha=0.8)

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(np.mean(test_tvodlulpe.theta.numpy(), axis=0))))}")
    print(f"alpha = {np.mean(test_tvodlulpe.alpha): 0.2f}, beta  = {np.mean(test_tvodlulpe.beta): 0.2f}")
    print(
        f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(test_tvodlulpe.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")



## Write estimation results

train_results_df, val_results_df \
    = map(lambda x: pd.concat([results.assign(model = model)[['model'] + list(results.columns)]
                               for model, results in x.items()],axis = 0), [train_results_dfs, test_results_dfs])

train_results_df.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_train_results_{'Fresno'}.csv")
val_results_df.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_validation_results_{'Fresno'}.csv")

## Write predictions

predictions = pd.DataFrame({'link_key': list(fresno_network.links_keys) * Y_train.shape[0],
                            'link_type': [link.link_type for link in fresno_network.links] * Y_train.shape[0],
                            'observed_traveltime': Y_train[:, :, 0].numpy().flatten(),
                            'observed_flow': Y_train[:, :, 1].numpy().flatten()})



predictions['date'] = sorted(df[df.hour == 16].loc[df[df.hour == 16].year == 2019, 'date'])

for model in [lue,odlue,odlulpe]:
    # model = odlue
    predicted_flows = model.flows()
    predicted_traveltimes = model.traveltimes()

    predictions['predicted_traveltime_' + model.key] = np.tile(predicted_traveltimes, (Y_train.shape[0], 1)).flatten()
    predictions['predicted_flow_' + model.key] = np.tile(predicted_flows, (Y_train.shape[0], 1)).flatten()

predictions.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_train_predictions_{'Fresno'}.csv")


## Summary of models parameters

models = [lue,odlue,odlulpe, tvodlulpe]
results = pd.DataFrame({'parameter': [], 'model': []})

for model in models:
    results = results.append(pd.DataFrame(
        {'parameter': ['tt'] + features_Z +
                      ['rr'] +
                      ['fixed_effect_mean', 'fixed_effect_std',
                       'alpha_mean', 'alpha_std',
                       'beta_mean', 'beta_std',
                       'od_mean', 'od_std',],
         'values': list(np.mean(model.theta.numpy(),axis =0))  +
                   [float(model.get_parameters_estimates().eval('tt_sd/tt'))] +
                   [np.mean(model.fixed_effect),np.std(model.fixed_effect),
                    np.mean(model.alpha),np.std(model.alpha),
                    np.mean(model.beta),np.std(model.beta),
                    np.mean(model.q),np.std(model.q)]}).\
                             assign(model = model.key)
                             )

# Summary of models goodness of fit

results_losses = pd.DataFrame({})
loss_columns = ['loss_flow', 'loss_tt', 'loss_eq_flow', 'loss_total']

for i, model in enumerate(models):

    results_losses_model = model.split_results(train_results_dfs[model.key])[1].assign(model = model.key)
    results_losses_model = results_losses_model[results_losses_model.epoch == _EPOCHS['learning']].iloc[[0]]
    results_losses = results_losses.append(results_losses_model)

results_losses[loss_columns] = (results_losses[loss_columns]-1)*100

print(results_losses[['model'] + loss_columns].round(1))

## Plot of convergence toward true rr across models

models = [lue,odlue,odlulpe, tvodlulpe]

train_estimates = {}
train_losses = {}

for model in models:
    train_estimates[model.key], train_losses[model.key] = model.split_results(results=train_results_dfs[model.key])

    train_estimates[model.key]['model'] = model.key

train_estimates_df = pd.concat(train_estimates.values())

train_estimates_df['rr'] = train_estimates_df['tt_sd']/train_estimates_df['tt']

estimates = train_estimates_df[['epoch','model','rr']].reset_index().drop('index',axis = 1)
estimates = estimates[estimates.epoch != 0]

fig, ax = plt.subplots(nrows=1, ncols=1)

g = sns.lineplot(data=estimates, x='epoch', hue='model', y='rr')

ax.set_ylabel('reliability ratio')

plt.ylim(ymin=0)

plt.show()

# Plot of relibility ratio by hour for all models

plot_rr_by_period_models(models, period_keys, period_feature='hour')

# Plot of total trips by hour for all models

plot_total_trips_models(models = models, period_feature = 'hour', period_keys = period_keys)
plt.show()