'''

Install isuelogit using pip install -q git+https://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@github.com/pabloguarda/isuelogit.git

- Notes:
        - Think on how to incorporate traffic count data, e.g. an additional term in the loss function

'''

import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
import isuelogit as isl
import glob

from src.aesuelogit.aesue import UtilityFunction, AESUELOGIT, Equilibrator, plot_predictive_performance
from src.aesuelogit.networks import load_k_shortest_paths, read_paths,build_fresno_network
from src.aesuelogit.etl import get_design_tensor, get_y_tensor, simulate_suelogit_data

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:', main_dir)

isl.config.dirs['read_network_data'] = "input/network-data/fresno/"

# To write estimation report and set seed for all algorithms involving some randomness
# estimation_reporter = isl.writer.Reporter(
#     folderpath=isl.config.dirs['output_folder'] + 'estimations/' + 'Fresno', seed=_SEED)

fresno_network = build_fresno_network()

# Paths
# load_k_shortest_paths(network=fresno_network, k=2, update_incidence_matrices=True)
# read_paths(network=fresno_network, update_incidence_matrices=True, filename = 'paths-full-model-fresno.csv')
read_paths(network=fresno_network, update_incidence_matrices=True, filename='paths-fresno.csv')

# Read spatiotemporal data
folderpath = isl.config.dirs['read_network_data'] + 'links/spatiotemporal-data/'
df = pd.concat([pd.read_csv(file) for file in glob.glob(folderpath + "*fresno-link-data*")], axis=0)

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df[df['date'].dt.dayofweek.between(0, 4)]
# df = df[df['date'].dt.year == 2019]

df['period'] = df['date'].astype(str) + '-' + df['hour'].astype(str)
df['period'] = df.period.map(hash)

# df1 = pd.read_csv(main_dir + '/input/network-data/' + fresno_network.key + '/links/2019-10-01-fresno-link-data.csv')
# df1['date'] = "2019-10-01"
# df1['period'] = 0
#
# df2 = pd.read_csv(main_dir + '/input/network-data/' + fresno_network.key + '/links/2020-10-06-fresno-link-data.csv')
# df2['date'] = "2020-10-06"
# df2['period'] = 1
#
# df = pd.concat([df1, df2], axis=0)

features_Z = ['speed_sd', 'median_inc', 'incidents', 'bus_stops', 'intersections']

# Data cleaning
# TODO: Perform data imputation of links of time LWLK with travel times equal to 0
df.loc[df['counts'] <= 0, "counts"] = np.nan

# Normalization of features to range [0,1] (TODO: enable this option in get_design_tensor method or see if tensorflow have it)
df[features_Z + ['tt_avg']] = preprocessing.MinMaxScaler().fit_transform(df[features_Z + ['tt_avg']])

utility_function = UtilityFunction(features_Y=['tt'],
                                   features_Z=features_Z,
                                   signs={'tt': '-', 'speed_sd': '-', 'incidents': '-', 'intersections': '-',
                                          'bus_stops': '-', 'median_inc': '+'},
                                   # initial_values={'tt': 0, 'median_inc': 0, 'incidents': 0}
                                   )

utility_function.constant_initializer(0)

# TODO: Make sure the dataframe is properly sorted before doing the reshaping into a tensor of multiple dimensions

n_links = len(fresno_network.links)
df['year'] = df.date.dt.year
X, Y = {}, {}

for year in sorted(df['year'].unique()):
    df_year = df[df['year'] == year]
    n_days, n_hours = len(df_year.date.unique()), len(df_year.hour.unique())

    traveltime_data = get_design_tensor(y=df_year['tt_avg'], n_links=n_links, n_days=n_days, n_hours=n_hours)
    flow_data = get_y_tensor(y=df_year[['counts']], n_links=n_links, n_days=n_days, n_hours=n_hours)

    Y[year] = tf.squeeze(tf.stack([traveltime_data, flow_data], axis=3))
    X[year] = get_design_tensor(Z=df_year[['tf_inrix'] + features_Z], y=df_year['tt_avg'],
                                n_links=n_links, n_days=n_days, n_hours=n_hours)


# Prepare the training and validation dataset.
# X, Y = tf.concat(list(X.values()),axis = 0), tf.concat(list(Y.values()),axis = 0)
# X_train, X_val, Y_train, Y_val = train_test_split(X.numpy(), Y.numpy(), test_size=0.5, random_state=42)
X_train, X_val, Y_train, Y_val = X[2019], X[2020], Y[2019], Y[2020]

X_train, X_val, Y_train, Y_val = [tf.constant(i) for i in [X_train, X_val, Y_train, Y_val]]

train_losses_dfs = {}
val_losses_dfs = {}

_EPOCHS = 10
_BATCH_SIZE = 4
_LR = 1e-1 #Default is 1e-3. With 1e-1, training becomes unstable

# models = dict(zip(['m1', 'm2', 'm3', 'm4'], True))
models = dict.fromkeys(['m1', 'm2', 'm3', 'm4'], False)
# models['m1'] = True
# models['m2'] = True
# models['m3'] = True
models['m4'] = True

if models['m1']:

    print('\nmodel 1\n')

    # Model 1 (Utility only)
    model_1 = AESUELOGIT(
        network=fresno_network,
        dtype=tf.float64,
        trainables={'theta': True, 'theta_links': False, 'q': False, 'alpha': False, 'beta': False, },
        utility_function=utility_function,
        inits={
            'q': fresno_network.q.flatten(),
            # 'q': 10*np.ones_like(fresno_network.q.flatten()),
            'theta': np.array(list(utility_function.initial_values.values())),
            'beta': np.array([4]),
            'alpha': np.array([0.15])
            # 'alpha': np.array([1 for link in fresno_network.links])
        },
    )

    train_losses_dfs['model_1'], val_losses_dfs['model_1'] = model_1.train(
        X_train, Y_train, X_val, Y_val,
        optimizer = tf.keras.optimizers.Adam(learning_rate=_LR),
        batch_size = _BATCH_SIZE,
        lambdas = {'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 1},
        epochs = _EPOCHS)

    plot_predictive_performance(train_losses_df=train_losses_dfs['model_1'], val_losses_df=val_losses_dfs['model_1'])

    print(f"theta = {dict(zip(utility_function.true_values.keys(), list(model_1.theta.numpy())))}")
    print(f"alpha = {model_1.alpha}, beta  = {model_1.beta}")
    print("Avg abs diff between observed and estimated OD:",f"{np.mean(np.abs(model_1.q - fresno_network.q.flatten()))}")

if models['m2']:
    print('\nmodel 2\n')

    # Model 2 (Utility and ODE with historic OD)
    model_2 = AESUELOGIT(
        network=fresno_network,
        dtype=tf.float64,
        trainables={'theta': True, 'theta_links': False, 'q': True, 'alpha': False, 'beta': False},
        utility_function=utility_function,
        inits={
            'q': fresno_network.q.flatten(),
            'theta': np.array(list(utility_function.initial_values.values())),
            'beta': np.array([4]),
            'alpha': np.array([0.15])
        },
    )

    train_losses_dfs['model_2'], val_losses_dfs['model_2'] = model_2.train(
        X_train, Y_train, X_val, Y_val,
        optimizer = tf.keras.optimizers.Adam(learning_rate=_LR),
        batch_size = _BATCH_SIZE,
        lambdas = {'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 0},
        epochs = _EPOCHS)

    plot_predictive_performance(train_losses_df=train_losses_dfs['model_2'], val_losses_df=val_losses_dfs['model_2'])

    print(f"theta = {dict(zip(utility_function.true_values.keys(), list(model_2.theta.numpy())))}")
    print(f"alpha = {model_2.alpha}, beta  = {model_2.beta}")
    print("Avg abs diff between observed and estimated OD:",f"{np.mean(np.abs(model_2.q - fresno_network.q.flatten()))}")

if models['m3']:
    print('\nmodel 3\n')

    # Model 3 (Utility, ODE, link performance parameters, no historic OD)
    model_3 = AESUELOGIT(
        network=fresno_network,
        dtype=tf.float64,
        trainables={'theta': True, 'theta_links': False, 'q': True, 'alpha': True, 'beta': True},
        utility_function=utility_function,
        inits={
            'q': fresno_network.q.flatten(),
            'theta': np.array(list(utility_function.initial_values.values())),
            'beta': np.array([4]),
            'alpha': np.array([0.15])
        },
    )

    train_losses_dfs['model_3'], val_losses_dfs['model_3'] = model_3.train(
        X_train, Y_train, X_val, Y_val,
        optimizer = tf.keras.optimizers.Adam(learning_rate=_LR),
        batch_size = _BATCH_SIZE,
        lambdas = {'od': 0, 'theta': 0, 'tt': 0, 'flow': 1, 'bpr': 0},
        epochs = _EPOCHS)

    plot_predictive_performance(train_losses_df=train_losses_dfs['model_3'], val_losses_df=val_losses_dfs['model_3'])

    print(f"theta = {dict(zip(utility_function.true_values.keys(), list(model_3.theta.numpy())))}")
    print(f"alpha = {model_3.alpha}, beta  = {model_3.beta}")
    print("Avg abs diff between observed and estimated OD:",f"{np.mean(np.abs(model_3.q - fresno_network.q.flatten()))}")


if models['m4']:
    print('\nmodel 4\n')

    # Model 4 (Time specific Utility and ODE, link performance parameters, no historic OD)
    model_4 = AESUELOGIT(
        network=fresno_network,
        dtype=tf.float64,
        trainables={'theta': True, 'theta_links': False, 'q': True, 'alpha': True, 'beta': True},
        utility_function=utility_function,
        inits={
            'q': np.repeat(fresno_network.q.flatten()[np.newaxis,:],n_hours,axis =0),
            'theta': np.repeat(np.array(list(utility_function.initial_values.values()))[np.newaxis,:],n_hours,axis =0),
            'beta': np.array([4]),
            'alpha': np.array([0.15])
        },
    )
    train_losses_dfs['model_4'], val_losses_dfs['model_4'] = model_4.train(
        X_train, Y_train, X_val, Y_val,
        optimizer = tf.keras.optimizers.Adam(learning_rate=_LR),
        batch_size = _BATCH_SIZE,
        lambdas = {'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 0},
        epochs = _EPOCHS)

    plot_predictive_performance(train_losses_df=train_losses_dfs['model_4'], val_losses_df=val_losses_dfs['model_4'])

    print(f"features = {utility_function.features}")
    print(f"theta = {model_4.theta.numpy()}")
    print(f"alpha = {model_4.alpha}, beta  = {model_4.beta}")
    print("Avg abs diff between observed and estimated OD:",f"{np.mean(np.abs(model_4.q - fresno_network.q.flatten()))}")

# model.predict_flow(X_val)

# Estimation results
# print(model.trainable_variables)

# theta = model.theta.numpy()
# if theta.ndim == 1:
#     print(f"theta = {dict(zip(utility_function.true_values.keys(), list(model.theta.numpy())))}")
# else:
#     print(f"features = {utility_function.features}")
#     print(f"theta = {model.theta.numpy()}")

# print(f"OD demand = {model.q}", '\n', f"True OD = {tntp_network.q.flatten()}")