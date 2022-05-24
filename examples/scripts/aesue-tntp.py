'''
Install isuelogit using pip install -q git+https://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@github.com/pabloguarda/isuelogit.git
'''

import os
from pathlib import Path
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf
import time

from src.aesuelogit.visualizations import plot_predictive_performance
from src.aesuelogit.models import UtilityFunction, AESUELOGIT, NGD
from src.aesuelogit.networks import load_k_shortest_paths, build_tntp_network, Equilibrator, ColumnGenerator
from src.aesuelogit.etl import get_design_tensor, get_y_tensor, simulate_suelogit_data

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:',main_dir)

tntp_network = build_tntp_network(network_name='SiouxFalls')

# Paths
load_k_shortest_paths(network=tntp_network, k=2, update_incidence_matrices=True)
# features_Z = []

# REad synthethic data which was generated under the assumption of path sets of size 2.
df = pd.read_csv(main_dir + '/output/network-data/' + tntp_network.key + '/links/' + tntp_network.key + '-link-data.csv')

n_days = len(df.period.unique())
n_links = len(tntp_network.links)
n_hours = 1

features_Z = ['c', 's']
# features_Z = []

n_sparse_features = 0
features_sparse = ['k' + str(i) for i in np.arange(0, n_sparse_features)]
# features_sparse = []
features_Z.extend(features_sparse)

utility_function = UtilityFunction(features_Y=['tt'],
                                   features_Z=features_Z,
                                   # true_values={'tt': -1, 'c': -6, 's': -3},
                                   # initial_values={'tt': -4, 'c': -6, 's': -3}
                                   )


# Prepare the training and validation dataset.

# Add free flow travel times
df['tt_ff'] = np.tile([link.bpr.tf for link in tntp_network.links],n_days)

# X_train, X_val, y_train, y_val = train_test_split(input_data, traveltime_data, test_size=0.2, random_state=42)

traveltime_data = get_design_tensor(y=df['traveltime'], n_links=n_links, n_days=n_days, n_hours=n_hours)
flow_data = get_y_tensor(y=df[['counts']], n_links=n_links, n_days=n_days, n_hours=n_hours)

Y = tf.concat([traveltime_data, flow_data], axis=3)
X = get_design_tensor(Z=df[['traveltime'] + features_Z], y=df['tt_ff'], n_links=n_links, n_days=n_days, n_hours=n_hours)


X_train, X_val, Y_train, Y_val = train_test_split(X.numpy(), Y.numpy(), test_size=0.5, random_state=42)

X_train, X_val, Y_train, Y_val = [tf.constant(i) for i in [X_train, X_val, Y_train, Y_val]]

_EPOCHS = 100
_BATCH_SIZE = 4
_LR = 5e-1

# optimizer = NGD(learning_rate=_LR)
# optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)
optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

equilibrator = Equilibrator(
    network=tntp_network,
    # paths_generator=paths_generator,
    utility_function=utility_function,
    max_iters=100,
    method='fw',
    iters_fw=50,
    accuracy=1e-4,
)

column_generator = ColumnGenerator(equilibrator=equilibrator,
                                   utility_function=utility_function,
                                   n_paths=0,
                                   ods_coverage=0.1,
                                   ods_sampling='sequential',
                                   # ods_sampling='demand',
                                   )

train_losses_dfs = {}
val_losses_dfs = {}

# Model 1 (Utility only)
model_1 = AESUELOGIT(
    key='model_1',
    network=tntp_network,
    dtype=tf.float64,
    trainables={'theta': True, 'theta_links': False, 'psc_factor': False, 'q': True, 'alpha': True, 'beta': True},
    equilibrator=equilibrator,
    column_generator=column_generator,
    utility_function=utility_function,
    inits={
        # 'q': tntp_network.q.flatten(),
        'q': 0.6*tntp_network.q.flatten(),
        # 'q': 10*np.ones_like(tntp_network.q.flatten()),
        'theta': np.array(list(utility_function.initial_values.values())),
        'beta': np.array([1]),
        'alpha': np.array([1])
        # 'alpha': np.array([1 for link in fresno_network.links])
    },
)


train_losses_dfs['model_1'], val_losses_dfs['model_1'] = model_1.train(
    X_train, Y_train, X_val, Y_val,
    optimizer=optimizer,
    batch_size=_BATCH_SIZE,
    lambdas={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 0},
    epochs=_EPOCHS)

plot_predictive_performance(train_losses=train_losses_dfs['model_1'], val_losses=val_losses_dfs['model_1'])

