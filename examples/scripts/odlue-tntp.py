'''

Install isuelogit using pip install -q git+https://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@github.com/pabloguarda/isuelogit.git

'''

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import isuelogit as isl
# from isuelogit.equilibrium import LUE_Equilibrator

# from src.spad.ueae import StochasticNetworkLoading
# from src.aesuelogit.aesue import simulate_features, build_tntp_network, simulate_suelogit_data, UtilityFunction, \
#     ODLUE, Equilibrator, get_design_tensor, get_y_tensor

from src.aesuelogit.models import UtilityFunction, ODLUE, Equilibrator, NGD
from src.aesuelogit.networks import build_tntp_network, build_small_network
from src.aesuelogit.etl import get_design_tensor, get_y_tensor, simulate_suelogit_data, simulate_features

# Path management
main_dir = str(Path(os.path.abspath('')).parents[0])
os.chdir(main_dir)
print('main dir:',main_dir)

tntp_network = build_tntp_network(network_name='SiouxFalls')
# tntp_network = build_small_network(network_name = 'Yang')

# Paths
isl.factory.PathsGenerator().load_k_shortest_paths(network=tntp_network, k=2)

n_days = 1
n_links = len(tntp_network.links)
# features_Z = ['c', 's']
features_Z = []

exogenous_features = simulate_features(links=tntp_network.links,
                                       features_Z= features_Z,
                                       option='continuous',
                                       range=(0, 1),
                                       n_days = n_days)

utility_function = UtilityFunction(features_Y=['tt'],
                                   features_Z=features_Z,
                                   true_values={'tt': -1, 'c': -6, 's': -3},
                                   initial_values={'tt': -5, 'c': -6, 's': -3}
                                   )

equilibrator = Equilibrator(network=tntp_network,
                            utility_function=utility_function,
                            # uncongested_mode=True,
                            # exogenous_traveltimes=True,
                            accuracy=1e-10,
                            max_iters=100,
                            method='fw',
                            iters_fw=100,
                            search_fw='grid')

# Generate data from multiple days by varying the value of the exogenous attributes instead of adding random noise only
df = simulate_suelogit_data(
    days= list(exogenous_features.period.unique()),
    features_data = exogenous_features,
    equilibrator=equilibrator,
    network = tntp_network)

input_data = get_design_tensor(Z = df[['traveltime'] + features_Z],n_links = n_links, n_days = n_days)

counts_data = get_y_tensor(y= df[['counts']],n_links = n_links, n_days = n_days)

# X = tf.constant(tntp_network.observed_counts_vector.flatten())
# X = tf.expand_dims(X, 0)
#
# rng = tf.random.Generator.from_seed(1234)
# n_samples = 10
# n_links = len(tntp_network.links)
# data = X + rng.normal((n_samples, n_links), mean=0, stddev=0.01, dtype=tf.float64)

model = ODLUE(
    network = tntp_network,
    dtype=tf.float64,
    trainables= {'q': True, 'theta': True},
    utility_function = utility_function,
    inits = {'q': np.ones_like(tntp_network.q.flatten()),
             'theta': np.array(list(utility_function.initial_values.values()))},
)

print(model.trainable_variables)
print(f"q = {model.q}")

# true_model = ODLUE(network = tntp_network, dtype=tf.float64, utility_function = utility_function)

# print("Difference between observed and predicted counts:", f"{np.sum(np.squeeze(counts_data)-true_model(input_data))}")
#
# print(f"Q = {tf.sparse.to_dense(true_model.Q)}")
# print(f"link cost = {true_model.link_cost(X)}")
# print(f"link logits = {true_model.link_logits(X)}")

optimizer = tf.keras.optimizers.Adagrad(learning_rate=5*1e-1)
# optimizer = NGD(learning_rate=2*1e-1)

loss = tf.reduce_mean((np.squeeze(counts_data) - model(input_data))**2)
i = 0
while loss > 1e-8 and i <= 5000:

    if i % 1 == 0:
        # print(f"{i}: loss={loss.numpy():0.4g}, demand={model.q}, cost coefficients={model.m}, link flow={x_hat.numpy().mean(0)}, link cost={model.link_cost(data).numpy().mean(0)}")
        # print(f"{i}: loss={loss.numpy():0.4g}, theta = {model.theta.numpy()}")
        print(f"{i}: loss={loss.numpy():0.4g}, theta = {model.theta.numpy()}, demand avg abs diff ={np.mean(np.abs(model.q-tntp_network.q.flatten()))}")

    with tf.GradientTape() as tape:
        x_hat = model(input_data)
        loss = (
            tf.reduce_mean((np.squeeze(counts_data) - x_hat)**2)
        )
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    i += 1

print(f"{i} [FINAL]: loss={loss.numpy():0.4g}")

# print(f"{i} [FINAL]: loss={loss.numpy():0.4g}, demand={model.q},cost coefficients={model.m}, link flow={x_hat.numpy().mean(0)}, link cost={model.link_cost(data).numpy().mean(0)}")

m_true = np.array([link.bpr.alpha * link.bpr.tf for link in tntp_network.links])

print(f"true theta = {utility_function.true_values}",f", theta = {model.theta.numpy()}")
# print(f"m = {model.m}", '\n', f"true m = {m_true}")
# print(f"b = {model.b}")
# print(f"OD demand = {model.q}", '\n', f"True OD = {tntp_network.q.flatten()}")
print("Avg abs difference between observed and estimated OD:", f"{np.mean(np.abs(model.q-tntp_network.q.flatten()))}")
# np.mean(tntp_network.q)
# np.mean((model(tf.expand_dims(X, 0)).numpy()-true_model(tf.expand_dims(X, 0)).numpy())**2)