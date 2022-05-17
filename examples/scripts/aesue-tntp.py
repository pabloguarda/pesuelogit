'''
Install isuelogit using pip install -q git+https://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@github.com/pabloguarda/isuelogit.git
'''

import os
from pathlib import Path
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf
import isuelogit as isl
import time

# from src.spad.ueae import StochasticNetworkLoading
from src.spad.aesue import simulate_features, build_tntp_network, simulate_suelogit_data, UtilityFunction, \
    AESUELOGIT, Equilibrator, get_design_tensor, get_y_tensor, load_k_shortest_paths

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:',main_dir)

tntp_network = build_tntp_network(network_name='SiouxFalls')

# links_df = isl.reader.read_tntp_linkdata(network_name='SiouxFalls')
# links_df['link_key'] = [(i, j, '0') for i, j in zip(links_df['init_node'], links_df['term_node'])]
#
# # Link performance functions (assumed linear for consistency with link_cost function definion)
# tntp_network.set_bpr_functions(bprdata=pd.DataFrame({'link_key': tntp_network.links_dict.keys(),
#                                                      'alpha': links_df.b,
#                                                      'beta': links_df.power,
#                                                      'tf': links_df.free_flow_time,
#                                                      'k': links_df.capacity
#                                                      }))

# Paths
load_k_shortest_paths(network=tntp_network, k=2, update_incidence_matrices=True)
# features_Z = []

df = pd.read_csv(main_dir + '/output/network-data/' + tntp_network.key + '/links/' + tntp_network.key + '-link-data.csv')

n_periods = len(df.period.unique())
n_links = len(tntp_network.links)
features_Z = ['c', 's']
n_sparse_features = 3
features_sparse = ['k' + str(i) for i in np.arange(0, n_sparse_features)]
# features_sparse = []


utility_function = UtilityFunction(features_Y=['tt'],
                                   features_Z=features_Z + features_sparse,
                                   true_values={'tt': -1, 'c': -6, 's': -3}
                                   )

input_data = get_design_tensor(Z = df[features_Z + features_sparse],y = df['traveltime'],
                               n_links = n_links, n_periods = n_periods)
traveltime_data = get_design_tensor(y = df['traveltime'],n_links = n_links, n_periods = n_periods)
counts_data = get_y_tensor(y = df[['counts']],n_links = n_links, n_periods = n_periods)

# x = tf.constant(tntp_network.observed_counts_vector.flatten())
# X = tf.expand_dims(x, 0)
#
# rng = tf.random.Generator.from_seed(1234)
# n_samples = 10
# n_links = len(tntp_network.links)
# data = x + rng.normal((n_samples, n_links), mean=0, stddev=0.01, dtype=tf.float64)

model = AESUELOGIT(
    network = tntp_network,
    dtype=tf.float64,
    trainables= {'q': False, 'theta': True, 'beta': False, 'alpha': False},
    utility_function = utility_function,
    inits = {'q': np.ones_like(tntp_network.q.flatten()),
             # 'theta': np.array(list(utility_function.true_values.values())),
             'theta': np.array(list(utility_function.initial_values.values())),
             'beta': np.array([2]),
             'alpha': np.array([1])
             },
)

print(model.trainable_variables)
print(f"q = {model.q}, beta  = {model.beta}")

true_model = AESUELOGIT(network = tntp_network, dtype=tf.float64, utility_function = utility_function)

# print("Difference between observed and predicted counts:", f"{np.sum(np.squeeze(counts_data)-true_model(input_data))}")

print(f"Q = {tf.sparse.to_dense(true_model.Q)}")
# print(f"link cost = {true_model.link_cost(X)}")
# print(f"link logits = {true_model.link_logits(X)}")

optimizer = tf.keras.optimizers.Adam()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)
# optimizer = tf.keras.optimizers.Adagrad()


# Prepare the training and validation dataset.
batch_size = 16

X_train, X_val, y_train, y_val = train_test_split(input_data, traveltime_data, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# Training
loss = tf.reduce_mean(tf.norm(model(input_data)-np.squeeze(traveltime_data)))
epoch = 0
epochs = 1000
t0 = time.time()
lambda_od = tf.constant(1e3,name="lambda_od", dtype=tf.float64)
lambda_theta = tf.constant(1e2,name="lambda_od", dtype=tf.float64)

while loss > 1e-8 and epoch <= epochs:

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:

            loss_traveltimes = tf.reduce_mean(tf.norm(np.squeeze(y_batch_train) - model(x_batch_train)))
            loss_od = lambda_od*tf.norm(model.q-tf.constant(tntp_network.q.flatten()))
            loss_theta = lambda_theta*tf.norm(model.theta,1)
            loss = loss_traveltimes + loss_od + loss_theta

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 10 == 0:
        # print(f"{i}: loss={loss.numpy():0.4g}, demand={model.q}, cost coefficients={model.m}, link flow={x_hat.numpy().mean(0)}, link cost={model.link_cost(data).numpy().mean(0)}")
        # print(f"{i}: loss={loss.numpy():0.4g}, theta = {model.theta.numpy()}")
        print(f"{epoch}: loss={loss_traveltimes.numpy():0.4g}, theta = {model.theta.numpy()}, alpha = {model.alpha.numpy()[0]:0.4g}, beta = {model.beta.numpy()[0]:0.4g}, demand avg abs diff ={np.mean(np.abs(model.q-tntp_network.q.flatten())):0.4g}, time: {time.time()-t0: 0.4g}")

        t0 = time.time()

    epoch += 1

print(f"{epoch} [FINAL]: loss={loss.numpy():0.4g}")

# print(f"{i} [FINAL]: loss={loss.numpy():0.4g}, demand={model.q},cost coefficients={model.m}, link flow={x_hat.numpy().mean(0)}, link cost={model.link_cost(data).numpy().mean(0)}")

alpha_true = np.array([link.bpr.alpha for link in tntp_network.links])
beta_true = np.array([link.bpr.beta for link in tntp_network.links])

print(f"true theta = {utility_function.true_values}",f", theta = {model.theta.numpy()}")
print(f"alpha = {model.alpha}", '\n', f"true alpha = {alpha_true}")
print(f"beta = {model.beta}", '\n', f"true beta = {beta_true}")

# print(f"OD demand = {model.q}", '\n', f"True OD = {tntp_network.q.flatten()}")
print("Avg abs difference between observed and estimated OD:", f"{np.mean(np.abs(model.q-tntp_network.q.flatten()))}")
# np.mean(tntp_network.q)
# np.mean((model(tf.expand_dims(x, 0)).numpy()-true_model(tf.expand_dims(x, 0)).numpy())**2)