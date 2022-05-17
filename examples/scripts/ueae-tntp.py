''' UEAE with TNTP network

Notes:
    - When q is initialized with zeros there is no gradient, and thus, no updates
    - Projection to the non-negative orthant may be better than the squared root / power trick
    - There are numerical issues and no gradient due to the computation of the logits using the exponential function. When the original q is used, the arguments of the exponential functions are too large and thus, all logits become zero.
    - It would be ideal to generate synthetic traffic counts with SUE-logit consistent with the assumption of full cyclic path sets.
    - In OD estimation, it is assumed that we know which cells of the OD matrix are non-zero
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as it
from typing import Dict

from src.spad.ueae import StochasticNetworkLoading
from src.spad.aesue import build_tntp_network

import isuelogit as isl
from isuelogit.networks import TNetwork

# pip install -q git+https://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@github.com/pabloguarda/isuelogit.git

network_name = 'SiouxFalls'

# Read data from tntp repository and build network object
links_df = isl.reader.read_tntp_linkdata(network_name=network_name)
links_df['link_key'] = [(i, j, '0') for i, j in zip(links_df['init_node'], links_df['term_node'])]

#Build network
network_generator = isl.factory.NetworkGenerator()
A = network_generator.generate_adjacency_matrix(links_keys=list(links_df['link_key'].values))
tntp_network = network_generator.build_network(A=A, network_name=network_name)

# Link performance functions (assumed linear for consistency with link_cost function definion)
tntp_network.set_bpr_functions(bprdata=pd.DataFrame({'link_key': tntp_network.links_dict.keys(),
                                                     'alpha': links_df.b,
                                                     'beta': links_df.power,
                                                     'tf': links_df.free_flow_time,
                                                     'k': links_df.capacity
                                                     }))

# Synthetic link features
linkdata_generator = isl.factory.LinkDataGenerator()
synthetic_features_df = linkdata_generator.simulate_features(links=tntp_network.links,
                                                             features_Z= ['c', 'w', 's'],
                                                             option='continuous',
                                                             range=(0, 1))
# Link features from TNTP repo
link_features_df = links_df[['link_key', 'length', 'speed', 'link_type', 'toll']]

# Load features data
tntp_network.load_features_data(
    linkdata=link_features_df.merge(synthetic_features_df, left_on='link_key', right_on='link_key'))

# Utility function (dependent on travel time only)
utility_function = isl.estimation.UtilityFunction(features_Y=['tt'],
                                                  # features_Z=['c', 's'],
                                                  true_values={'tt': -1, 'c': -6, 's': -3}
                                                  )

# OD matrix
Q = isl.reader.read_tntp_od(network_name=network_name)
tntp_network.load_OD(Q= Q)
tntp_network.scale_OD(scale = 1e-3)

# Paths
isl.factory.PathsGenerator().load_k_shortest_paths(network=tntp_network, k=2)

# Synthetic traffic counts from multiple periods by varying the value of the exogenous attributes instead of adding random noise only)
counts, _ = linkdata_generator.simulate_counts(
    network=tntp_network,
    equilibrator=isl.equilibrium.LUE_Equilibrator(network=tntp_network,
                                                  utility_function=utility_function,
                                                  uncongested_mode=True,
                                                  max_iters=100,
                                                  method='fw',
                                                  iters_fw=100,
                                                  search_fw='grid'),
    noise_params={'mu_x': 0, 'sd_x': 0},
    coverage=1)

tntp_network.load_traffic_counts(counts=counts)

x = tf.constant(tntp_network.observed_counts_vector.flatten())
X = tf.expand_dims(x, 0)

rng = tf.random.Generator.from_seed(1234)
n_samples = 10
n_links = len(tntp_network.links)
data = x + rng.normal((n_samples, n_links), mean=0, stddev=0.01, dtype=tf.float64)

# class SNL(StochasticNetworkLoading):

class SNL(tf.keras.Model):

    def __init__(self,
                 network: TNetwork,
                 trainables: Dict[str, bool] = None,
                 inits: Dict[str, bool] = None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.network = network

        params = ['q', 'm', 'b']

        if trainables is None:
            trainables = {}

        assert set(trainables.keys()).issubset(params)
        self.trainables = {i: trainables.get(i, False) for i in params}

        if inits is None:
            inits = {}

        assert set(inits.keys()).issubset(params)
        self.inits = {i: inits.get(i, None) for i in params}

        q_init = network.q.flatten()

        if self.trainables['q']:
            q_init = inits.get('q', np.zeros_like(q_init))

        m_init = [link.bpr.alpha * link.bpr.tf for link in network.links]

        if self.trainables['m']:
            m_init = inits.get('m', np.ones_like(m_init))

        b_init = [link.bpr.tf for link in network.links]

        self._q = tf.Variable(np.sqrt(q_init), trainable=self.trainables['q'], name="q", dtype=self.dtype)
        self._m = tf.Variable(np.log(m_init), trainable=self.trainables['m'], name="m", dtype=self.dtype)
        self.b = tf.Variable(b_init, trainable=self.trainables['b'], name="b", dtype=self.dtype)

        self.linklist = [(link.key[0],link.key[1]) for link in self.network.links]
        self.n_nodes = self.network.get_n_nodes()
        self.triplist = self.network.ods
        self.I = np.identity(self.n_nodes)

    @property
    def m(self):
        return tf.exp(self._m)

    @property
    def q(self):
        return self._q ** 2

    def link_cost(self, x):
        return tf.multiply(self.m, x) + self.b

    def link_logits(self, x):
        #TODO: this operation has numerical issues because the exponential. A normalization should be performed before applying the exponential. Look Eq. A.58 Boyles book
        return tf.exp(-self.link_cost(x))

    def transition_logits(self, x):
        n_samples = len(x)
        indices = [
            list(it.chain([i], adj))
            for i, adj in it.product(range(n_samples), self.linklist)
        ]
        return tf.SparseTensor(
            indices=indices,
            values=tf.reshape(self.link_logits(x), [-1]),
            dense_shape=(n_samples, self.n_nodes, self.n_nodes)
        )

    @property
    def Q(self):
        return tf.SparseTensor(
            indices=self.triplist,
            values=self.q,
            dense_shape=(self.n_nodes, self.n_nodes)
        )

    def link_weights(self, L):
        _L = tf.sparse.to_dense(L)
        Q = tf.sparse.to_dense(self.Q)
        V = tf.linalg.inv(self.I - _L)
        _V = 1.0 / tf.where(tf.equal(V, 0.0), 1.0, V)
        return tf.einsum("iru, ivs, irs, rs -> iuv", V, V, _V, Q)

    def normalizer(self, L):
        # TODO support multiple OD: tf.einsum("iuv, iru, ivs, irs, rs -> iuv", L, V, V, 1.0/V, Q)
        V = tf.linalg.inv(self.I - tf.sparse.to_dense(L))
        V_rs = 1.0 / V[:, self.r, self.s]
        V_r = V[:, self.r, :]
        V_s = V[:, :, self.s]
        return tf.einsum("iu, iv, i -> iuv", V_r, V_s, V_rs)

    def link_assignment(self, x):
        L = self.transition_logits(x)
        W = self.link_weights(L)
        H = L * W
        return tf.reshape(H.values, [len(x), -1])

    def call(self, x):
        x = tf.cast(x, self.dtype)
        return self.link_assignment(x)


model = SNL(network = tntp_network, dtype=tf.float64,
            trainables= {'q': False, 'm': True}, inits = {'q': np.ones_like(tntp_network.q.flatten()) })
print(model.trainable_variables)
print(f"q = {model.q}, m  = {model.m}")

true_model = SNL(network = tntp_network, dtype=tf.float64)


print(f"{X} =?= {true_model(X)}")
print(f"Q = {tf.sparse.to_dense(true_model.Q)}")
print(f"link cost = {true_model.link_cost(X)}")
print(f"link logits = {true_model.link_logits(X)}")
print(f"L = {tf.sparse.to_dense(true_model.transition_logits(X))}")
print(f"W = {true_model.link_weights(true_model.transition_logits(X))}")


optimizer = tf.keras.optimizers.Adam()

loss = tf.reduce_mean((data - model(data))**2)
i = 0
while loss > 1e-8 and i <= 5000:
    with tf.GradientTape() as tape:
        x_hat = model(data)
        loss = (
            tf.reduce_mean((data - x_hat)**2)
        )
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if i % 1000 == 0:
        # print(f"{i}: loss={loss.numpy():0.4g}, demand={model.q}, cost coefficients={model.m}, link flow={x_hat.numpy().mean(0)}, link cost={model.link_cost(data).numpy().mean(0)}")
        print(f"{i}: loss={loss.numpy():0.4g}")
    i += 1
print(f"{i} [FINAL]: loss={loss.numpy():0.4g}, demand={model.q},cost coefficients={model.m}, link flow={x_hat.numpy().mean(0)}, link cost={model.link_cost(data).numpy().mean(0)}")

m_true = np.array([link.bpr.alpha * link.bpr.tf for link in tntp_network.links])

print(f"learned flow = {model(tf.expand_dims(x, 0)).numpy()}",'\n',f"flow with true model = {true_model(tf.expand_dims(x, 0)).numpy()}", '\n', f" true flow = {x} ")
print(f"Learned cost = {model.link_cost(x).numpy()}; true cost = {true_model.link_cost(x).numpy()}")
# print(f"learned path cost = {linkpath.T @  model.link_cost(x).numpy()}; true path cost = {linkpath.T @ true_model.link_cost(x).numpy()}")
print(f"m = {model.m}", '\n', f"true m = {m_true}")
print(f"b = {model.b}")
print(f"OD demand = {model.q}", '\n', f"True OD = {tntp_network.q.flatten()}")

np.mean((model(tf.expand_dims(x, 0)).numpy()-true_model(tf.expand_dims(x, 0)).numpy())**2)