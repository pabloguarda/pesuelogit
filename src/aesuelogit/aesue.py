"""
Module for AutoEncoded Stochastic User Equilibrium with Logit Assignment (ODLUE)
"""

import isuelogit as isl
from isuelogit.networks import TNetwork
from isuelogit.printer import block_output, printIterationBar
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple
import ast
import time
from sklearn import preprocessing
import glob
import os
import matplotlib.pyplot as plt


class UtilityFunction(isl.estimation.UtilityFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Equilibrator(isl.equilibrium.LUE_Equilibrator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ODLUE(tf.keras.Model):

    # ODLUE is an extension of the logit utility estimation (LUE) problem solved in the isuelogit package.  ODLUE is
    # fed with link features of multiple time periods and it allows for the estimation of the OD matrix and of the
    # utility function parameters. ODLUE is framed as a computational graph which facilitates the computation of gradient
    # of the loss function.

    # The current implementation assumes that the path sets and the OD matrix is the same for all time periods.
    # However, if the path set were updated dynamically, the path sets would vary among time periods and od pairs over
    # iterations. This means that the matrices M and C will change frequently, which may be computationally intractable.

    def __init__(self,
                 network: TNetwork,
                 utility_function: UtilityFunction,
                 trainables: Dict[str, bool] = None,
                 inits: Dict[str, bool] = None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.n_feature = None
        self.n_links = None
        self.n_days = None
        self.n_hours = None
        self.network = network

        self.utility_function = utility_function

        params = ['q', 'alpha', 'beta', 'theta', 'theta_links']

        if trainables is None:
            trainables = {}

        # assert set(trainables.keys()).issubset(params)
        self.trainables = {i: trainables.get(i, False) for i in params}

        self.inits = inits

        if self.inits is None:
            self.inits = {}

        assert set(self.inits.keys()).issubset(params)
        # self.inits = {i: inits.get(i, None) for i in params}

        q_init = self.network.q.flatten()

        if self.trainables['q']:
            q_init = self.inits.get('q', np.ones_like(q_init))

        theta_init = np.array(list(self.utility_function.true_values.values()))

        if self.trainables['theta'] or None in theta_init:
            theta_init = self.inits.get('theta', np.array(list(self.utility_function.initial_values.values())))

        self._q = tf.Variable(np.sqrt(q_init), trainable=self.trainables['q'], name="q", dtype=self.dtype)
        # self._q = self._q[tf.newaxis,:]

        self._theta = tf.Variable(theta_init, trainable=self.trainables['theta'], name="theta", dtype=self.dtype)
        self._theta_links = tf.Variable(tf.zeros(len(self.network.links), tf.float64),
                                        trainable=self.trainables['theta_links'], name="theta_links", dtype=self.dtype)

        # self.linklist = [(link.key[0], link.key[1]) for link in self.network.links]
        self.n_nodes = self.network.get_n_nodes()
        # self.triplist = self.network.ods
        # self.I = np.identity(self.n_nodes)

        # TODO: Generate D matrix with linklist and a sparse tensor

        # Incidence matrices (assumes paths sets are the same over time periods and iterations)
        self.D = tf.constant(self.network.D, dtype=self.dtype)
        self.M = tf.constant(self.network.M, dtype=self.dtype)
        self.C = tf.constant(self.network.C, dtype=self.dtype)
        # self.C = self.network.generate_C(self.M)

    @property
    def q(self):
        return tf.math.pow(self._q, 2)

    @property
    def Q(self):
        return tf.SparseTensor(
            indices=self.triplist,
            values=self.q,
            dense_shape=(self.n_nodes, self.n_nodes)
        )

    @property
    def theta_links(self):
        return self._theta_links

    def project_theta(self, theta):
        clips_min = []
        clips_max = []

        signs = self.utility_function.signs

        for feature in self.utility_function.features:

            sign = signs[feature]
            if sign == '+':
                clips_min.append(0)
                clips_max.append(tf.float64.max)
            if sign == '-':
                clips_min.append(tf.float64.min)
                clips_max.append(0)

        return tf.clip_by_value(theta, clips_min, clips_max)

    @property
    def theta(self):
        return self.project_theta(self._theta)

    def normalizer(self, vf):

        '''
        This operation is computationally expensive
        # np.nanmax is not available in tensorflow, which introduces inneficiency
        # This function can be eliminated

        '''
        C_nan = np.where(self.C.astype('float') == 0, float('nan'), self.C)
        v_max = np.nanmax(np.einsum("ij,jk -> ijk", vf, C_nan), axis=1)
        return vf - v_max
        # v_max = np.nanmax(vf * np.where(self.C.astype('float') == 0, float('nan'), self.C), axis=1)
        # return (vf - v_max)[:, np.newaxis]

    def link_utilities(self, X):

        """ TODO: Make the einsum operation in one line"""

        if tf.rank(self.theta) == 1:
            return tf.einsum("ijkl,l -> ijk", X, self.theta) + self._theta_links
        else:
            return tf.einsum("ijkl,jl -> ijk", X, self.theta) + self._theta_links

    def path_utilities(self, V):
        return tf.einsum("ijk,kl -> ijl", V, self.D)

    # def path_attributes(self, X):
    #     return tf.einsum("ijk,jl -> ilk", X, self.D)
    #     # return tf.tensordot(tf.transpose(self.D),X,1)
    #
    # def path_utilities(self, Xf):
    #     # return tf.tensordot(Xf, self.theta)
    #     return tf.einsum("ijk,k -> ij", Xf, self.theta)Link specific effects are also related to a panel fixed effect

    def path_probabilities_sparse(self, vf, normalization=False):
        ''' Sparse version. Computation time is roughly the same than non-sparse version but it does not require
        to store the matrix C which has dimensions n_paths x n_paths
        tf.sparse.reduce_max has no gradient registered, thus this op is ignored from the backprop:
        https://www.tensorflow.org/api_docs/python/tf/stop_gradient

        #TODO: Optimize repetition of M_sparse matrix over days and hours dimensions

        '''

        M_sparse = tf.cast(tf.sparse.from_dense(self.network.M), tf.float64)
        M_sparse = tf.sparse.concat(0, [tf.sparse.expand_dims(M_sparse, 0)] * vf.shape[1])
        M_sparse = tf.sparse.concat(0, [tf.sparse.expand_dims(M_sparse, 0)] * vf.shape[0])

        indices = M_sparse.indices

        V = tf.sparse.SparseTensor(indices=indices,
                                   # values = tf.exp(tf.reshape(vf,-1)),
                                   values=tf.reshape(vf, -1),
                                   dense_shape=(vf.shape[0], vf.shape[1], *self.M.shape))

        if normalization:
            normalized_values = V.values - tf.reshape(
                tf.einsum("ijk,kl -> ijl", tf.stop_gradient(tf.sparse.reduce_max(V, axis=3)), self.M), -1)
            V = tf.sparse.SparseTensor(indices=indices, values=tf.exp(normalized_values),
                                       dense_shape=(vf.shape[0], vf.shape[1], *self.M.shape))

        else:
            V = tf.sparse.map_values(tf.exp, V)

        return tf.reshape(V.values, vf.shape) / tf.einsum("ijk,kl -> ijl", tf.sparse.reduce_sum(V, axis=3), self.M)

        # return tf.exp(vf)/tf.einsum("ij,jk -> ik", tf.sparse.reduce_sum(V, axis=2), self.M)

    def path_probabilities(self, vf, sparse_mode=True, normalization=True):

        if sparse_mode:
            return self.path_probabilities_sparse(vf, normalization=normalization)

        else:
            # TODO: Readapt this code to account for the hurs dimension or simply remove this as it is not used by default
            if normalization:
                vf = self.normalizer(vf)

            return tf.einsum("ij,ij -> ij", tf.exp(vf), 1 / (tf.einsum("ij,jl -> il", tf.exp(vf), self.C)))

        # return tf.divide(tf.exp(vf), tf.tensordot(self.C, tf.exp(vf)) + epsilon,1)

    def path_flows(self, pf):
        # TODO: Test and try to combine the einsums if possible

        if tf.rank(self.q) == 1:
            return tf.einsum("ij,i, klj -> klj", self.M, self.q, pf)
            # return tf.einsum("j,lij -> lij", tf.einsum("ij,i-> j", self.M, self.q), pf)

        else:
            # bad implemented: return tf.einsum("ij,ki, lkj -> lij", self.M, self.q, pf)
            return tf.einsum("ij, lij -> lij", tf.einsum("ij,ki-> kj", self.M, self.q), pf)

    def link_flows(self, f):
        return tf.einsum("ijk,lk -> ijl", f, self.D)

    def call(self, X):
        """
        X is tensor of dimension (n_days, n_hours, n_links, n_features)
        """

        self.n_days, self.n_hours, self.n_links, self.n_feature = X.shape

        X = tf.cast(X, self.dtype)

        return self.link_flows(self.path_flows(self.path_probabilities(self.path_utilities(self.link_utilities(X)))))


class AESUELOGIT(ODLUE):

    def __init__(self,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # Free flow travel time
        self._tt_ff = np.array([link.bpr.tf for link in self.network.links])

        self._epsilon = 1e-12

        alpha_init = np.array([link.bpr.alpha for link in self.network.links])

        if self.trainables['alpha']:
            alpha_init = self.inits.get('alpha', np.ones_like(alpha_init))

        beta_init = np.array([link.bpr.beta for link in self.network.links])

        if self.trainables['beta']:
            beta_init = self.inits.get('beta', np.ones_like(beta_init))

        self._alpha = tf.Variable(np.log(alpha_init), trainable=self.trainables['alpha'], name="alpha",
                                  dtype=self.dtype)
        self._beta = tf.Variable(beta_init, trainable=self.trainables['beta'], name="beta", dtype=self.dtype)

    @property
    def alpha(self):
        # TODO: Check that this transformation still allows for perfect parameter recovery in Sioux Falls
        return tf.exp(self._alpha)

    @property
    def beta(self):
        return tf.clip_by_value(self._beta, 0 + self._epsilon, 4)
        # return tf.exp(self._beta)

    @property
    def tt_ff(self):
        return self._tt_ff

    @tt_ff.setter
    def tt_ff(self, value):
        self._tt_ff = value

    def link_traveltimes(self, x):
        """
        Link performance function
        """

        # Links capacities
        k = np.array([link.bpr.k for link in self.network.links])

        return self.tt_ff * (1 + self.alpha * tf.math.pow(x / k, self.beta))

    def predict_flow(self, X):

        # TODO: enable use of spatiotemporal travel in ODLUE

        self.tt_ff, feature_data = X[:, :, :, 0], X[:, :, :, 1:]

        # Add RELU layer to avoid negative flows into input of BPR function with exponent
        return tf.keras.activations.relu(super(AESUELOGIT, self).call(feature_data))

    def loss_function(self, X, Y, lambdas: Dict[str, float]):
        """
        Return a dictionary with the different type of loss as keys

        # TODO: I may add an option in this function to specify which loss function to use
        """

        lambdas_vals = {'tt': 1.0, 'od': 0.0, 'theta': 0.0, 'flow': 0.0, 'bpr': 0.0}

        assert set(lambdas.keys()).issubset(lambdas_vals.keys()), 'Invalid key in lambdas attribute'

        for attr, val in lambdas.items():
            lambdas_vals[attr] = val

        lambdas = {'od': tf.constant(lambdas_vals['od'], name="lambda_od", dtype=tf.float64),
                   'theta': tf.constant(lambdas_vals['theta'], name="lambda_theta", dtype=tf.float64),
                   'flow': tf.constant(lambdas_vals['flow'], name="lambda_flow", dtype=tf.float64),
                   'tt': tf.constant(lambdas_vals['tt'], name="lambda_tt", dtype=tf.float64),
                   'bpr': tf.constant(lambdas_vals['bpr'], name="lambda_bpr", dtype=tf.float64)
                   }

        tt, flow = Y[:, :, :, 0], Y[:, :, :, 1]
        pred_flow = self.predict_flow(X)

        # loss_metric = btcg_mse
        loss_metric = mse

        loss = {'od': loss_metric(self.q, tf.constant(self.network.q.flatten())),
                'theta':tf.reduce_mean(tf.norm(self.theta, 1)),
                'flow':  loss_metric(tf.squeeze(flow), pred_flow),
                'tt': loss_metric(tf.squeeze(tt), self.link_traveltimes(pred_flow)),
                'bpr': loss_metric(tf.squeeze(tt), self.link_traveltimes(flow)),
                'total': 0}

        for key,val in lambdas_vals.items():
            if val != 0:
                loss['total'] += lambdas[key]*loss[key]

        return loss

    def call(self, X):
        """
        Output is matrix of dimension (n_days, n_links)
        """

        return self.link_traveltimes(self.predict_flow(X))

    def normalized_losses(self, losses) -> pd.DataFrame:
        losses_df = []
        for epoch, loss in enumerate(losses):
            losses_df.append({'epoch': epoch})
            for key, val in loss.items():
                losses_df[-1][key] = [val.numpy()]
                losses_df[-1][key] = losses_df[-1][key] / losses[0][key].numpy() * 100

        return pd.concat([pd.DataFrame(i) for i in losses_df], axis=0, ignore_index=True)

    def train(self,
              X_train: tf.constant,
              Y_train: tf.constant,
              X_val: tf.constant,
              Y_val: tf.constant,
              optimizer: tf.keras.optimizers,
              lambdas: Dict[str, float],
              epochs=1,
              batch_size=None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if batch_size is None:
            batch_size = X_train.shape[0]

        epoch = 0
        t0 = time.time()

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        # val_dataset = val_dataset.batch(batch_size)

        # Initial Losses
        train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=lambdas)['total']
        # val_loss = self.loss_function(X=X_val, Y=Y_val, lambdas=lambdas)['total']

        train_losses, val_losses = [], []

        while train_loss > 1e-8 and epoch <= epochs:

            if epoch % 1 == 0:
                # print(f"{i}: loss={loss.numpy():0.4g}, theta = {model.theta.numpy()}")
                train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=lambdas)
                val_loss = self.loss_function(X=X_val, Y=Y_val, lambdas=lambdas)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"{epoch}: train_loss={train_loss['total'].numpy():0.2g},  val_loss={val_loss['total'].numpy():0.2g}, "
                    f"train_loss tt={train_loss['tt'].numpy():0.2g}, val_loss tt={val_loss['tt'].numpy():0.2g}, "
                    f"train_loss flow={(train_loss['flow']).numpy():0.2g}, val_loss flow={val_loss['flow'].numpy():0.2g}, "
                    f"train_loss bpr={train_loss['bpr'].numpy():0.2g}, val_loss bpr={val_loss['bpr'].numpy():0.2g}, "
                    f"theta = {self.theta.numpy()}, avg alpha = {np.mean(self.alpha.numpy()):0.2g}, "
                    f"avg beta = {np.mean(self.beta.numpy()):0.2g}, "
                    f"avg abs diff demand ={np.mean(np.abs(self.q - self.network.q.flatten())):0.2g}, "
                    f"time: {time.time() - t0: 0.2g}")

                t0 = time.time()

            for step, (X_batch_train, Y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    train_loss = self.loss_function(X=X_batch_train, Y=Y_batch_train, lambdas=lambdas)['total']

                grads = tape.gradient(train_loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))

            epoch += 1

        train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=lambdas)['total']
        val_loss = self.loss_function(X=X_val, Y=Y_val, lambdas=lambdas)['total']
        print(f"{epoch} [FINAL]: train loss={train_loss.numpy():0.4g},val loss={val_loss.numpy():0.4g}")

        train_losses_df = self.normalized_losses(train_losses)
        val_losses_df = self.normalized_losses(val_losses)

        return train_losses_df, val_losses_df


def error(actual: tf.constant, predicted: tf.constant):
    return tf.boolean_mask(predicted - actual, tf.math.is_finite(predicted - actual))


def mse(actual: tf.constant, predicted: tf.constant):
    return tf.reduce_mean(tf.math.pow(error(actual, predicted), 2))


def btcg_mse(actual: tf.constant, predicted: tf.constant):
    ''' Normalization used by Wu et al. (2018), TRC. This metric has more numerical issues than using MSE'''

    rel_error = tf.math.divide_no_nan(predicted, actual)

    return 1 / 2 * tf.reduce_mean(tf.math.pow(tf.boolean_mask(rel_error, tf.math.is_finite(rel_error)) - 1, 2))
    # return 1 / 2 * tf.reduce_mean(tf.math.pow(error(actual, predicted) /
    #                                           (tf.boolean_mask(actual, tf.math.is_finite(actual)) + epsilon), 2))


def simulate_features(n_periods, **kwargs):
    linkdata_generator = isl.factory.LinkDataGenerator()

    df_list = []

    for i in range(1, n_periods + 1):
        df_day = linkdata_generator.simulate_features(**kwargs)
        df_day.insert(0, 'period', i)
        df_list.append(df_day)

    df = pd.concat(df_list)

    return df


def convert_multiperiod_df_to_tensor(df, n_days, n_links, features, n_hours=1):
    '''
    Convert to a tensor of dimensions (n_days, n_hours, n_links, n_features).
    df is a dataframe that contains the feature data
    '''

    return tf.constant(np.array(df[features]).reshape(n_days, n_hours, n_links, len(features)))


def simulate_features_tensor(**kwargs):
    return convert_multiperiod_df_to_tensor(df=simulate_features(**kwargs),
                                            n_links=len(kwargs['links']),
                                            features=kwargs['features_Z'],
                                            n_days=kwargs['n_days']
                                            )


def simulate_suelogit_data(periods: List,
                           features_data: pd.DataFrame,
                           network: TNetwork,
                           equilibrator: Equilibrator,
                           **kwargs):
    linkdata_generator = isl.factory.LinkDataGenerator()

    df_list = []

    for i, period in enumerate(periods):
        printIterationBar(i + 1, len(periods), prefix='periods:', length=20)

        # linkdata_generator.simulate_features(**kwargs)
        df_period = features_data[features_data.period == period]

        network.load_features_data(linkdata=df_period)

        with block_output(show_stdout=False, show_stderr=False):
            counts, _ = linkdata_generator.simulate_counts(
                network=network,
                equilibrator=equilibrator,
                noise_params={'mu_x': 0, 'sd_x': 0},
                coverage=1)

        network.load_traffic_counts(counts=counts)

        df_period['traveltime'] = [link.true_traveltime for link in network.links]

        df_period['counts'] = network.observed_counts_vector

        df_list.append(df_period)

    df = pd.concat(df_list)

    return df


def get_design_tensor(Z: pd.DataFrame = None,
                      y: pd.DataFrame = None,
                      **kwargs) -> tf.Tensor:
    """
    return tensor with dimensions (n_days, n_links, 1+n_features)
    """

    if Z is None:
        df = y
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)
    elif y is None:
        df = Z
    else:
        df = pd.concat([y, Z], axis=1)

    return convert_multiperiod_df_to_tensor(df=df, features=df.columns, **kwargs)


def get_y_tensor(y: pd.DataFrame, **kwargs):
    return convert_multiperiod_df_to_tensor(y, features=y.columns, **kwargs)


def build_tntp_network(network_name):
    '''
    Read data from tntp repository and build network object
    '''

    links_df = isl.reader.read_tntp_linkdata(network_name=network_name)
    links_df['link_key'] = [(i, j, '0') for i, j in zip(links_df['init_node'], links_df['term_node'])]

    network_generator = isl.factory.NetworkGenerator()
    A = network_generator.generate_adjacency_matrix(links_keys=list(links_df['link_key'].values))
    tntp_network = network_generator.build_network(A=A, network_name=network_name)

    # Link performance functions
    tntp_network.set_bpr_functions(bprdata=pd.DataFrame({'link_key': tntp_network.links_dict.keys(),
                                                         'alpha': links_df.b,
                                                         'beta': links_df.power,
                                                         'tf': links_df.free_flow_time,
                                                         'k': links_df.capacity
                                                         }))

    # Link features from TNTP repo
    # link_features_df = links_df[['link_key', 'length', 'speed', 'link_type', 'toll']]

    # OD matrix
    Q = isl.reader.read_tntp_od(network_name=network_name)
    tntp_network.load_OD(Q=Q)

    return tntp_network


def build_fresno_network():
    # Read nodes data
    nodes_df = pd.read_csv(isl.config.dirs['read_network_data'] + 'nodes/' + 'fresno-nodes-data.csv')

    # Read link specific attributes
    links_df = pd.read_csv(isl.config.dirs['read_network_data'] + 'links/' 'fresno-link-specific-data.csv',
                           converters={"link_key": ast.literal_eval, "pems_id": ast.literal_eval})

    links_df['free_flow_speed'] = links_df['length'] / links_df['tf_inrix']

    network_generator = isl.factory.NetworkGenerator()

    network = \
        network_generator.build_fresno_network(
            A=network_generator.generate_adjacency_matrix(links_keys=list(links_df['link_key'].values)),
            links_df=links_df, nodes_df=nodes_df, network_name='Fresno')

    bpr_parameters_df = pd.DataFrame({'link_key': links_df['link_key'],
                                      'alpha': links_df['alpha'],
                                      'beta': links_df['beta'],
                                      'tf': links_df['tf_inrix'],
                                      # 'tf': links_df['tf'],
                                      'k': pd.to_numeric(links_df['k'], errors='coerce', downcast='float')
                                      })

    # Normalize free flow travel time between 0 and 1
    bpr_parameters_df['tf'] = pd.DataFrame(
        preprocessing.MinMaxScaler().fit_transform(np.array(bpr_parameters_df['tf']).reshape(-1, 1)))

    network.set_bpr_functions(bprdata=bpr_parameters_df)

    network_generator.read_OD(network=network, sparse=True)

    return network


def load_k_shortest_paths(network, k, update_incidence_matrices=False, **kwargs):
    isl.factory.PathsGenerator().load_k_shortest_paths(network=network,
                                                       k=k,
                                                       update_incidence_matrices=False,
                                                       **kwargs)
    if update_incidence_matrices:
        paths_od = network.paths_od
        network.D = network.generate_D(paths_od=paths_od, links=network.links)
        network.M = network.generate_M(paths_od=paths_od)

        # TODO: remove dependency on C after translating operations to sparse representation
        network.C = network.generate_C(network.M)


def read_paths(network, **kwargs):
    return isl.factory.PathsGenerator().read_paths(network=network, **kwargs)


def plot_predictive_performance(train_losses_df: pd.DataFrame, val_losses_df: pd.DataFrame):
    fig = plt.figure()

    plt.plot(train_losses_df['epoch'], train_losses_df['tt'], label="Train loss (travel time)", color='red',
             linestyle='-')
    plt.plot(val_losses_df['epoch'], val_losses_df['tt'], label="Validation loss (travel time)", color='red',
             linestyle='--')
    plt.plot(train_losses_df['epoch'], train_losses_df['flow'], label="Train loss (flow)", color='blue', linestyle='-')
    plt.plot(val_losses_df['epoch'], val_losses_df['flow'], label="Validation loss (flow)", color='blue',
             linestyle='--')
    plt.plot(train_losses_df['epoch'], train_losses_df['bpr'], label="Train loss (bpr)", color='gray', linestyle='-')
    plt.plot(val_losses_df['epoch'], val_losses_df['bpr'], label="Validation loss (bpr)", color='gray',
             linestyle='--')

    plt.xticks(np.arange(train_losses_df['epoch'].min(), train_losses_df['epoch'].max() + 1, 5))

    plt.xlabel('epoch')
    plt.ylabel('decrease in mse (%)')
    plt.legend()

    fig.show()
