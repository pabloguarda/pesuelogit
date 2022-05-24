"""
Module for AutoEncoded Stochastic User Equilibrium with Logit Assignment (ODLUE)
"""

import isuelogit as isl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from typing import Dict, List, Tuple
import time
from .networks import Equilibrator, ColumnGenerator


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


class NGD(tf.keras.optimizers.SGD):

    """ NGD: Normalizaed Gradient Descent """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):

        grads_and_vars = [(g, v) for g, v in args[0] if g is not None]

        normalized_grads, vars = [], []
        for g,v in grads_and_vars:
            normalized_grads.append(g / tf.norm(g, 2))
            vars.append(v)

        return tf.keras.optimizers.SGD.apply_gradients(self,zip(normalized_grads,vars))


class UtilityFunction(isl.estimation.UtilityFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ODLUE(tf.keras.Model):

    # ODLUE is an extension of the logit utility estimation (LUE) problem solved in the isuelogit package.  It is
    # fed with link features of multiple time days and it allows for the estimation of the OD matrix and of the
    # utility function parameters. It is framed as a computational graph which facilitates the computation of gradient
    # of the loss function.

    # The current implementation assumes that the path sets and the OD matrix is the same for all time days.
    # However, if the path set were updated dynamically, the path sets would vary among time days and od pairs over
    # iterations. As consequence, the matrices M and C would change frequently, which may be computationally intractable.

    def __init__(self,
                 network: isl.networks.TNetwork,
                 utility_function: UtilityFunction,
                 key: str = None,
                 equilibrator: Equilibrator = None,
                 column_generator: ColumnGenerator = None,
                 trainables: Dict[str, bool] = None,
                 inits: Dict[str, bool] = None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.key = key

        self.n_features = None
        self.n_links = None
        self.n_days = None
        self.n_hours = None

        self.network = network
        self.utility_function = utility_function
        self.equilibrator = equilibrator
        self.column_generator = column_generator

        params = ['q', 'alpha', 'beta', 'theta', 'theta_links', 'psc_factor']

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
            if tf.rank(q_init) == 2 and  q_init.shape[0] == 1:
                q_init = q_init.flatten()

        theta_init = np.array(list(self.utility_function.true_values.values()))

        if self.trainables['theta'] or None in theta_init:
            theta_init = self.inits.get('theta', np.array(list(self.utility_function.initial_values.values())))
            if tf.rank(theta_init) == 2 and theta_init.shape[0] == 1:
                #TODO: Review
                theta_init = theta_init.flatten()

        self._q = tf.Variable(np.sqrt(q_init), trainable=self.trainables['q'], name="q", dtype=self.dtype)
        # self._q = self._q[tf.newaxis,:]

        self._theta = tf.Variable(theta_init, trainable=self.trainables['theta'], name="theta", dtype=self.dtype)

        # Initialize the psc_factor in a value different than zero to generate gradient
        self._psc_factor = tf.Variable(0, trainable=self.trainables['psc_factor'], name="psc_factor", dtype=self.dtype)

        # Link specific effect (act as an intercept)
        self._theta_links = tf.Variable(tf.zeros(len(self.network.links), tf.float64),
                                        trainable=self.trainables['theta_links'], name="theta_links", dtype=self.dtype)

        # self.linklist = [(link.key[0], link.key[1]) for link in self.network.links]
        self.n_nodes = self.network.get_n_nodes()
        self.triplist = self.network.ods
        # self.I = np.identity(self.n_nodes)

        # TODO: Generate D matrix with linklist and a sparse tensor
        # # Incidence matrices (assumes paths sets are the same over time days and iterations)
        # self._D = tf.constant(self.network.D, dtype=self.dtype)
        # self._M = tf.constant(self.network.M, dtype=self.dtype)
        # self._C = tf.constant(self.network.C, dtype=self.dtype)
        # # self.C = self.network.generate_C(self.M)

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

        if signs is not None:
            for feature in self.utility_function.features:

                sign = signs.get(feature)

                if sign == '+':
                    clips_min.append(0)
                    clips_max.append(tf.float64.max)
                if sign == '-':
                    clips_min.append(tf.float64.min)
                    clips_max.append(0)

            return tf.clip_by_value(theta, clips_min, clips_max)

        return theta


    @property
    def psc_factor(self):
        """ Path size correction factor """
        return self._psc_factor

    @property
    def theta(self):
        return self.project_theta(self._theta)

    @property
    def D(self):
        return tf.constant(self.network.D, dtype=self.dtype)

    @property
    def M(self):
        return tf.constant(self.network.M, dtype=self.dtype)

    @property
    def C(self):
        return tf.constant(self.network.C, dtype=self.dtype)

    def path_size_correction(self, Vf):

        return Vf + self.psc_factor*tf.math.log(tf.constant(
            isl.paths.compute_path_size_factors(D = self.network.D, paths_od = self.network.paths_od).flatten()))

        # return Vf + self.psc_factor * tf.constant(
        #     isl.paths.compute_path_size_factors(D=self.network.D, paths_od=self.network.paths_od).flatten())

    def normalizer(self, vf):

        '''
        This operation is computationally expensive
        # np.nanmax is not available in tensorflow, which introduces inneficiency
        # This function can be eliminated

        '''
        C_nan = np.where(self.C.numpy().astype('float') == 0, float('nan'), self.C.numpy())
        v_max = np.nanmax(np.einsum("ijk,jk -> ijk", vf, C_nan), axis=2)
        return vf - v_max
        # v_max = np.nanmax(vf * np.where(self.C.astype('float') == 0, float('nan'), self.C), axis=1)
        # return (vf - v_max)[:, np.newaxis]

    def link_utilities(self, X):

        """ TODO: Make the einsum operation in one line"""

        if tf.rank(self.theta) == 1:
            return tf.einsum("ijkl,l -> ijk", X, self.theta) + self._theta_links

        return tf.einsum("ijkl,jl -> ijk", X, self.theta) + self._theta_links

    def path_utilities(self, V):
        return self.path_size_correction(tf.einsum("ijk,kl -> ijl", V, self.D))

    # def path_attributes(self, X):
    #     return tf.einsum("ijk,jl -> ilk", X, self.D)
    #     # return tf.tensordot(tf.transpose(self.D),X,1)
    #
    # def path_utilities(self, Xf):
    #     # return tf.tensordot(Xf, self.theta)
    #     return tf.einsum("ijk,k -> ij", Xf, self.theta)Link specific effects are also related to a panel fixed effect

    def path_probabilities_sparse(self, vf, normalization=False):
        ''' Sparse version. Computation time is roughly the same than non-sparse version but it does not require
        to store the matrix C which has dimensions n_paths X n_paths
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

        # TODO: Readapt this code to account for the hurs dimension or simply remove this as it is not used by default
        if normalization:
            vf = self.normalizer(vf)

        return tf.einsum("ijk,ijk -> ijk", tf.exp(vf), 1 / (tf.einsum("ijk,kl -> ijl", tf.exp(vf), self.C)))

        # return tf.divide(tf.exp(vf), tf.tensordot(self.C, tf.exp(vf)) + epsilon,1)

    def path_flows(self, pf):
        # TODO: Test and try to combine the einsums if possible to avoid ifelse clause

        if tf.rank(self.q) == 1:
            return tf.einsum("ij,i, klj -> klj", self.M, self.q, pf)
            # return tf.einsum("j,lij -> lij", tf.einsum("ij,i-> j", self.M, self.q), pf)

        # todo: have a single einsum, e.g. return tf.einsum("ij,ki, lkj -> lij", self.M, self.q, pf)
        return tf.einsum("ij, lij -> lij", tf.einsum("ij,ki-> kj", self.M, self.q), pf)

    def link_flows(self, f):
        return tf.einsum("ijk,lk -> ijl", f, self.D)

    def call(self, X):
        """
        X is tensor of dimension (n_days, n_hours, n_links, n_features)
        """

        self.n_days, self.n_hours, self.n_links, self.n_features = X.shape

        self.n_features-=1

        X = tf.cast(X, self.dtype)

        return self.link_flows(self.path_flows(self.path_probabilities(self.path_utilities(self.link_utilities(X)))))


class AESUELOGIT(ODLUE):

    #TODO: Move some functions to ODLUE

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
        # TODO: Check that this transformation still allows for perfect parameter recovery in experiments
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

        self.tt_ff, feature_data = X[:, :, :, 0], X[:, :, :, 1:]

        # Add RELU layer to avoid negative flows into input of BPR function with exponent
        return tf.keras.activations.relu(super(AESUELOGIT, self).call(feature_data))

    def predict_equilibrium_flow(self,
                                 X: tf.constant,
                                 q: tf.constant,
                                 utility_parameters: Dict[str,float],
                                 **kwargs) -> tf.constant:

        # Update values of utility function
        self.utility_function.values = utility_parameters

        # Update BPR functions in the network object based on current estimate of BPR parameters
        link_keys = []
        alphas = self.alpha.numpy()
        betas = self.beta.numpy()
        for link, alpha, beta in zip(self.network.links, alphas, betas):
            link_keys.append(link.key)
            link.bpr.alpha = alpha
            link.bpr.beta = beta

        # TODO: load self.tt_ff

        # Load features in network (Convert columns of X in lists, and provide features_Z
        features_Z = self.utility_function.features_Z

        linkdata = pd.DataFrame({i: j for i, j in zip(features_Z, tf.unstack(X, axis=1))})
        linkdata['link_key'] = [link.key for link in self.network.links]
        self.network.load_features_data(linkdata)

        results_eq = self.equilibrator.path_based_suelogit_equilibrium(
            theta=self.utility_function.values,
            q=np.expand_dims(q, 1),
            features_Z=self.utility_function.features_Z,
            column_generation={'n_paths': None, 'paths_selection': None},
            **kwargs)

        print('\n')

        return tf.constant(list(results_eq['x'].values()))

    def generalization_error(self, Y: tf.constant, X: tf.constant, **kwargs):

        q = self.q
        theta = self.theta

        random_day = np.random.randint(0, X.shape[0])
        hours = 1
        if tf.rank(q)>1:
            hours = self.q.shape[0]

        sum_mse = 0
        sum_n = 0

        for hour in range(hours):

            Xt = X[random_day, hour, :, :]
            # Update values of utility function
            # self.utility_function.values = {k: v for k, v in zip(self.utility_function.features, list(self.theta))}
            if tf.rank(theta)> 1:
                theta = self.theta[hour]
            if tf.rank(q)> 1:
                q = self.q[hour]

            utility_parameters = {k: v for k, v in zip(self.utility_function.features, list(theta))}

            predicted_flow = self.predict_equilibrium_flow(Xt, utility_parameters = utility_parameters, q = q, **kwargs)

            # It is assumed the equilibrium flows only varies according to the hour of the day
            observed_flow = Y[:, hour, :, 1]

            n = tf.cast(tf.math.count_nonzero(~tf.math.is_nan(observed_flow)),tf.float64)

            sum_mse += mse(actual=observed_flow, predicted=predicted_flow)*n
            sum_n += n

        return sum_mse/sum_n

    def historic_od(self, pred_q):

        """
        :param hours:
        :return: a historic OD matrix of size (ods,hours) with values set to nan except for the first hour index (:,0).
        It assumes that the historic_od is only meaningful for the first hour
        """

        historic_od = tf.expand_dims(tf.constant(self.network.q.flatten()),axis =0)
        if tf.rank(pred_q) > 1:
            extra_od_cols = tf.cast(tf.constant(float('nan'), shape = (pred_q.shape[0]-1,tf.size(historic_od))), tf.float64)
            historic_od = tf.concat([historic_od, extra_od_cols], axis =0)

        # return tf.expand_dims(tf.constant(self.network.q.flatten()),axis =0)

        return historic_od

    def loss_function(self,
                      X,
                      Y,
                      lambdas: Dict[str, float],
                      # loss_metric = btcg_mse,
                      loss_metric = mse,
                      ):
        """
        Return a dictionary with keys defined as the different terms of the loss function

        # loss_metric = btcg_mse, mse

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

        loss = {'od': loss_metric(actual=self.historic_od(pred_q=self.q), predicted=self.q),
                'theta': tf.reduce_mean(tf.norm(self.theta, 1)),
                'flow': loss_metric(actual=tf.squeeze(flow), predicted=pred_flow),
                'tt': loss_metric(actual=tf.squeeze(tt), predicted=self.link_traveltimes(pred_flow)),
                'bpr': loss_metric(actual=tf.squeeze(tt), predicted=self.link_traveltimes(flow)),
                'total': 0}

        for key, val in lambdas_vals.items():
            if val != 0:
                loss['total'] += lambdas[key] * loss[key]

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
                normalizer = losses[0][key].numpy()
                if normalizer == 0:
                    losses_df[-1][key] = 100
                else:
                    losses_df[-1][key] = losses_df[-1][key]/normalizer * 100

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

        """ It assumes the first column of tensors X_train and X_val are the free flow travel times. The following
        columns are the travel times and exogenous features of each link """

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

            t0 = time.time()

            if epoch % 1 == 0:
                print(f"\nEpoch: {epoch}")
                # print(f"{i}: loss={loss.numpy():0.4g}, theta = {model.theta.numpy()}")
                train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=lambdas)
                val_loss = self.loss_function(X=X_val, Y=Y_val, lambdas=lambdas)

                # train_loss['generalization_error'] = self.generalization_error(X=X_train[:, :, :, 1:],Y=Y_train)
                # val_loss['generalization_error'] = self.generalization_error(X=X_val[:, :, :, 1:],Y=Y_val)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"{epoch}: train_loss={train_loss['total'].numpy():0.2g},  val_loss={val_loss['total'].numpy():0.2g}, "
                    f"train_loss tt={train_loss['tt'].numpy():0.2g}, val_loss tt={val_loss['tt'].numpy():0.2g}, "
                    f"train_loss flow={(train_loss['flow']).numpy():0.2g}, val_loss flow={val_loss['flow'].numpy():0.2g}, "
                    f"train_loss bpr={train_loss['bpr'].numpy():0.2g}, val_loss bpr={val_loss['bpr'].numpy():0.2g}, "
                    
                    # f"val generalization error ={val_loss['generalization_error'].numpy():0.2g}, "
                    
                    f"avg abs diff demand ={np.nanmean(np.abs(self.q - self.historic_od(self.q))):0.2g}, "
                    f"theta = {self.theta.numpy()}, psc_factor = {self.psc_factor.numpy()}, "
                    f"avg alpha = {np.mean(self.alpha.numpy()):0.2g}, avg beta = {np.mean(self.beta.numpy()):0.2g}, "
                    f"time: {time.time() - t0: 0.2g}")

                t0 = time.time()

            # Gradient based learning

            for step, (X_batch_train, Y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    train_loss = self.loss_function(X=X_batch_train, Y=Y_batch_train, lambdas=lambdas)['total']

                grads = tape.gradient(train_loss, self.trainable_variables)

                # # Apply some clipping (tf.linalg.normad
                # grads = [tf.clip_by_norm(g, 2) for g in grads]

                # # The normalization of gradient of NGD can be hardcoded as
                # if isinstance(optimizer, NGD):
                #     grads = [g/tf.linalg.norm(g, 2) for g in grads]

                optimizer.apply_gradients(zip(grads, self.trainable_variables))

            epoch += 1

            # TODO: Column generation (limit the options to more fundamental ones)
            self.column_generator.generate_paths(theta=self.theta,
                                                 network=self.network
                                                 )

            # TODO: Path set selection (confirm if necessary)

        train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=lambdas)['total']
        val_loss = self.loss_function(X=X_val, Y=Y_val, lambdas=lambdas)['total']
        print(f"{epoch} [FINAL]: train loss={train_loss.numpy():0.4g},val loss={val_loss.numpy():0.4g}, "
              f"'training time={time.time()-t0:0.1f}")

        train_losses_df = self.normalized_losses(train_losses)
        val_losses_df = self.normalized_losses(val_losses)

        # train_losses_df['generalization_error'] = train_generalization_errors
        # val_losses_df['generalization_error'] = val_generalization_errors

        return train_losses_df, val_losses_df




