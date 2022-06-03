"""
Module for AutoEncoded Stochastic User Equilibrium with Logit Assignment (ODLUE)
"""

import isuelogit as isl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from typing import Dict, List, Tuple, Union
import time
from .networks import Equilibrator, ColumnGenerator, TransportationNetwork
from isuelogit.estimation import Parameter, compute_vot
from .descriptive_statistics import error, mse, rmse, nrmse, btcg_mse


class NGD(tf.keras.optimizers.SGD):
    """ NGD: Normalizaed Gradient Descent """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        grads_and_vars = [(g, v) for g, v in args[0] if g is not None]

        normalized_grads, vars = [], []
        for g, v in grads_and_vars:
            normalized_grads.append(g / tf.norm(g, 2))
            vars.append(v)

        return tf.keras.optimizers.SGD.apply_gradients(self, zip(normalized_grads, vars))


class Parameters(isl.estimation.UtilityFunction):
    """ Extension of isl.Parameters class. It supports parameters on multiple periods"""

    def __init__(self,
                 keys=None,
                 shapes=None,
                 trainables=None,
                 periods: int = 1,
                 link_specifics=None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        if keys is None:
            keys = {}

        self.periods = periods

        for key in keys:
            self.parameters[key] = Parameter(key=key,
                                             type=None,
                                             sign=kwargs.get('signs', {}).get(key, None),
                                             fixed=kwargs.get('fixed', {}).get(key, False),
                                             initial_value=kwargs.get('initial_values', {}).get(key, 0),
                                             true_value=kwargs.get('true_values', {}).get(key))

        for parameter in self.parameters.values():
            parameter.trainable = True

        self.trainables = trainables

        for parameter in self.parameters.values():
            parameter.link_specific = False

        self.link_specifics = link_specifics

        for parameter in self.parameters.values():
            parameter.shape = (1,)

        self.shapes = shapes

    def keys(self):
        return list(self.parameters.keys())

    @property
    def link_specifics(self):
        return {key: parameter.link_specific for key, parameter in self.parameters.items()}

    @link_specifics.setter
    def link_specifics(self, values: Dict[str, str]):

        if values is not None:
            for feature, value in values.items():
                value = values[feature]
                assert isinstance(value, bool)
                self.parameters[feature].link_specific = values[feature]

    @property
    def trainables(self):
        return {key: parameter.trainable for key, parameter in self.parameters.items()}

    @trainables.setter
    def trainables(self, values: Dict[str, str]):

        if values is not None:
            for feature, value in values.items():
                value = values[feature]
                assert isinstance(value, bool)
                self.parameters[feature].trainable = values[feature]

    @property
    def shapes(self):
        return {key: parameter.shape for key, parameter in self.parameters.items()}

    @shapes.setter
    def shapes(self, values: Dict[str, str]):
        if values is None:
            for feature, value in self.initial_values.items():
                if isinstance(value, float) or isinstance(value, int):
                    value = np.array([value])
                if isinstance(value, np.ndarray):
                    self.parameters[feature].shape = value.shape

        if values is not None:
            for feature, value in values.items():
                self.parameters[feature].shape = values[feature]

    def true_values_array(self, features=None) -> np.array:

        values_list = list(self.true_values.values())

        if features is not None:
            values_list = [self.true_values[feature] for feature in features]

        if self.periods == 1:
            return np.array(list(values_list))

        return np.repeat(np.array(values_list)[np.newaxis, :], self.periods, axis=0)

    def initial_values_array(self, features=None) -> np.array:

        values_list = list(self.initial_values.values())

        if features is not None:
            values_list = [self.initial_values[feature] for feature in features]

        if self.periods == 1:
            return np.array(list(values_list))

        return np.repeat(np.array(values_list)[np.newaxis, :], self.periods, axis=0)

    def constant_initializer(self, value):
        self.initial_values = dict.fromkeys(self.keys(), value)

    def random_initializer(self, range_values: Union[Tuple, Dict[str, Tuple]], keys: List = None):
        """ Randomly initialize values of the utility parameters based on true values and range """

        assert len(range_values) == 2, 'range must have length'

        if keys is None:
            keys = self.keys()

        random_utility_values = []

        for feature in keys:
            initial_value = float(self.true_values[feature])

            if isinstance(range_values, tuple):
                range_vals = range_values
            else:
                range_vals = range_values[feature]

            random_utility_values.append(initial_value + np.random.uniform(*range_vals))

        self.initial_values = dict(zip(keys, random_utility_values))

        # print('New initial values', self.utility_function.initial_values)

        return self.initial_values


class UtilityParameters(Parameters):
    """ Support utility function associated to multiple periods"""

    def __init__(self,
                 *args,
                 **kwargs):
        # TODO: change label features_Y by endogenous_feature and features_Z by exogenous_features.

        # kwargs['features_Y'] = None

        super().__init__(keys=['psc_factor', 'fixed_effect'], *args, **kwargs)


class BPRParameters(Parameters):

    def __init__(self, keys, *args, **kwargs):
        kwargs['features_Z'] = None
        kwargs['features_Y'] = None

        super().__init__(keys=keys, *args, **kwargs)


class ODParameters(Parameters):
    """ Support OD with multiple periods """

    def __init__(self,
                 trainable,
                 initial_values: np.array,
                 historic_values: Dict[str, np.array] = None,
                 true_values: np.array = None,
                 shape=None,
                 key='od',
                 *args,
                 **kwargs):
        kwargs['features_Z'] = None
        kwargs['features_Y'] = None

        kwargs['initial_values'] = {key: initial_values}
        kwargs['true_values'] = {key: true_values}

        super().__init__(keys=[key], *args, **kwargs)

        # self.shape = shape
        self.trainable = trainable
        self.historic_values = historic_values

    @property
    def historic_values_array(self):
        # historic_od = tf.expand_dims(tf.constant(self.network.q.flatten()), axis=0)
        # if len(list(self.historic_values.keys())) > 1:

        historic_od = np.empty((self.periods, self.shape[0]))
        historic_od[:] = np.nan

        for period, od in self.historic_values.items():
            historic_od[period - 1, :] = od

        return historic_od
        # return self._historic_values

    @property
    def key(self):
        return self.keys()[0]

    # @property
    # def trainable(self):
    #     return self.trainables[self.key]

    @property
    def shape(self):
        return self.shapes[self.key]

    @property
    def initial_value(self):
        return self._initial_values[self.key]

    @property
    def true_value(self):
        return self._true_values[self.key]

    def true_values_array(self) -> np.array:

        if self.periods == 1:
            return self.true_value

        return np.repeat(self.true_value[np.newaxis, :], self.periods, axis=0)

    def initial_values_array(self) -> np.array:

        if self.periods == 1:
            return self.initial_value

        return np.repeat(self.initial_value[np.newaxis, :], self.periods, axis=0)


class AESUELOGIT(tf.keras.Model):
    """ Auto-encoded stochastic user equilibrium with logit assignment"""

    def __init__(self,
                 network: TransportationNetwork,
                 utility: UtilityParameters,
                 endogenous_flows = False,
                 key: str = None,
                 od: ODParameters = None,
                 bpr: Parameters = None,
                 equilibrator: Equilibrator = None,
                 column_generator: ColumnGenerator = None,
                 *args,
                 **kwargs):

        self.observed_traveltimes = None
        self._flows = None
        kwargs['dtype'] = kwargs.get('dtype', tf.float64)

        super().__init__(*args, **kwargs)

        # Parameters (i.e. tf variables)
        self._alpha = tf.float64
        self._beta = tf.float64
        self.bpr = bpr
        self._psc_factor = None
        self._fixed_effect = None
        self._theta = None
        self._q = None

        self.endogenous_flows = endogenous_flows

        self.key = key

        self.network = network
        self.equilibrator = equilibrator
        self.column_generator = column_generator
        
        self.n_features = None
        self.n_links = len(self.network.links)
        self.n_days = None
        self.n_hours = 1

        self.utility = utility
        self.od = od

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

        # Free flow travel time
        self._tt_ff = np.array([link.bpr.tf for link in self.network.links])

        # Tolerance parameter
        self._epsilon = 1e-12

    def create_tensor_variables(self, keys: Dict[str, bool] = None):

        if keys is None:
            keys = dict.fromkeys(['q', 'theta', 'psc_factor', 'fixed_effect', 'alpha', 'beta'], True)

        # Link specific effect (act as an intercept)
        if self.endogenous_flows:
            self._flows = tf.Variable(
                initial_value=tf.math.sqrt(tf.constant(tf.zeros([self.n_hours,self.n_links], dtype=tf.float64))),
                # initial_value=tf.constant(tf.zeros([self.n_hours,self.n_links]), dtype=tf.float64),
                trainable= self.endogenous_flows,
                name='flows',
                dtype=self.dtype)

        if keys.get('alpha', False):
            self._alpha = tf.Variable(np.log(self.bpr.parameters['alpha'].initial_value),
                                      trainable=self.bpr.parameters['alpha'].trainable,
                                      name=self.bpr.parameters['alpha'].key,
                                      dtype=self.dtype)
        if keys.get('beta', False):
            self._beta = tf.Variable(np.log(self.bpr.parameters['beta'].initial_value),
                                     trainable=self.bpr.parameters['beta'].trainable,
                                     name=self.bpr.parameters['beta'].key,
                                     dtype=self.dtype)

        if keys.get('q', False):
            self._q = tf.Variable(initial_value=np.sqrt(self.od.initial_values_array()),
                                  trainable=self.od.trainable,
                                  name=self.od.key,
                                  dtype=self.dtype)
        #
        # TODO: Meantime, the feature parameters of the utility function can be or be not trained altogether based on
        #  on the value of 'tt'. I should allows for separate estimation

        if keys.get('theta', False):
            self._theta = tf.Variable(initial_value=self.utility.initial_values_array(self.utility.features),
                                      trainable=self.utility.trainables['tt'],
                                      name="theta",
                                      dtype=self.dtype)
        if keys.get('psc_factor', False):
            # Initialize the psc_factor in a value different than zero to generate gradient
            self._psc_factor = tf.Variable(initial_value=self.utility.initial_values['psc_factor'],
                                           trainable=self.utility.trainables['psc_factor'],
                                           name=self.utility.parameters['psc_factor'].key,
                                           dtype=self.dtype)

        if keys.get('fixed_effect', False):
            # Link specific effect (act as an intercept)
            self._fixed_effect = tf.Variable(
                initial_value=tf.constant(self.utility.initial_values['fixed_effect'],
                                          shape=tf.TensorShape(self.utility.shapes['fixed_effect']), dtype=tf.float64),
                trainable=self.utility.trainables['fixed_effect'],
                name=self.utility.parameters['fixed_effect'].key,
                dtype=self.dtype)

    def flows(self):
        return tf.math.pow(self._flows, 2)

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
    def fixed_effect(self):
        return self._fixed_effect

    def project_theta(self, theta):
        clips_min = []
        clips_max = []

        signs = self.utility.signs

        if signs is not None:
            for feature in self.utility.features:

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

        return Vf + self.psc_factor * tf.math.log(tf.constant(
            isl.paths.compute_path_size_factors(D=self.network.D, paths_od=self.network.paths_od).flatten()))

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
            return tf.einsum("ijkl,l -> ijk", X, self.theta[1:])+ self.theta[0]*self.traveltimes() + self.fixed_effect

        return tf.einsum("ijkl,jl -> ijk", X, self.theta[:,1:]) + self.fixed_effect + self.theta[:,0]*self.traveltimes()

    def link_traveltimes(self, x):
        """
        Link performance function
        """
        # Links capacities
        k = np.array([link.bpr.k for link in self.network.links])

        return self.tt_ff * (1 + self.alpha * tf.math.pow(x / k, self.beta))

    def traveltimes(self):
        """ Return tensor variable associated to endogenous travel times (assumed dependent on link flows)"""

        return self.link_traveltimes(x=self.flows())

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

    @property
    def alpha(self):
        return tf.exp(self._alpha)

    @property
    def beta(self):
        # return tf.clip_by_value(self._beta, 0 + self._epsilon, 5)
        return tf.exp(self._beta)

    @property
    def tt_ff(self):
        return self._tt_ff

    @tt_ff.setter
    def tt_ff(self, value):
        self._tt_ff = value

    def equilibrium_link_flows(self,
                               X: tf.constant,
                               q: tf.constant,
                               utility: Dict[str, float],
                               **kwargs) -> tf.constant:

        # Update values of utility function
        self.utility.values = utility

        # Update BPR functions in the network object based on current estimate of BPR parameters
        link_keys = []
        alphas = self.alpha.numpy()
        betas = self.beta.numpy()
        tt_ffs = np.array(self.tt_ff).flatten()
        links = self.network.links

        if alphas.size == 1:
            alphas = [alphas] * len(links)

        if betas.size == 1:
            betas = [betas] * len(links)

        for link, tt_ff, alpha, beta in zip(self.network.links, tt_ffs, alphas, betas):
            link_keys.append(link.key)
            link.bpr.alpha = alpha
            link.bpr.beta = beta
            link.bpr.tf = tt_ff

        # Load features in network (Convert columns of X in lists, and provide features_Z
        features_Z = self.utility.features_Z

        linkdata = pd.DataFrame({i: j for i, j in zip(features_Z, tf.unstack(X, axis=1))})
        linkdata['link_key'] = link_keys
        self.network.load_features_data(linkdata)

        results_eq = self.equilibrator.path_based_suelogit_equilibrium(
            theta=self.utility.values,
            q=np.expand_dims(q, 1),
            features_Z=self.utility.features_Z,
            column_generation={'n_paths': None, 'paths_selection': None},
            **kwargs)

        print('\n')

        return tf.constant(list(results_eq['x'].values()))

    def generalization_error(self, Y: tf.constant, X: tf.constant, loss_metric=nrmse, **kwargs):

        """

        :param Y: tensor with endogenous observed measurements [link flows, travel times]
        :param X: tensor with input system level data
        :param loss_metric:
        :param kwargs:
        :return:
        """

        q = self.q
        theta = self.theta

        random_day = np.random.randint(0, X.shape[0])
        hours = 1
        if tf.rank(q) > 1:
            hours = self.q.shape[0]

        sum_mse = 0
        sum_n = 0

        for hour in range(hours):

            Xt = X[random_day, hour, :, :]
            # Update values of utility function
            # self.utility.values = {k: v for k, v in zip(self.utility.features, list(self.theta))}
            if tf.rank(theta) > 1:
                theta = self.theta[hour]
            if tf.rank(q) > 1:
                q = self.q[hour]

            utility_parameters = {k: v for k, v in zip(self.utility.features, list(theta))}

            predicted_flow = self.equilibrium_link_flows(Xt, utility=utility_parameters, q=q, **kwargs)

            # It is assumed the equilibrium flows only varies according to the hour of the day
            observed_flow = Y[:, hour, :, 1]

            n = tf.cast(tf.math.count_nonzero(~tf.math.is_nan(observed_flow)), tf.float64)

            sum_mse += loss_metric(actual=observed_flow, predicted=predicted_flow) * n
            sum_n += n

        return sum_mse / sum_n

    def historic_od(self, pred_q):

        """
        :param hours:
        :return: a historic OD matrix of size (ods,hours) with values set to nan except for the first hour index (:,0).
        It assumes that the historic_od is only meaningful for the first hour
        """

        historic_od = tf.expand_dims(tf.constant(self.network.q.flatten()), axis=0)
        if tf.rank(pred_q) > 1:
            extra_od_cols = tf.cast(tf.constant(float('nan'), shape=(pred_q.shape[0] - 1, tf.size(historic_od))),
                                    tf.float64)
            historic_od = tf.concat([historic_od, extra_od_cols], axis=0)

        # return tf.expand_dims(tf.constant(self.network.q.flatten()),axis =0)

        return historic_od

    def loss_function(self,
                      X,
                      Y,
                      lambdas: Dict[str, float],
                      # loss_metric = btcg_mse,
                      loss_metric=mse,
                      ):
        """
        Return a dictionary with keys defined as the different terms of the loss function

        # loss_metric = btcg_mse, mse

        """

        lambdas_vals = {'tt': 1.0, 'od': 0.0, 'theta': 0.0, 'flow': 0.0, 'bpr': 0.0, 'eq_flow': 0.0, 'eq_tt': 0.0}

        assert set(lambdas.keys()).issubset(lambdas_vals.keys()), 'Invalid key in loss_weights attribute'

        for attr, val in lambdas.items():
            lambdas_vals[attr] = val

        loss = dict.fromkeys(list(lambdas_vals.keys()) + ['total'], tf.constant(0, dtype=tf.float64))

        if Y.shape[-1] > 0:
            self.observed_traveltimes, self.observed_flows = tf.unstack(Y,axis = -1)
            predicted_flow = self.compute_link_flows(X)
            predicted_traveltimes = self.link_traveltimes(predicted_flow)

            loss = {'od': loss_metric(actual=self.historic_od(pred_q=self.q), predicted=self.q),
                    'flow': loss_metric(actual=tf.squeeze(self.observed_flows), predicted=predicted_flow),
                    'tt': loss_metric(actual=self.observed_traveltimes, predicted=predicted_traveltimes),
                    'theta': tf.reduce_mean(tf.norm(self.theta, 1)),
                    'bpr': loss_metric(actual=tf.squeeze(self.observed_traveltimes), predicted=predicted_traveltimes),
                    'total': tf.constant(0, tf.float64)}

        # tf.squeeze(self.observed_flows)[0]-predicted_flow[0]

        #TODO: allows for computation even when they are not endogenous (create method flow())
        if self.endogenous_flows:
            loss['eq_flow'] = loss_metric(actual=self.flows(), predicted=predicted_flow)

        # if self.endogenous_traveltimes:
        loss['eq_tt'] = loss_metric(actual=self.traveltimes(), predicted=tf.squeeze(predicted_traveltimes))

        # self.traveltimes()
        # tf.squeeze(predicted_traveltimes)[0]

        lambdas = {k: tf.constant(v, name="lambda_" + k, dtype=tf.float64) for k, v in lambdas_vals.items()}

        # lambdas = {'od': tf.constant(lambdas_vals['od'], name="lambda_od", dtype=tf.float64),
        #            'theta': tf.constant(lambdas_vals['theta'], name="lambda_theta", dtype=tf.float64),
        #            'flow': tf.constant(lambdas_vals['flow'], name="lambda_flow", dtype=tf.float64),
        #            'tt': tf.constant(lambdas_vals['tt'], name="lambda_tt", dtype=tf.float64),
        #            'bpr': tf.constant(lambdas_vals['bpr'], name="lambda_bpr", dtype=tf.float64)
        #            }

        for key, val in lambdas_vals.items():
            # if any(list(map(lambda x: isinstance(val, x), [float, int]))):
            if val > 0:
                loss['total'] += lambdas[key] * loss[key]

        # Add prefix "loss_"
        loss = {'loss_' + k: v for k, v in loss.items()}

        return loss

    def normalized_losses(self, losses) -> pd.DataFrame:

        # if losses[0]['total'] == 0:
        #     return losses

        losses_df = []
        for epoch, loss in enumerate(losses):
            losses_df.append({'epoch': epoch})
            for key, val in loss.items():
                losses_df[-1][key] = [val.numpy()]
                normalizer = losses[0][key].numpy()
                if normalizer == 0:
                    losses_df[-1][key] = [100]
                else:
                    losses_df[-1][key] = losses_df[-1][key] / normalizer * 100

        return pd.concat([pd.DataFrame(i) for i in losses_df], axis=0, ignore_index=True)

    def get_parameters_estimates(self) -> pd.DataFrame:

        # TODO: extend for multiperiod theta and multilinks alpha, beta
        estimates = {}
        estimates.update(dict(zip(self.utility.features, self.theta.numpy().flatten())))
        estimates.update(dict(zip(['alpha', 'beta'], [float(self.alpha.numpy()), float(self.beta.numpy())])))
        estimates['psc_factor'] = float(self.psc_factor.numpy())

        return pd.DataFrame(estimates, index=[0])

    def get_true_parameters(self) -> pd.DataFrame:

        true_values = {k: v for k, v in {**self.bpr.true_values, **self.utility.true_values}.items()}

        if set(['c', 'tt']).issubset(true_values.keys()):
            true_values['vot'] = compute_vot(true_values)

        return pd.DataFrame({'parameter': true_values.keys(), 'truth': true_values.values()})

    # def best_results(self, results: pd.DataFrame):
    #     return results[results['loss_total'].argmin()]

    def split_results(self, results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        col_losses = ['epoch'] + [col for col in results.columns if any(x in col for x in ['loss_', 'error'])]

        results_losses = results[col_losses]

        results_parameters = results[['epoch'] + [col for col in results.columns if col not in col_losses]]

        return results_parameters, results_losses

    def train(self,
              X_train: tf.constant,
              Y_train: tf.constant,
              X_val: tf.constant,
              Y_val: tf.constant,
              optimizer: tf.keras.optimizers,
              loss_weights: Dict[str, float],
              generalization_error: Dict[str, bool] = None,
              epochs=1,
              batch_size=None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        """ It assumes the first column of tensors X_train and X_val are the free flow travel times. The following
        columns are the travel times and exogenous features of each link """

        if batch_size is None:
            batch_size = X_train.shape[0]

        if generalization_error is None:
            generalization_error = {'train': False, 'validation': False}

        X_train, Y_train, X_val, Y_val = map(lambda x: tf.cast(x,tf.float64),[X_train, Y_train, X_val, Y_val])

        self.n_days, self.n_hours, self.n_links, self.n_features = X_train.shape

        self.create_tensor_variables()

        if self.endogenous_flows:
            # Smart initialization is performed running a single pass of traffic assignment under initial theta and q
            self._flows.assign(tf.math.sqrt(tf.reduce_mean(self.call(X_train),axis = 0)))
            # self._flows.assign(tf.squeeze(tf.reduce_mean(self.call(tf.unstack(Y_train,axis = -1)[1]),axis=-1)))

        epoch = 0
        t0 = time.time()
        total_t0 = time.time()

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        # val_dataset = val_dataset.batch(batch_size)

        # Initial Losses
        train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=loss_weights)['loss_total']
        # val_loss = self.loss_function(X=X_val, Y=Y_val, loss_weights=loss_weights)['total']

        train_losses, val_losses = [], []

        estimates = []

        while epoch <= epochs:

            estimates.append(self.get_parameters_estimates())

            if epoch % 1 == 0:
                print(f"\nEpoch: {epoch}, n_train: {X_train.shape[0]}, n_test: {X_val.shape[0]}")
                # print(f"{i}: loss={loss.numpy():0.4g}, theta = {model.theta.numpy()}")
                train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=loss_weights)
                val_loss = self.loss_function(X=X_val, Y=Y_val, lambdas=loss_weights)

                if generalization_error.get('train', False):
                    train_loss['generalization_error'] = self.generalization_error(X=X_train, Y=Y_train)
                if generalization_error.get('validation', False):
                    val_loss['generalization_error'] = self.generalization_error(X=X_val, Y=Y_val)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"\n{epoch}: train_loss={float(train_loss['loss_total'].numpy()):0.1g}, "
                    f"val_loss={float(val_loss['loss_total'].numpy()):0.2g}, "
                    f"train_loss tt={float(train_loss['loss_tt'].numpy()):0.2g}, "
                    f"val_loss tt={float(val_loss['loss_tt'].numpy()):0.2g}, "
                    f"train_loss flow={float(train_loss['loss_flow'].numpy()):0.2g}, "
                    f"val_loss flow={float(val_loss['loss_flow'].numpy()):0.2g}, "
                    f"train_loss bpr={float(train_loss['loss_bpr'].numpy()):0.2g}, "
                    f"val_loss bpr={float(val_loss['loss_bpr'].numpy()):0.2g}, "
                    f"theta = {self.theta.numpy()}, "
                    f"vot = {np.array(compute_vot(self.get_parameters_estimates().to_dict(orient='records')[0])):0.2f}, "
                    f"psc_factor = {self.psc_factor.numpy()}, "
                    f"avg abs theta fixed effect = {np.mean(np.abs(self.fixed_effect)):0.2g}, "
                    f"avg alpha = {np.mean(self.alpha.numpy()):0.2g}, avg beta = {np.mean(self.beta.numpy()):0.2g}, "
                    f"avg abs diff demand ={np.nanmean(np.abs(self.q - self.historic_od(self.q))):0.2g}, ",end = '')

                if train_loss.get('loss_eq_tt', False):
                    print(f"train tt equilibrium loss ={float(train_loss['loss_eq_tt'].numpy()):0.2g}, ", end = '')

                if train_loss.get('loss_eq_flow', False):
                    print(f"train flow equilibrium loss ={float(train_loss['loss_eq_flow'].numpy()):0.2g}, ", end = '')

                if generalization_error.get('train', False):
                    print(f"train generalization error ={train_loss['generalization_error'].numpy():0.2g}, ", end = '')
                if generalization_error.get('validation', False):
                    print(f"val generalization error ={val_loss['generalization_error'].numpy():0.2g}, ", end = '')

                print(f"time: {time.time() - t0: 0.1f}")


                t0 = time.time()

            if train_loss['loss_total'] > 1e-8 and epoch <= epochs:

                # Gradient based learning

                for step, (X_batch_train, Y_batch_train) in enumerate(train_dataset):
                    with tf.GradientTape() as tape:
                        train_loss = \
                            self.loss_function(X=X_batch_train, Y=Y_batch_train, lambdas=loss_weights)['loss_total']

                    grads = tape.gradient(train_loss, self.trainable_variables)

                    # # Apply some clipping (tf.linalg.normada
                    # grads = [tf.clip_by_norm(g, 2) for g in grads]

                    # # The normalization of gradient of NGD can be hardcoded as
                    # if isinstance(optimizer, NGD):
                    #     grads = [g/tf.linalg.norm(g, 2) for g in grads]

                    optimizer.apply_gradients(zip(grads, self.trainable_variables))

            epoch += 1

            if self.column_generator is not None:
                # TODO: Column generation (limit the options to more fundamental ones)
                self.column_generator.generate_paths(theta=self.theta,
                                                     network=self.network
                                                     )

            # TODO: Path set selection (confirm if necessary)

        train_losses_df = self.normalized_losses(train_losses)
        val_losses_df = self.normalized_losses(val_losses)

        train_results_df = pd.concat([train_losses_df, pd.concat(estimates, axis=0).reset_index(drop=True)], axis=1)
        val_results_df = val_losses_df

        # train_losses_df['generalization_error'] = train_generalization_errors
        # val_losses_df['generalization_error'] = val_generalization_errors

        return train_results_df, val_results_df

    def compute_link_flows(self,X):

        return self.link_flows(self.path_flows(self.path_probabilities(self.path_utilities(self.link_utilities(X)))))


    def call(self, X):
        """
        X: tensor of link features of dimension (n_daus, n_hours, n_links, n_features)

        return tensor of dimension (n_days, n_links)
        """

        return self.compute_link_flows(X)

class AETSUELOGIT(AESUELOGIT):
    """ Auto-encoded travel time based stochastic user equilibrium with logit assignment"""

    def __init__(self,
                 endogenous_traveltimes = True,
                 *args,
                 **kwargs):

        kwargs.update({'endogenous_flows': False})

        self.endogenous_traveltimes = endogenous_traveltimes

        super().__init__(*args, **kwargs)

    def create_tensor_variables(self, keys: Dict[str, bool] = None):

        if self.endogenous_traveltimes:

            self._traveltimes = tf.Variable(
                # initial_value=tf.math.sqrt(tf.constant(tf.zeros(self.n_links, dtype=tf.float64))),
                initial_value=tf.math.sqrt(tf.tile(tf.expand_dims(self.tt_ff,0),tf.constant([self.n_hours,1]))),
                trainable= self.endogenous_traveltimes,
                name='traveltimes',
                dtype=self.dtype)

        AESUELOGIT.create_tensor_variables(self, keys=keys)

    def traveltimes(self):
        """
        Return exogenous or endogenous travel times via tensorflow constant or variables, respectively
        """

        if self.endogenous_traveltimes:
            # return tensorflow variable of dimension (n_hours, n_links) and initialized using average over hours-links
            return tf.math.pow(self._traveltimes,2)

        else:
            #TODO: tensorflow constant from Y tensor
           
            pass



    def call(self, X):
        """

        X: tensor of link features of dimension (n_daus, n_hours, n_links, n_features)

        return matrix of dimension (n_days, n_links)
        """

        return self.link_traveltimes(self.compute_link_flows(X))


class ODLUE(AESUELOGIT):

    # ODLUE is an extension of the logit utility estimation (LUE) problem solved in the isuelogit package.  It is
    # fed with link features of multiple time days and it allows for the estimation of the OD matrix and of the
    # utility function parameters. It is framed as a computational graph which facilitates the computation of gradient
    # of the loss function.

    # The current implementation assumes that the path sets and the OD matrix is the same for all time days.
    # However, if the path set were updated dynamically, the path sets would vary among time days and od pairs over
    # iterations. As consequence, the matrices M and C would change frequently, which may be computationally intractable.

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def create_tensor_variables(self, keys: Dict[str, bool] = None):
        if keys is None:
            keys = dict.fromkeys(['q', 'theta', 'psc_factor', 'fixed_effect'], True)
            # keys = dict.fromkeys(['alpha', 'beta'], False)

        AESUELOGIT.create_tensor_variables(self, keys=keys)

    def call(self, X):
        """
        X is tensor of dimension (n_days, n_hours, n_links, n_features)
        """

        return self.compute_link_flows(X)
