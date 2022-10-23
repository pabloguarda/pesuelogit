"""
Module for AutoEncoded Stochastic User Equilibrium with Logit Assignment (ODLUE)
"""
import copy
import isuelogit as isl
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from tensorflow import keras

from typing import Dict, List, Tuple, Union
import time
from .networks import Equilibrator, ColumnGenerator, TransportationNetwork
from isuelogit.estimation import Parameter, compute_vot
from .descriptive_statistics import error, mse, rmse, nrmse, btcg_mse, mnrmse,l1norm


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
            if self._true_values is not None:
                initial_value = float(self.true_values[feature])
            else:
                initial_value = self.initial_values[feature]

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


class GISUELOGIT(tf.keras.Model):
    """ Gradient based inverse stochastic user equilibrium with logit assignment"""

    def __init__(self,
                 network: TransportationNetwork,
                 utility: UtilityParameters,
                 endogenous_flows = True,
                 key: str = None,
                 od: ODParameters = None,
                 bpr: Parameters = None,
                 equilibrator_model: tf.keras.Model = None,
                 equilibrator: Equilibrator = None,
                 column_generator: ColumnGenerator = None,
                 *args,
                 **kwargs):

        self.observed_traveltimes = None
        self.endogenous_traveltimes = False
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
        self._parameters = {}

        self.endogenous_flows = endogenous_flows

        self.key = key

        self.network = network
        self.equilibrator = equilibrator
        self.column_generator = column_generator

        self.equilibrator_model = equilibrator_model
        
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

    def create_tensor_variables(self, keys: Dict[str, bool] = None,
                                trainables: Dict[str, bool] = None,
                                initial_values: Dict[str, bool] = None
                                ):

        if keys is None:
            keys = dict.fromkeys(['q', 'theta', 'psc_factor', 'fixed_effect', 'alpha', 'beta'], True)

        trainables_defaults = {'flows': self.endogenous_flows,
                               'alpha': self.bpr.parameters['alpha'].trainable,
                               'beta': self.bpr.parameters['beta'].trainable,
                               'q': self.od.trainable,
                               'theta': self.utility.trainables
                               }

        if trainables is not None:
            for k,v in trainables_defaults.items():
                if k not in trainables.keys():
                    trainables[k] = trainables_defaults[k]
        else:
            trainables = trainables_defaults

        initial_values_defaults = {
            'flows': tf.constant(tf.zeros([self.n_hours,self.n_links], dtype=tf.float64)),
            'alpha': self.bpr.parameters['alpha'].initial_value,
            'beta': self.bpr.parameters['beta'].initial_value,
            'q': self.od.initial_values_array(),
            'theta': self.utility.initial_values,
            'psc_factor':self.utility.initial_values['psc_factor'],
            'fixed_effect': tf.constant(self.utility.initial_values['fixed_effect'],
                                        shape=tf.TensorShape(self.utility.shapes['fixed_effect']), dtype=tf.float64)
        }

        if initial_values is not None:
            for k,v in initial_values_defaults.items():
                if k not in initial_values.keys():
                    initial_values[k] = initial_values_defaults[k]

        else:
            initial_values = initial_values_defaults

        # Link specific effect (act as an intercept)
        # if self.endogenous_flows:
        self._flows = tf.Variable(
            initial_value=tf.math.sqrt(initial_values['flows']),
            # initial_value=tf.constant(tf.zeros([self.n_hours,self.n_links]), dtype=tf.float64),
            trainable= trainables['flows'],
            name='flows',
            dtype=self.dtype)

        # Log is to avoid that parameters are lower and equal than zero.
        # Sqrt is to avoid that that parameters are strictly lower than zero

        if keys.get('alpha', False):
            self._alpha = tf.Variable(
                # self.bpr.parameters['alpha'].initial_value,
                initial_value=np.log(initial_values['alpha']),
                # np.sqrt(self.bpr.parameters['alpha'].initial_value),
                                      trainable= trainables['alpha'],
                                      name=self.bpr.parameters['alpha'].key,
                                      dtype=self.dtype)

            self._parameters['alpha'] = self._alpha

        if keys.get('beta', False):
            self._beta = tf.Variable(
                # self.bpr.parameters['beta'].initial_value,
                initial_value=np.log(initial_values['beta']),
                # np.sqrt(self.bpr.parameters['beta'].initial_value),
                trainable=trainables['beta'],
                name=self.bpr.parameters['beta'].key,
                dtype=self.dtype)

            self._parameters['beta'] = self._beta


        if keys.get('q', False):
            self._q = tf.Variable(initial_value=np.sqrt(initial_values['q']),
                                  trainable=trainables['q'],
                                  name=self.od.key,
                                  dtype=self.dtype)

            self._parameters['q'] = self._q
        #
        # TODO: Meantime, the feature parameters of the utility function can be or be not trained altogether based on
        #  on the value of 'tt'. I should allows for separate estimation

        if keys.get('theta', False):

            self._theta = []

            for feature in self.utility.features:
                self._theta.append(tf.Variable(initial_value= initial_values['theta'][feature],
                                               trainable=trainables['theta'][feature],
                                               name=feature,
                                               dtype=self.dtype))

            self._parameters['theta'] = self._theta
        if keys.get('psc_factor', False):
            # Initialize the psc_factor in a value different than zero to generate gradient
            self._psc_factor = tf.Variable(initial_value=initial_values['psc_factor'],
                                           trainable=trainables['theta']['psc_factor'],
                                           name=self.utility.parameters['psc_factor'].key,
                                           dtype=self.dtype)

            self._parameters['psc_factor'] = self._psc_factor

        if keys.get('fixed_effect', False):
            # Link specific effect (act as an intercept)
            self._fixed_effect = tf.Variable(
                initial_value= initial_values['fixed_effect'],
                trainable=trainables['theta']['fixed_effect'],
                name=self.utility.parameters['fixed_effect'].key,
                dtype=self.dtype)

            self._parameters['fixed_effects'] = self._fixed_effect

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
        return self.project_theta(tf.stack(self._theta))

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

        path_utilities = Vf

        if self.psc_factor == 0:
            return path_utilities
        else:
            return path_utilities + self.psc_factor * tf.math.log(tf.constant(
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


    def mask_predicted_traveltimes(self,x,k, k_threshold = 1e5):

        # mask1 = np.where((k >= k_threshold) | (self.tt_ff == 0), 1, 0)
        # mask2 = np.where((k >= k_threshold), 1, 0)
        # mask3 = np.where((self.tt_ff == 0), 1, 0)
        #
        # return (1-mask1)*(1-mask2)*self.tt_ff * (1 + self.alpha * tf.math.pow(x / k, self.beta)) + (1-mask3)*mask2*self.tt_ff

        mask1 = np.where((k >= k_threshold), 1, 0)
        mask2 = np.where((self.tt_ff == 0), 1, 0)

        tt = self.tt_ff * (1 + self.alpha * tf.math.pow(x / k, self.beta))

        return (1-mask2)*((1-mask1)*tt+ mask1*self.tt_ff)

    def bpr_traveltimes(self, x):
        """
        Link performance function
        """
        # Links capacities
        k = np.array([link.bpr.k for link in self.network.links])

        traveltimes = self.mask_predicted_traveltimes(x, k)
        # traveltimes = self.tt_ff* (1 + self.alpha * tf.math.pow(x / k, self.beta))

        return traveltimes


        # return self.tt_ff * (1 + self.alpha * tf.math.pow(x / k, self.beta))

    def traveltimes(self):
        """ Return tensor variable associated to endogenous travel times (assumed dependent on link flows)"""

        return self.bpr_traveltimes(x=self.flows())

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

        # Without the exponential trick, the estimation of alpha is highly unstable.

        # return tf.clip_by_value(self._alpha, self._epsilon, 1e10)
        # return tf.exp(self._alpha)
        return tf.clip_by_value(tf.exp(self._alpha), 0, 2)
        # return tf.clip_by_value(tf.math.pow(self._alpha,2),0,1e10)

    @property
    def beta(self):
        # return tf.clip_by_value(self._beta, self._epsilon, 10)
        return tf.clip_by_value(tf.exp(self._beta),0,8)
        # return tf.clip_by_value(self._beta, 1, 5)
        # return tf.exp(self._beta)

        # return tf.math.pow(self._beta, 2)
        # return tf.clip_by_value(tf.math.pow(self._beta, 2),0,10)

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

        historic_od = tf.expand_dims(tf.constant(self.od.historic_values[1].q.flatten()), axis=0)
        if tf.rank(pred_q) > 1:
            extra_od_cols = tf.cast(tf.constant(float('nan'), shape=(pred_q.shape[0] - 1, tf.size(historic_od))),
                                    tf.float64)
            historic_od = tf.concat([historic_od, extra_od_cols], axis=0)

        # return tf.expand_dims(tf.constant(self.network.q.flatten()),axis =0)

        return historic_od

    def mask_observed_traveltimes(self,tt,k,k_threshold = 1e5):

        # mask1 = np.where((k >= k_threshold) | (self.tt_ff == 0), 1, 0)
        # mask2 = np.where((k >= k_threshold), 1, 0)
        # mask3 = np.where((self.tt_ff == 0), 1, 0)
        #
        # return (1-mask3)*((1-mask1)*tt + mask2*self.tt_ff)

        mask1 = np.where((k >= k_threshold), 1, 0)
        mask2 = np.where((self.tt_ff == 0), 1, 0)

        return (1-mask2)*((1-mask1)*tt + self.tt_ff)

    def loss_function(self,
                      X,
                      Y,
                      lambdas: Dict[str, float],
                      loss_metric = None
                      ):
        """
        Return a dictionary with keys defined as the different terms of the loss function

        # loss_metric = btcg_mse, mse

        """

        if loss_metric is None:
            # loss_metric = mnrmse
            loss_metric = mse

        lambdas_vals = {'tt': 0.0, 'od': 0.0, 'theta': 0.0, 'flow': 0.0, 'eq_flow': 0.0, 'eq_tt': 0.0}

        assert set(lambdas.keys()).issubset(lambdas_vals.keys()), 'Invalid key in loss_weights attribute'

        for attr, val in lambdas.items():
            lambdas_vals[attr] = val

        loss = dict.fromkeys(list(lambdas_vals.keys()) + ['total'], tf.constant(0, dtype=tf.float64))

        if Y.shape[-1] > 0:
            self.observed_traveltimes, self.observed_flows = tf.unstack(Y,axis = -1)

            self.observed_traveltimes = self.mask_observed_traveltimes(tt = self.observed_traveltimes,
                                                                       k = np.array([link.bpr.k for link in self.network.links]))

            # Under recurrent traffic conditions, we assume that the equilibrium flow and travel time is the same regardless the day  Thus, using self.flows() or self.traveltimes() is preferred.
            # predicted_flow = self.compute_link_flows(X)
            # predicted_traveltimes = self.bpr_traveltimes(predicted_flow)
            # output_flow = predicted_flow

            predicted_flow = self.flows()
            predicted_traveltimes = self.traveltimes()
            output_flow = self.compute_link_flows(X)

            # np.nanmean(self.observed_traveltimes)
            # np.nanmean(predicted_traveltimes)

            loss = {
                'od': loss_metric(actual=tf.constant(self.od.historic_values[1].flatten()),
                                  predicted=self.q),
                'flow': loss_metric(actual=self.observed_flows, predicted=predicted_flow),
                # 'flow': loss_metric(actual=self.observed_flows, predicted=output_flow),
                'tt': loss_metric(actual=self.observed_traveltimes, predicted=predicted_traveltimes),
                # 'theta': tf.reduce_mean(tf.norm(self.theta, 1)),
                # #todo: review bpr or tt loss, should they be equivalent?
                # 'bpr': loss_metric(actual=self.observed_traveltimes, predicted=predicted_traveltimes),
                'total': tf.constant(0, tf.float64)}

        # tf.squeeze(self.observed_flows)[0]-predicted_flow[0]

        #TODO: allows for computation even when they are not endogenous (create method flow())
        if self.endogenous_flows:
            loss['eq_flow'] = loss_metric(actual=self.flows(), predicted=output_flow)

            # loss['eq_flow'] = tf.reduce_mean(tf.divide(predicted_flow,self.flows())) #-1
            # np.nanmean(np.abs(1 * (tf.divide(predicted_flow, self.flows()))))
            # loss['eq_flow'] = l1norm(actual=self.flows(), predicted=predicted_flow)
            # print(np.mean(np.abs(self.flows()-predicted_flow)))
            # np.nanmean(np.abs(1 * (tf.divide(self.flows(),predicted_flow) - 1)))
            # np.nanmean(np.abs(1 * (tf.divide(predicted_flow,self.flows()) - 1)))
            # np.mean(np.abs(100*(self.flows()/predicted_flow-1)))
            # np.mean(self.observed_flows)
        else:
            loss['eq_flow'] = loss_metric(actual=self.bpr_flows(self.traveltimes()), predicted=output_flow)

        if self.endogenous_traveltimes:
            loss['eq_tt'] = loss_metric(actual=self.traveltimes(), predicted=predicted_traveltimes)
            print(np.mean(np.abs(self.traveltimes() - predicted_traveltimes)))
        else:
            loss['eq_tt'] = loss_metric(actual=self.bpr_traveltimes(self.flows()), predicted=predicted_traveltimes)

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

    def normalized_losses(self, losses: pd.DataFrame) -> pd.DataFrame:

        columns = [col for col in losses.columns if col != 'epoch']
        losses[columns] = losses[columns]/losses[losses['epoch'] == losses['epoch'].min()][columns].values

        return losses

        # if losses[0]['total'] == 0:
        #     return losses

        # losses_df = []
        # for epoch, loss in enumerate(losses):
        #     losses_df.append({'epoch': epoch})
        #     for key, val in loss.items():
        #         losses_df[-1][key] = [val.numpy()]
        #         normalizer = losses[0][key].numpy()
        #         if normalizer == 0:
        #             losses_df[-1][key] = [100]
        #         else:
        #             losses_df[-1][key] = losses_df[-1][key] / normalizer * 100
        #
        # return pd.concat([pd.DataFrame(i) for i in losses_df], axis=0, ignore_index=True)

    def get_parameters_estimates(self) -> pd.DataFrame:

        # TODO: extend for multiperiod theta and multilinks alpha, beta
        estimates = {}
        estimates.update(dict(zip(self.utility.features, self.theta.numpy().flatten())))
        estimates.update(dict(zip(['alpha', 'beta'], [np.mean(self.alpha.numpy()), np.mean(self.beta.numpy())])))
        estimates['psc_factor'] = float(self.psc_factor.numpy())

        return pd.DataFrame(estimates, index=[0])

    def get_true_parameters(self) -> pd.DataFrame:

        true_values = {k: v for k, v in {**self.bpr.true_values, **self.utility.true_values}.items()}

        if {'c', 'tt'}.issubset(true_values.keys()):
            true_values['vot'] = compute_vot(true_values)

        return pd.DataFrame({'parameter': true_values.keys(), 'truth': true_values.values()})

    # def best_results(self, results: pd.DataFrame):
    #     return results[results['loss_total'].argmin()]

    def split_results(self, results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        col_losses = ['epoch'] + [col for col in results.columns if any(x in col for x in ['loss_', 'error', 'relative_gap'])]

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
              epochs: Dict[str, int],
              initial_values: Dict[str, float] = None,
              trainables: Dict[str, bool] = None,
              threshold_relative_gap: float = 1e-4,
              loss_metric=None,
              momentum_equilibrium = 1,
              relative_losses = True,
              generalization_error: Dict[str, bool] = None,
              epochs_print_interval:int = 1,
              batch_size=None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        """ It assumes the first column of tensors X_train and X_val are the free flow travel times. The following
        columns are the travel times and exogenous features of each link """

        if batch_size is None:
            batch_size = X_train.shape[0]

        if generalization_error is None:
            generalization_error = {'train': False, 'validation': False}

        X_train, Y_train, X_val, Y_val = map(lambda x: tf.cast(x,tf.float64),[X_train, Y_train, X_val, Y_val])

        self.n_days, self.n_hours, self.n_links, self.n_features = X_train.shape

        self.create_tensor_variables(initial_values = initial_values, trainables=trainables)

        if np.sum(self.flows().numpy()) == 0:
            # Initialization of endogenous travel times and flows
            predicted_flow = self.compute_link_flows(X_train)
            predicted_traveltimes = self.bpr_traveltimes(predicted_flow)

            # if self.endogenous_flows:
            # Smart initialization is performed running a single pass of traffic assignment under initial theta and q
            # self._flows.assign(tf.math.sqrt(tf.reduce_mean(predicted_flow,axis = 0)))
            self._flows.assign(tf.math.sqrt(predicted_flow))
            # self._flows.assign(tf.squeeze(tf.reduce_mean(self.call(tf.unstack(Y_train,axis = -1)[1]),axis=-1)))

            if self.endogenous_traveltimes:
                self._traveltimes.assign(tf.reduce_mean(predicted_traveltimes, axis = 0))

        epoch = 0
        t0 = time.time()
        total_t0 = time.time()

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        # val_dataset = val_dataset.batch(batch_size)

        # Initial Losses
        # train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=loss_weights)['loss_total']
        # val_loss = self.loss_function(X=X_val, Y=Y_val, loss_weights=loss_weights)['total']

        train_losses, val_losses = [], []

        estimates = [self.get_parameters_estimates()]

        # MSE is keep here regardless the selected loss metric so it is printed the true loss
        train_losses = [self.loss_function(X=X_train, Y=Y_train, lambdas=loss_weights, loss_metric=mse)]
        val_losses = [self.loss_function(X=X_val, Y=Y_val, lambdas=loss_weights, loss_metric=mse)]

        relative_gaps = [threshold_relative_gap]
        terminate_algorithm = False

        if 'equilibrium' not in epochs.keys():
            epochs['equilibrium'] = 0

        if 'learning' not in epochs.keys():
            epochs['learning'] = 0

        sue_objectives = []


        while not terminate_algorithm:

            current_sue_objectives = []

            if not terminate_algorithm:

                path_flows = self.path_flows(self.path_probabilities(self.path_utilities(self.link_utilities(X_train))))
                link_flow = self.link_flows(path_flows)
                relative_x = float(np.nanmean(np.abs(tf.divide(link_flow,self.flows()) - 1)))

                for i in range(X_train.shape[0]):

                    sue_objective = sue_objective_function_fisk(f = path_flows[i,0,:].numpy().flatten(),
                                                                X = X_train[i, 0, :, :],
                                                                theta = dict(zip(self.utility.features,self.theta.numpy())),
                                                                k_Z = self.utility.features_Z,
                                                                k_Y= self.utility.features_Y,
                                                                network = self.network)

                    current_sue_objectives.append(sue_objective)

                sue_objectives.append(current_sue_objectives)

                if len(sue_objectives)>=2:
                    relative_gap = np.nanmean(np.abs(np.divide(sue_objectives[-1], sue_objectives[-2]) - 1))
                    relative_gaps.append(relative_gap)
                    # print(f"{relative_gap:0.2g}")
                    # print(sue_objective)

                # print(f"{i}: loss={loss.numpy():0.4g}, theta = {model.theta.numpy()}")

                if generalization_error.get('train', False):
                    train_losses[-1]['generalization_error'] = self.generalization_error(X=X_train, Y=Y_train)
                if generalization_error.get('validation', False):
                    val_losses[-1]['generalization_error'] = self.generalization_error(X=X_val, Y=Y_val)

            if epoch == epochs['learning'] or abs(relative_gaps[-1]) < threshold_relative_gap:
                terminate_algorithm = True

            if epoch % epochs_print_interval == 0 or epoch == 1 or terminate_algorithm:

                print(f"\nEpoch: {epoch}, n_train: {X_train.shape[0]}, n_test: {X_val.shape[0]}")

                print(f"\n{epoch}: train_loss={float(train_losses[-1]['loss_total'].numpy()):0.2g}, "
                    f"val_loss={float(val_losses[-1]['loss_total'].numpy()):0.2g}, "
                    f"train_loss tt={float(train_losses[-1]['loss_tt'].numpy()):0.2g}, "
                    f"val_loss tt={float(val_losses[-1]['loss_tt'].numpy()):0.2g}, "
                    f"train_loss flow={float(train_losses[-1]['loss_flow'].numpy()):0.2g}, "
                    f"val_loss flow={float(val_losses[-1]['loss_flow'].numpy()):0.2g}, "
                    # f"train_loss bpr={float(train_loss['loss_bpr'].numpy()):0.2g}, "
                    # f"val_loss bpr={float(val_loss['loss_bpr'].numpy()):0.2g}, "
                    f"theta = {self.theta.numpy()}, "
                    f"vot = {np.array(compute_vot(self.get_parameters_estimates().to_dict(orient='records')[0])):0.2f}, "
                    f"psc_factor = {self.psc_factor.numpy()}, "
                    f"avg abs theta fixed effect = {np.mean(np.abs(self.fixed_effect)):0.2g}, "
                    f"avg alpha={np.mean(self.alpha.numpy()):0.2g}, avg beta={np.mean(self.beta.numpy()):0.2g}, "
                    # f"avg abs diff demand ={np.nanmean(np.abs(self.q - self.historic_od(self.q))):0.2g}, ",end = '')
                    f"loss demand={float(train_losses[-1]['loss_od'].numpy()):0.2g}, "
                      f"lambda eq={loss_weights['eq_flow']:0.2g}, "
                      f"relative x={relative_x:0.2g}, "
                      f"relative gap={relative_gaps[-1]:0.2g}, ", end='')

                if train_losses[-1].get('loss_eq_tt', False):
                    print(f"train tt equilibrium loss={float(train_losses[-1]['loss_eq_tt'].numpy()):0.2g}, ", end = '')

                if train_losses[-1].get('loss_eq_flow', False):
                    print(f"train flow equilibrium loss={float(train_losses[-1]['loss_eq_flow'].numpy()):0.2g}, ", end = '')

                if generalization_error.get('train', False):
                    print(f"train generalization error ={train_losses[-1]['generalization_error'].numpy():0.2g}, ", end = '')
                if generalization_error.get('validation', False):
                    print(f"val generalization error ={val_losses[-1]['generalization_error'].numpy():0.2g}, ", end = '')

                print(f"time:{time.time() - t0: 0.1f}")

                t0 = time.time()

            if not terminate_algorithm:

                # Gradient based learning

                # current_loss_weights = loss_weights

                # if relative_gap >= 1e-1:
                #     current_loss_weights = loss_weights_eq
                #     # self._theta.trainable = False
                #     optimizer.lr = lr_eq
                # else:
                #     current_loss_weights = loss_weights
                #     self._theta.trainable = True
                #     optimizer.lr = lr

                trainable_variables = self.trainable_variables

                loss_weights['eq_flow'] = loss_weights['eq_flow'] / momentum_equilibrium

                # Normalize weights to one
                loss_weights = {k:(v/np.sum(list(loss_weights.values()))) for k,v in loss_weights.items()}

                for step, (X_batch_train, Y_batch_train) in enumerate(train_dataset):

                    with tf.GradientTape() as tape:
                        train_loss = \
                            self.loss_function(X=X_batch_train, Y=Y_batch_train, lambdas=loss_weights,
                                               loss_metric=loss_metric)['loss_total']

                    grads = tape.gradient(train_loss, trainable_variables)

                    # # Apply some clipping (tf.linalg.normada
                    # grads = [tf.clip_by_norm(g, 2) for g in grads]

                    # # The normalization of gradients in NGD can be hardcoded as
                    # if isinstance(optimizer, NGD):
                    #     grads = [g/tf.linalg.norm(g, 2) for g in grads]

                    optimizer.apply_gradients(zip(grads, trainable_variables))

                # Store losses and estimates
                train_loss = self.loss_function(X=X_train, Y=Y_train, lambdas=loss_weights, loss_metric=mse)
                val_loss = self.loss_function(X=X_val, Y=Y_val, lambdas=loss_weights, loss_metric=mse)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                estimates.append(self.get_parameters_estimates())

                epoch += 1

            if self.column_generator is not None:
                # TODO: Column generation (limit the options to more fundamental ones)
                self.column_generator.generate_paths(theta=self.theta,
                                                     network=self.network
                                                     )

            # TODO: Path set selection (confirm if necessary)

        train_losses_df = pd.concat([pd.DataFrame([losses_epoch], index=[0]).astype(float).assign(epoch = epoch) for epoch, losses_epoch in enumerate(train_losses)])

        val_losses_df = pd.concat([pd.DataFrame([losses_epoch], index=[0]).astype(float).assign(epoch = epoch) for epoch, losses_epoch in enumerate(val_losses)])

        # Replace equilibirum loss in first epoch with the second epoch, to avoid zero loss when initializing utility parameters to zero
        train_losses_df.loc[train_losses_df['epoch'] == 0,'loss_eq_flow'] \
            = train_losses_df.loc[train_losses_df['epoch'] == 1,'loss_eq_flow'].copy()
        val_losses_df.loc[val_losses_df['epoch'] == 0, 'loss_eq_flow'] \
            = val_losses_df.loc[val_losses_df['epoch'] == 1, 'loss_eq_flow'].copy()

        train_losses_df.loc[train_losses_df['epoch'] == 0,'loss_od'] \
            = train_losses_df.loc[train_losses_df['epoch'] == 1,'loss_od'].copy()
        val_losses_df.loc[val_losses_df['epoch'] == 0, 'loss_od'] \
            = val_losses_df.loc[val_losses_df['epoch'] == 1, 'loss_od'].copy()

        train_results_df = pd.concat([train_losses_df.reset_index(drop=True),
                                      pd.concat(estimates, axis=0).reset_index(drop=True)], axis=1).assign(relative_gap = relative_gaps)

        val_results_df = val_losses_df.reset_index(drop=True)

        # train_losses_df['generalization_error'] = train_generalization_errors
        # val_losses_df['generalization_error'] = val_generalization_errors

        if epoch == epochs['learning'] and epochs['equilibrium']>0:

            print('\nEquilibrium stage\n')
            train_results_eq, val_results_eq = self.train_equilibrium(
                X_train = X_train, Y_train = Y_train, X_val = X_val, Y_val = Y_val,
                # generalization_error={'train': False, 'validation': True},
                # loss_metric = mse,
                loss_metric=loss_metric,
                optimizer=optimizer,
                batch_size=batch_size,
                relative_losses = False,
                threshold_relative_gap=threshold_relative_gap,
                epochs_print_interval=epochs_print_interval,
                epochs={'learning':epochs['equilibrium']})

            train_results_eq['epoch'] += train_results_df['epoch'].max()
            val_results_eq['epoch'] += val_results_df['epoch'].max()

            # for i in ['loss_flow','loss_tt','loss_total','loss_eq_flow']:
            #     train_results_eq[i] *= train_results_df.iloc[-1,:][i]
            #     val_results_eq[i] *= val_results_df.iloc[-1, :][i]

            train_results_df = pd.concat([train_results_df,train_results_eq]).reset_index().drop('index', axis = 1)
            val_results_df = pd.concat([val_results_df, val_results_eq]).reset_index().drop('index', axis = 1)

        if relative_losses:
            # Compution of relative losses
            # losses_columns = [column for column in train_losses_df.keys() if column not in ["loss_eq_flow"]]
            losses_columns = [column for column in train_losses_df.keys()]
            train_results_df[losses_columns] = self.normalized_losses(train_results_df[losses_columns])#.assign(loss_eq_flow = train_losses_df['loss_eq_flow'])
            val_results_df[losses_columns] = self.normalized_losses(val_results_df[losses_columns])#.assign(loss_eq_flow = val_losses_df['loss_eq_flow'])


        return train_results_df, val_results_df

    def train_equilibrium(self, **kwargs):

        suelogit = GISUELOGIT(
            key='suelogit',
            # endogenous_flows=True,
            network=self.network,
            dtype=tf.float64,
            equilibrator=self.equilibrator,
            # column_generator=column_generator,
            utility=self.utility,
            bpr=self.bpr,
            od=self.od
        )

        trainables = {'flows': True,
                      'theta': dict(
                          zip(self.utility.trainables.keys(), [False] * len(self.utility.trainables))),
                      'alpha': False,
                      'beta': False,
                      'q': False
                      }

        initial_values = {'flows': self.flows(),
                          'theta': {parameter.name[:-2]: float(parameter)
                                    for parameter in self._parameters['theta']},
                          'alpha': self.alpha,
                          'beta': self.beta,
                          'q': self.q,
                          'fixed_effect': self.fixed_effect,
                          'psc_factor': self.psc_factor,
                          }

        # predicted_flow = self.compute_link_flows(X_train)
        # predicted_traveltimes = self.bpr_traveltimes(predicted_flow)
        # trainable_variables = self._flows

        kwargs.update(trainables=trainables, initial_values=initial_values)

        train_results_eq, val_results_eq = suelogit.train(
           **kwargs, loss_weights={'od': 0, 'theta': 0, 'tt': 0, 'flow': 0, 'eq_flow': 1})

        return  train_results_eq, val_results_eq



    def compute_link_flows(self,X):

        link_flows = self.link_flows(self.path_flows(self.path_probabilities(self.path_utilities(self.link_utilities(X)))))

        if tf.rank(link_flows) == 3:
            link_flows = tf.reduce_mean(link_flows,axis = 0)

        return link_flows


    def call(self, X):
        """
        X: tensor of link features of dimension (n_daus, n_hours, n_links, n_features)

        return tensor of dimension (n_days, n_links)
        """

        return self.compute_link_flows(X)

def almost_zero(array: np.array, tol = 1e-5):
    array[np.abs(array) < tol] = 0

    return array

def entropy_path_flows_sue(f):

    ''' It corrects for numerical issues by masking terms before the entropy computation'''

    epsilon = 1e-12

    return np.sum(almost_zero(f, tol=epsilon) * (np.log(f + epsilon)))

def sue_objective_function_fisk(f,
                                X,
                                network,
                                theta: dict,
                                k_Z: [],
                                k_Y: [] = ['tt']
                                ):

    # links_dict = network.links_dict
    # x_vector = np.array(list(x_dict.values()))
    x_vector = network.D.dot(f)

    if not np.all(f >= 0):
        print('some elements in the path flow vector are negative')

    # Objective function

    # Component for endogeonous attributes dependent on link flow
    bpr_integrals = [float(link.bpr.bpr_integral_x(x=x)) for link, x in
                     zip(network.links, x_vector.flatten().tolist())]
    # bpr_integrals = [float(link.bpr.bpr_integral_x(x=x_dict[i])) for i, link in links_dict.items()]

    tt_utility_integral = float(theta[k_Y[0]]) * np.sum(np.sum(bpr_integrals))

    # Component for exogenous attributes (independent on link flow)
    Z_utility_integral = 0

    # if k_Z:
    #     for attr in k_Z:
            # Zx_vector = np.array(list(network.Z_data[attr]))[:, np.newaxis]
            # Z_utility_integral += float(theta[attr]) * Zx_vector.T.dot(x_vector)

    Z_utility_integral = X.numpy().T.dot(x_vector).dot(np.array([theta[k_z] for k_z in k_Z]).T)

    # Objective function in multiattribute problem
    utility_integral = tt_utility_integral + float(Z_utility_integral)
    # utility_integral = float(Z_utility_integral)

    # entropy = cp.sum(cp.entr(cp.hstack(list(cp_f.values()))))

    entropy_function = entropy_path_flows_sue(f)

    # if not np.all(f > 0):
    #     print('some elements in the path flow vector are 0')
    #     f = no_zeros(f)
    # entropy_function = np.sum(np.log(f)*f)

    # objective_function = utility_integral #- entropy_function
    objective_function = utility_integral - entropy_function

    return float(objective_function)


class AETSUELOGIT(GISUELOGIT):
    """ Auto-encoded travel time based stochastic user equilibrium with logit assignment"""

    def __init__(self,
                 endogenous_traveltimes = True,
                 *args,
                 **kwargs):

        kwargs.update({'endogenous_flows': False})

        super().__init__(*args, **kwargs)

        self.endogenous_traveltimes = endogenous_traveltimes

    def create_tensor_variables(self, keys: Dict[str, bool] = None):

        if self.endogenous_traveltimes:

            self._traveltimes = tf.Variable(
                # initial_value=tf.math.sqrt(tf.constant(tf.zeros(self.n_links, dtype=tf.float64))),
                initial_value=tf.tile(tf.expand_dims(self.tt_ff, 0), tf.constant([self.n_hours, 1])),
                # initial_value=tf.math.sqrt(tf.tile(tf.expand_dims(self.tt_ff,0),tf.constant([self.n_hours,1]))),
                # initial_value=tf.math.sqrt(tf.tile(tf.constant(self.tt_ff[tf.newaxis,tf.newaxis,:]),
                #                                    tf.constant([self.n_days, self.n_hours,1]))),
                trainable= self.endogenous_traveltimes,
                name='traveltimes',
                dtype=self.dtype)

        GISUELOGIT.create_tensor_variables(self, keys=keys)

    def traveltimes(self, link_flow = None):
        """
        return tensorflow variable of dimension (n_hours, n_links) and initialized using average over hours-links
        """
        return tf.clip_by_value(self._traveltimes, clip_value_min = tf.constant(self._tt_ff), clip_value_max = tf.float64.max)
        # return tf.math.pow(self._traveltimes,2)

        # return tf.math.pow(self._traveltimes,2)

    def bpr_flows(self, t):
        """
        Inverse of link performance function. Subject to large approximation errors when beta=4
        """
        # Links capacities
        k = np.array([link.bpr.k for link in self.network.links])

        return tf.math.sqrt((t/self.tt_ff-1)/self.alpha, self.beta)*k

        # return self.tt_ff * (1 + self.alpha * tf.math.pow(x / k, self.beta))

    def call(self, X):
        """

        X: tensor of link features of dimension (n_daus, n_hours, n_links, n_features)

        return matrix of dimension (n_days, n_links)
        """

        return self.bpr_traveltimes(self.compute_link_flows(X))


class ODLUE(GISUELOGIT):

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

        GISUELOGIT.create_tensor_variables(self, keys=keys)

    def call(self, X):
        """
        X is tensor of dimension (n_days, n_hours, n_links, n_features)
        """

        return self.compute_link_flows(X)
