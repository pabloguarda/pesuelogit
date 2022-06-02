import pandas as pd
import numpy as np
import os
import copy
import isuelogit as isl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
from isuelogit.printer import block_output, printProgressBar, printIterationBar

# from isuelogit.experiments import NetworkExperiment
from .visualizations import *
from .descriptive_statistics import mse, rmse, nrmse


class NetworkExperiment(isl.experiments.NetworkExperiment):

    def __init__(self, model, optimizer, X, Y, noise: Dict = None, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        # self.utility_function = self.model.utility
        self.X = X
        self.Y = Y
        self.noise = noise
        self.optimizer = optimizer

    def setup_experiment(self,
                         replicates=None,
                         epochs=None,
                         batch_size=None,
                         range_initial_values: tuple = None):

        print('\n' + self.options['name'], end='\n')

        # self.config.set_experiments_log_files(networkname=self.config.sim_options['current_network'].lower())

        self.options['range_initial_values'] = range_initial_values

        # if epochs is not None:
        self.options['epochs'] = epochs

        # if batch_size is not None:
        self.options['batch_size'] = batch_size

        # if replicates is not None:
        self.options['replicates'] = replicates

    def write_experiment_report(self,
                                folder=None,
                                filename=None):
        # self.write_report(filepath = self.dirs['experiment_folder'] + '/' + 'experiment_options.csv')

        if filename is None:
            filename = 'experiment_report.csv'
        if folder is None:
            folder = os.path.join(os.getcwd(), 'output/experiments', self.model.network.key)
            # folder = self.dirs['experiment_folder']

        filepath = os.path.join(folder, filename)

        # Network information
        if self.network is not None:
            self.options['network'] = self.network.key
            self.options['links'] = len(self.network.links)
            self.options['paths'] = len(self.network.paths)
            self.options['ods'] = len(self.network.ods)
            self.options['scale_OD'] = self.network.OD.scale

        self.options['features'] = self.model.utility.features
        # self.options['initial parameters'] = utility_function.initial_values
        self.options['true parameters'] = self.model.utility.true_values

        # self.options['data_generator'] = self.linkdata_generator.options

        # TODO: Add model options

        # for learner in self.learners:
        #     self.options[learner.name + '_learner'] \
        #         = {k: v for k, v in learner.options.items() if
        #            k in ['bilevel_iters']}
        #
        #     self.options[learner.name + '_optimizer'] \
        #         = {k: v for k, v in learner.outer_optimizer.options.items() if
        #            k in ['method', 'eta', 'iters']}
        #
        #     self.options[learner.name + '_equilibrator'] = {k: v for k, v in learner.equilibrator.options.items()}

        df = pd.DataFrame({'option': self.options.keys(), 'value': self.options.values()})

        df.to_csv(filepath,
                  sep=',',
                  encoding='utf-8',
                  index=False)

    def train_test_split(self, **kwargs):

        if kwargs['test_size'] > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(self.X.numpy(), self.Y.numpy(), **kwargs)
        else:
            X_train, X_val, Y_train, Y_val = self.X, np.array([]), self.Y, np.array([])

        return [tf.constant(i) for i in [X_train, X_val, Y_train, Y_val]]

    def add_gaussian_error(self, tensor: tf.float64, std_mean: float = 0) -> tf.float64:

        """std_mean: standard deviation of Gaussian noise is the product between std_mean and the mean of tensor"""

        # if std_mean <= 0:
        #     return tensor

        noisy_tensor = isl.factory.LinkDataGenerator().add_error_counts(
            original_counts=tensor.numpy().flatten()[:, np.newaxis], sd_x=std_mean)

        return tf.constant(tf.reshape(noisy_tensor.flatten(),tensor.shape),tf.float64)


class ConvergenceExperiment(NetworkExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, batch_size, epochs, loss_weights, test_size=0):
        X_train, X_val, Y_train, Y_val = self.train_test_split(test_size=test_size, random_state=42)

        train_results_df, val_results_df = self.model.train(
            X_train, Y_train, X_val, Y_val,
            optimizer=self.optimizer,
            batch_size=batch_size,
            loss_weights=loss_weights,
            generalization_error ={'train': True},
            epochs=epochs)

        train_results_estimates, train_results_losses = self.model.split_results(results=train_results_df)
        val_results_estimates, val_results_losses = self.model.split_results(results=val_results_df)

        plot_predictive_performance(train_losses=train_results_losses, val_losses=val_results_losses)

        true_values = pd.Series(
            {k: v for k, v in {**self.model.bpr.true_values, **self.model.utility.true_values}.items()
             if k in train_results_estimates.columns})

        plot_convergence_estimates(estimates=train_results_estimates, true_values=true_values)


class MultidayExperiment(NetworkExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self,
            levels: List,
            loss_weights: Dict,
            show_replicate_plot=True,
            replicate_report=True,
            range_initial_values=None,
            **kwargs):

        # Update options
        self.setup_experiment(**kwargs)

        # self.utility_function.add_sparse_features(Z=['k' + str(i) for i in np.arange(0, n_sparse_features)])

        self.write_experiment_report()

        # range_initial_values = self.options.get('range_initial_values')
        replicates = self.options['replicates']

        self.options['levels'] = levels
        # self.options['type'] = type

        results_experiment = pd.DataFrame({})

        # Noise or scale difference in Q matrix
        # sd_x = self.linkdata_generator.options['noise_params']['sd_x']

        for replicate in range(1, replicates + 1):

            if replicate_report or show_replicate_plot:
                print('\nReplicate', replicate)
            elif not replicate_report:
                printIterationBar(replicate, replicates, prefix='Replicates:', length=20)

            self.create_replicate_folder(replicate=replicate)

            # Initilization of initial estimate
            if range_initial_values is not None:
                self.model.utility.random_initializer(range_values=range_initial_values,
                                                      keys=self.model.utility.features)
                self.model.create_tensor_variables(keys = {'theta': True})
            # else:
            #     self.model.utility.zero_initializer()

            # initial_values = copy.deepcopy(self.model.utility.initial_values)

            results_replicate = pd.DataFrame({})

            Qs = {'true': self.model.network.OD.Q_true}

            with block_output(show_stdout=replicate_report, show_stderr=replicate_report):
                for level in levels:
                    X_train, X_val, Y_train, Y_val = self.train_test_split(test_size=self.X.shape[0] - level)

                    tt_train, flow_train = tf.unstack(Y_train, axis=3)
                    tt_val, flow_val = tf.unstack(Y_val, axis=3)

                    # Add random error
                    noisy_tt_train = self.add_gaussian_error(tt_train, std_mean=self.noise['tt'])
                    noisy_flow_train = self.add_gaussian_error(flow_train, std_mean=self.noise['flow'])

                    Y_train = tf.stack([noisy_tt_train, noisy_flow_train], axis=3)

                    # self.model.utility.initial_values = initial_values

                    train_results_df, val_results_df = self.model.train(
                        X_train, Y_train, X_val, Y_val,
                        optimizer=self.optimizer,
                        batch_size=self.options['batch_size'],
                        loss_weights=loss_weights,
                        epochs=self.options['epochs'])

                    # self.write_convergence_table(results_norefined=learning_results_norefined,
                    #                              results_refined=learning_results_refined,
                    #                              folder=self.dirs['replicate_folder'],
                    #                              filename='convergence_' + str(level) + '.csv')

                    best_results = pd.DataFrame(train_results_df.loc[train_results_df['loss_total'].argmin(), :]).T

                    Qs[level] = tf.sparse.to_dense(self.model.Q).numpy()

                    results_parameters, results_losses = self.model.split_results(best_results)

                    results_parameters['vot'] = compute_vot(pd.Series(results_parameters.to_dict(orient='records')[0]))

                    results_parameters = pd.melt(results_parameters,
                                                 id_vars='epoch', var_name='parameter', value_name='estimate')

                    results_parameters = pd.merge(results_parameters, self.model.get_true_parameters(), how='left')

                    results = results_parameters.assign(
                        replicate=replicate,
                        level=level,
                        bias=results_parameters.eval('estimate-truth'),
                        loss=float(results_losses['loss_total']),
                        # nrmse_train=float(nrmse(actual=noisy_flow_train, predicted=self.model.predict_flow(X_train))),
                        nrmse_val=float(nrmse(actual=flow_val, predicted=self.model.compute_link_flows(X_val))),
                        generalization_val = float(self.model.generalization_error(Y = Y_val, X = X_val,loss_metric = nrmse))
                    )

                    results_replicate = pd.concat([results_replicate, results])

                    # Reset initial values of tensor variables
                    self.model.create_tensor_variables()

                results_experiment = pd.concat([results_experiment, results_replicate])

                # self.write_replicate_table(df=results_replicate, filename='inference', replicate=replicate)

                fig = plot_levels_experiment(
                    results=results_experiment,
                    noise=self.noise,
                    folder=self.dirs['replicate_folder'],
                    range_initial_values=range_initial_values)

                plot_heatmap_demands(Qs = Qs, vmin = np.min(Qs['true']), vmax = np.max(Qs['true']))

                if show_replicate_plot:
                    plt.show()
                else:
                    plt.close(fig)

        plot_levels_experiment(
            results=results_experiment,
            noise=self.noise,
            folder=self.dirs['experiment_folder'],
            range_initial_values=range_initial_values)

        plt.show()


