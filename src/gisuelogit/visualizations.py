import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
plt.style.use('default')

import seaborn as sns
import pandas as pd
import numpy as np
from isuelogit.estimation import compute_vot
from typing import Union, Dict, List, Tuple
from isuelogit.mytypes import Matrix
from .models import compute_rr



def plot_convergence_estimates(estimates: pd.DataFrame,
                               true_values: Dict = None,
                               xticks_spacing: int = 5):
    # # Add vot
    # estimates = estimates.assign(vot=true_values.apply(compute_vot, axis=1))

    estimates = pd.melt(estimates, ['epoch'], var_name = 'parameter')

    # #Add vot
    # true_values = true_values.assign(vot=true_values.apply(compute_vot, axis=1))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:len(estimates['parameter'].unique())]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (5,4))

    g = sns.lineplot(data=estimates, x='epoch', hue='parameter', y='value')

    if true_values is not None:

        true_values = pd.Series(true_values).to_frame().T
        true_values = true_values[estimates['parameter'].unique()]

        ax.hlines(y=true_values.values,
                  xmin=estimates['epoch'].min(), xmax=estimates['epoch'].max(), colors=colors, linestyle='--')

    # ax.grid(False)

    # fig.set_size_inches(4, 3)

    plt.xticks(np.arange(estimates['epoch'].min(), estimates['epoch'].max() + 1, xticks_spacing))
    plt.xlim(xmin=estimates['epoch'].min(), xmax=estimates['epoch'].max())

    plt.legend(prop={'size': 10})

    return fig, ax

def plot_predictive_performance(train_losses: pd.DataFrame,
                                val_losses: pd.DataFrame = None,
                                xticks_spacing: int = 5,
                                show_validation = False,
                                **kwargs) -> None:

    fig, ax = plt.subplots(figsize = (5,4))

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.plot(train_losses['epoch'], train_losses['loss_tt'], label="Travel times", color='red',
            linestyle='-')
    ax.plot(train_losses['epoch'], train_losses['loss_flow'], label="Link flows", color='blue',
            linestyle='-')
    ax.plot(train_losses['epoch'], train_losses['loss_eq_flow'], label="Equilibrium", color='gray',
            linestyle='-')
    # ax.plot(train_losses['epoch'], train_losses['loss_od'], label="OD loss", color='green', linestyle='-')

    if show_validation:

        ax.plot(train_losses['epoch'], train_losses['loss_tt'], label="Travel times (train)", color='red',
                linestyle='-')
        ax.plot(train_losses['epoch'], train_losses['loss_flow'], label="Link flows (train)", color='blue',
                linestyle='-')
        ax.plot(train_losses['epoch'], train_losses['loss_eq_flow'], label="Equilibrium (train)", color='gray',
                linestyle='-')
        # ax.plot(train_losses['epoch'], train_losses['loss_od'], label="OD loss (train)", color='green', linestyle='-')

        ax.plot(val_losses['epoch'], val_losses['loss_tt'], label="Travel times (test)", color='red',linestyle='--')
        ax.plot(val_losses['epoch'], val_losses['loss_flow'], label="Link flows (test)", color='blue', linestyle='--')
        ax.plot(val_losses['epoch'], val_losses['loss_eq_flow'], label="Equilibrium (test)", color='gray',
                linestyle='--')

        # ax.plot(val_losses['epoch'], val_losses['loss_od'], label="OD loss (test)", color='green', linestyle='--')

    if 'generalization_error' in train_losses.keys():
        plt.plot(train_losses['epoch'], train_losses['generalization_error'], label="Train loss (generalization)",
                 color='black', linestyle='-')
    if show_validation and 'generalization_error' in val_losses.keys():
        plt.plot(val_losses['epoch'], val_losses['generalization_error'], label="Test loss (generalization)",
                 color='black', linestyle='--')

    # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
    plt.xticks(np.arange(train_losses['epoch'].min(), train_losses['epoch'].max() + 1, xticks_spacing))
    plt.xlim(xmin=train_losses['epoch'].min(), xmax=train_losses['epoch'].max())

    # plt.ylim(ymin=0, ymax=100)
    plt.ylim(ymin=0)
    plt.xlabel('epoch')

    # ax.set_ylabel('loss')
    ax.set_ylabel('relative loss (%)')
    # ax.set_ylabel('change in equilibrium metric (%)')

    # ax1.legend(loc = 0)
    # ax2.legend(loc = 1)
    plt.legend(loc = "upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes, prop={'size': 10})

    # fig.show()

    # ax.grid(False)

    # plt.legend(prop={'size': 8})

    return fig, ax

def plot_predictive_performance_twoaxes(train_losses: pd.DataFrame,
                                val_losses: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots()

    ax1.plot(train_losses['epoch'], train_losses['loss_tt'], label="Travel time loss (train)", color='red',
             linestyle='-')
    ax1.plot(val_losses['epoch'], val_losses['loss_tt'], label="Travel time loss (test)", color='red',
             linestyle='--')
    ax1.plot(train_losses['epoch'], train_losses['loss_flow'], label="Train flow loss (train)", color='blue', linestyle='-')
    ax1.plot(val_losses['epoch'], val_losses['loss_flow'], label="Test flow loss (test)", color='blue',
             linestyle='--')
    # plt.plot(train_losses['epoch'], train_losses['loss_bpr'], label="Train loss (bpr)", color='gray', linestyle='-')
    # plt.plot(val_losses['epoch'], val_losses['loss_bpr'], label="Test loss (bpr)", color='gray',
    #          linestyle='--')

    ax2 = ax1.twinx()
    ax2.plot(train_losses['epoch'], train_losses['loss_eq_flow'], label="Equilibrium loss (train)", color='gray', linestyle='-')
    ax2.plot(val_losses['epoch'], val_losses['loss_eq_flow'], label="Equilibrium loss (test)", color='gray', linestyle='--')

    if 'generalization_error' in train_losses.keys():
        plt.plot(train_losses['epoch'], train_losses['generalization_error'], label="Train loss (generalization)",
                 color='black', linestyle='-')
    if 'generalization_error' in val_losses.keys():
        plt.plot(val_losses['epoch'], val_losses['generalization_error'], label="Test loss (generalization)",
                 color='black', linestyle='--')

    # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
    plt.xticks(np.arange(train_losses['epoch'].min(), train_losses['epoch'].max() + 1, 5))
    plt.xlim(xmin=train_losses['epoch'].min(), xmax=train_losses['epoch'].max())
    # plt.ylim(ymin=0, ymax=100)
    plt.ylim(ymin=0)
    plt.xlabel('epoch')

    ax1.set_ylabel('change in loss (%)')
    ax2.set_ylabel('change in equilibrium metric (%)')

    # ax1.legend(loc = 0)
    # ax2.legend(loc = 1)
    plt.legend(loc = "upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    fig.show()


def plot_levels_experiment(results: pd.DataFrame,
                           noise: Dict,
                           folder: str = None,
                           range_initial_values=None):

    assert 'level' in results.keys()

    # if folder is None:
    #     folder = self.folder

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 6))

    # 1) bias in estimates
    sns.boxplot(x="level", y="bias", data=results[results.parameter != 'vot'], color="white",
                ax=ax[(0, 0)], showfliers=False)

    # ax[(0, 0)].set_ylabel('bias ' + r"$(\hat{\theta})$")
    ax[(0, 0)].set_ylabel('bias in coefficients')
    if range_initial_values is not None:
        ax[(0, 0)].set(ylim=range_initial_values)
    # ax[(1, 0)].set_ylabel('bias ' + r"$(\hat{\theta} - \theta)$")

    # 2) Bias in value of time
    if 'vot' in results.parameter.values:
        sns.boxplot(x="level", y="bias", data=results[results.parameter == 'vot'], color="white", showfliers=False,
                    ax=ax[(0, 1)])
        ax[(0, 1)].axhline(0, linestyle='--', color='gray')
        ax[(0, 1)].set_ylabel('bias in value of time')

        if range_initial_values is not None:
            # ax[(0, 1)].set(ylim=(-5e-1,5e-1))
            ax[(0, 1)].set(ylim=range_initial_values)

    # 3) NRMSE Test

    # sns.barplot(x="level", y="nrmse_train", data=results, color="white",
    #             errcolor="black", edgecolor="black", linewidth=1.5, errwidth=1.5, ax=ax[(1, 0)])
    sns.barplot(x="level", y="nrmse_val", data=results, color="white",
                errcolor="black", edgecolor="black", linewidth=1.5, errwidth=1.5, ax=ax[(1, 0)])
    ax[(1, 0)].set_yticks(np.arangploye(0, 1 + 0.1, 0.2))
    ax[(1, 0)].set(ylim=(0, 1))
    # ax[(1, 0)].set_ylabel("nrmse training set")
    ax[(1, 0)].set_ylabel("nrmse in test set")

    # 4) Generalization error

    sns.barplot(x="level", y="generalization_val", data=results, color="white",
                errcolor="black", edgecolor="black", linewidth=1.5, errwidth=1.5, ax=ax[(1, 1)])
    ax[(1, 1)].set_yticks(np.arange(0, 1 + 0.1, 0.2))
    ax[(1, 1)].set(ylim=(0, 1))
    ax[(1, 1)].set_ylabel("generalization error")
    # ax[(1, 1)].set_ylabel("generalization error in test set")

    # Change color style to white and black in box plots
    for axi in [ax[(0, 0)],ax[(0, 1)]]:
        for i, box in enumerate(axi.patches):
            box.set_edgecolor("black")
            box.set_facecolor("white")

        plt.setp(axi.patches, edgecolor='black', facecolor='white')
        plt.setp(axi.lines, color='black')

        plt.setp(axi.artists, edgecolor='black', facecolor='white')

    ax[(0, 0)].axhline(0, linestyle='--', color='gray')
    ax[(1, 0)].axhline(0, linestyle='--', color='gray')
    ax[(1, 0)].axhline(noise['flow'], linestyle='--', color='gray')
    ax[(1, 1)].axhline(noise['flow'], linestyle='--', color='gray')
    # ax[(1, 1)].axhline(alpha, linestyle='--', color='gray')

    fig.tight_layout()

    # plt.show()


    # self.save_fig(fig, folder, 'inference_summary')


def plot_top_od_flows_periods(model, df, period_feature, top_k = 10):

    """
    Plot top od pairs according to the variation of the od flows over time periods
    """

    period_keys = df[[period_feature, 'period_id']].drop_duplicates()

    period_ids = list(map(int, np.sort(np.unique(model.original_period_ids[:, 0]))))

    q_df = pd.DataFrame({})
    for i, j in zip(range(model.q.shape[0]),period_ids):
        # q_dict = dict(zip(fresno_network.ods, list(tvodlulpe.q[i].numpy())))
        q_dict = dict(zip(model.triplist, list(model.q[i].numpy())))

        if model.q.shape[0] > 1:
            label_period_feature = period_keys[period_keys['period_id'] == j][period_feature]
        else:
            label_period_feature_1 = period_keys[period_keys['period_id'] == period_ids[0]][period_feature]
            label_period_feature_2 = period_keys[period_keys['period_id'] == period_ids[-1]][period_feature]
            label_period_feature = f"{label_period_feature_1}-{label_period_feature_2}"

        q_df = q_df.append(pd.DataFrame(q_dict, index=[label_period_feature]))

    top_q = q_df[q_df.var().sort_values(ascending=False)[0:top_k].index].sort_index()

    sns.heatmap(top_q.transpose(), linewidth=0.5, cmap="Blues", vmin=0)

    plt.xlabel(period_feature, fontsize=12)
    plt.ylabel('od pair', fontsize=12)

    # plt.show()

    return top_q

def plot_utility_parameters_periods(model, df, period_feature, include_vot = False):

    period_keys = df[[period_feature, 'period_id']].drop_duplicates()

    period_ids = list(map(int,np.sort(np.unique(model.original_period_ids[:, 0]))))

    theta_df = pd.DataFrame({})
    for i,j in zip(range(model.theta.shape[0]), period_ids):
        theta_dict = dict(zip(model.utility.features, list(model.theta[i].numpy())))

        if include_vot:
            theta_dict['vot'] = float(compute_rr(theta_dict))

        label_period_feature = f"{period_ids[0]}-{period_ids[-1]}"

        if model.theta.shape[0]> 1:
            label_period_feature = period_keys[period_keys['period_id'] == j][period_feature]
        else:
            label_period_feature_1 = period_keys[period_keys['period_id'] == period_ids[0]][period_feature].values[0]
            label_period_feature_2 = period_keys[period_keys['period_id'] == period_ids[-1]][period_feature].values[0]
            label_period_feature = f"{label_period_feature_1}-{label_period_feature_2}"

        theta_df = theta_df.append(pd.DataFrame(theta_dict, index=[label_period_feature]))

    if include_vot:
        theta_df[theta_df['vot'].isna()] = 0

    theta_df = theta_df.sort_index()

    cols = theta_df.columns
    theta_df[cols] = theta_df[cols].apply(pd.to_numeric, errors='coerce')

    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    bound = np.nanmax(theta_df.abs().values)

    sns.heatmap(theta_df.transpose(), linewidth=0.5, cmap=cmap,
                vmin = -bound, vmax = bound)

    plt.xlabel(period_feature, fontsize=12)
    plt.ylabel('parameter', fontsize=12)

    # plt.show()

    return theta_df



def plot_heatmap_demands(Qs: Dict[str, Matrix],
                         subplots_dims: Tuple,
                         figsize: Tuple,
                         vmin=None,
                         vmax=None,
                         folderpath: str = None,
                         filename: str = None) -> None:

    """

    Modification of heatmap_demand function from isuelogit package

    Assume list 'Qs' has 4 elements
    """

    fig, ax = plt.subplots(*subplots_dims, figsize=figsize)

    for Q, title, axi in zip(Qs.values(),Qs.keys(),  ax.flat):

        rows, cols = Q.shape

        od_df = pd.DataFrame({'origin': pd.Series([], dtype=int),
                              'destination': pd.Series([], dtype=int),
                              'trips': pd.Series([], dtype=int)})

        counter = 0
        for origin in range(0, rows):
            for destination in range(0, cols):
                # od_df.loc[counter] = [(origin+1,destination+1), N['train'][current_network].Q[(origin,destination)]]
                od_df.loc[counter] = [int(origin + 1), int(destination + 1), Q[(origin, destination)]]
                counter += 1

        od_df.origin = od_df.origin.astype(int)
        od_df.destination = od_df.destination.astype(int)

        od_pivot_df = od_df.pivot_table(index='origin', columns='destination', values='trips')

        # uniform_data = np.random.rand(10, 12)
        sns.heatmap(od_pivot_df, linewidth=0.5, cmap="Blues", vmin = vmin, vmax = vmax, ax = axi)

        axi.set_title(title)

    plt.tight_layout()

    # plt.show()

    # fig.savefig(folderpath + '/' + filename, pad_inches=0.1, bbox_inches="tight")

    # plt.close(fig)

    return fig


def congestion_map():
    raise NotImplementedError
