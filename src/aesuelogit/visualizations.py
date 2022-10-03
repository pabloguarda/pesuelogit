import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
from isuelogit.estimation import compute_vot
from typing import Union, Dict, List, Tuple
from isuelogit.mytypes import Matrix



def plot_convergence_estimates(estimates: pd.DataFrame,
                               true_values: Dict = None):
    # # Add vot
    # estimates = estimates.assign(vot=true_values.apply(compute_vot, axis=1))

    estimates = pd.melt(estimates, ['epoch'], var_name = 'parameter')

    true_values = pd.Series(true_values).to_frame().T
    true_values = true_values[estimates['parameter'].unique()]

    # #Add vot
    # true_values = true_values.assign(vot=true_values.apply(compute_vot, axis=1))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:len(estimates['parameter'].unique())]

    fig, ax = plt.subplots(nrows=1, ncols=1)

    g = sns.lineplot(data=estimates, x='epoch', hue='parameter', y='value')

    ax.hlines(y=true_values.values,
              xmin=estimates['epoch'].min(), xmax=estimates['epoch'].max(), colors=colors, linestyle='--')


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
    fig.legend(loc = "upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    fig.show()

def plot_predictive_performance(train_losses: pd.DataFrame,
                                val_losses: pd.DataFrame,
                                xticks_spacing: int = 5,
                                **kwargs) -> None:

    fig, ax = plt.subplots()

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.plot(train_losses['epoch'], train_losses['loss_tt'], label="Travel time loss (train)", color='red', linestyle='-')
    ax.plot(val_losses['epoch'], val_losses['loss_tt'], label="Travel time loss (test)", color='red',linestyle='--')
    ax.plot(train_losses['epoch'], train_losses['loss_flow'], label="Link flow loss (train)", color='blue', linestyle='-')
    ax.plot(val_losses['epoch'], val_losses['loss_flow'], label="Link flow loss (test)", color='blue', linestyle='--')
    # ax.plot(train_losses['epoch'], train_losses['loss_od'], label="Train loss (od)", color='green', linestyle='-')
    # ax.plot(val_losses['epoch'], val_losses['loss_od'], label="Test loss (od)", color='green', linestyle='--')

    # plt.plot(train_losses['epoch'], train_losses['loss_bpr'], label="Train loss (bpr)", color='gray', linestyle='-')
    # plt.plot(val_losses['epoch'], val_losses['loss_bpr'], label="Test loss (bpr)", color='gray',
    #          linestyle='--')

    ax.plot(train_losses['epoch'], train_losses['loss_eq_flow'], label="Equilibrium loss (train)", color='gray', linestyle='-')
    ax.plot(val_losses['epoch'], val_losses['loss_eq_flow'], label="Equilibrium loss (test)", color='gray', linestyle='--')

    if 'generalization_error' in train_losses.keys():
        plt.plot(train_losses['epoch'], train_losses['generalization_error'], label="Train loss (generalization)",
                 color='black', linestyle='-')
    if 'generalization_error' in val_losses.keys():
        plt.plot(val_losses['epoch'], val_losses['generalization_error'], label="Test loss (generalization)",
                 color='black', linestyle='--')

    # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
    plt.xticks(np.arange(train_losses['epoch'].min(), train_losses['epoch'].max() + 1, xticks_spacing))
    plt.xlim(xmin=train_losses['epoch'].min(), xmax=train_losses['epoch'].max()),

    # plt.ylim(ymin=0, ymax=100)
    plt.ylim(ymin=0)
    plt.xlabel('epoch')

    # ax.set_ylabel('loss')
    ax.set_ylabel('relative loss (%)')
    # ax.set_ylabel('change in equilibrium metric (%)')

    # ax1.legend(loc = 0)
    # ax2.legend(loc = 1)
    fig.legend(loc = "upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

    # fig.show()



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
        sns.boxplot(x="level", y="bias", data=results[results.parameter == 'vot'], color="white", showfliers=False, ax=ax[(0, 1)])
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
    ax[(1, 0)].set_yticks(np.arange(0, 1 + 0.1, 0.2))
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
