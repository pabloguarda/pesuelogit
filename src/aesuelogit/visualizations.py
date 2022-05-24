import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_predictive_performance(train_losses: pd.DataFrame,
                                val_losses: pd.DataFrame) -> None:
    fig = plt.figure()

    plt.plot(train_losses['epoch'], train_losses['tt'], label="Train loss (travel time)", color='red',
             linestyle='-')
    plt.plot(val_losses['epoch'], val_losses['tt'], label="Validation loss (travel time)", color='red',
             linestyle='--')
    plt.plot(train_losses['epoch'], train_losses['flow'], label="Train loss (flow)", color='blue', linestyle='-')
    plt.plot(val_losses['epoch'], val_losses['flow'], label="Validation loss (flow)", color='blue',
             linestyle='--')
    plt.plot(train_losses['epoch'], train_losses['bpr'], label="Train loss (bpr)", color='gray', linestyle='-')
    plt.plot(val_losses['epoch'], val_losses['bpr'], label="Validation loss (bpr)", color='gray',
             linestyle='--')

    if 'generalization_error' in train_losses.keys():
        plt.plot(train_losses['epoch'], train_losses['generalization_error'], label="Train loss (eq flow)",
                 color='gray', linestyle='-')
    if 'generalization_error' in val_losses.keys():
        plt.plot(val_losses['epoch'], val_losses['generalization_error'], label="Validation loss (eq flow)",
                 color='green',linestyle='--')

    plt.xticks(np.arange(train_losses['epoch'].min(), train_losses['epoch'].max() + 1, 5))
    plt.xlim(xmin=train_losses['epoch'].min(), xmax=train_losses['epoch'].max())
    plt.ylim(ymin=0, ymax=100)
    plt.xlabel('epoch')
    plt.ylabel('decrease in mse (%)')
    plt.legend()

    fig.show()


def congestion_map():
    raise NotImplementedError
