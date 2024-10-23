# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib as mpl

# Configure matplotlib settings for consistent plot styling
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams['axes.unicode_minus'] = False 
plt.rcParams.update({"font.size": 14})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Import custom modules
from preparedataset import data_preprocess, traindata_watershed, plot_data_preprocess_phy
from utils import cal_metrics

def single_evaluation_plot(ax, plot_set, obs_name, pred_name, force, title):
    """
    Create a single evaluation plot comparing observed and predicted values.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on
        plot_set (pandas.DataFrame): The dataset to plot
        obs_name (str): Column name for observed values
        pred_name (str): Column name for predicted values
        force (str): Column name for force data (e.g., precipitation)
        title (str): Title for the plot
    """
    # Plot observed and predicted values
    ax.plot(plot_set[obs_name], label="observation", color='black')
    ax.plot(plot_set[pred_name], label="simulation", color='red', lw=1.5, alpha=0.8)
    
    # Create a secondary y-axis for force data
    ax_sub = ax.twinx()
    ax_sub.plot(plot_set.index, plot_set[force], label=force, color='blue', ls='--', alpha=0.3)
    ax_sub.invert_yaxis()
    ax_sub.set_ylabel(force, color='blue', alpha=0.5)
    ax_sub.tick_params(axis='y', colors='blue')
    
    # Calculate and display performance metrics
    r, nse, kge, rmse = cal_metrics(plot_set[pred_name].values, plot_set[obs_name].values)
    ax.set_title(f"{title}: NSE={nse:.3f}, r={r:.2f}, RMSE={rmse:.2f}, KGE={kge:.2f}")
    ax.set_ylabel(f"{title} (m3/s)")
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    
    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_sub.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=12, loc='upper right')

def evaluation_plot(plot_set, fig_name, path_t):
    """
    Create a full evaluation plot with inflow and release subplots.
    
    Args:
        plot_set (pandas.DataFrame): The dataset to plot
        fig_name (str): Name for the figure
        path_t (str): Path to save the figure
    """
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex='row', figsize=(10, 8), dpi=200)
    plt.suptitle(fig_name, fontsize=10, y=0.925, x=0.2)
    
    single_evaluation_plot(axs[0], plot_set, 'inflow_obs', 'inflow_pred', 'prcp', 'Inflow')
    single_evaluation_plot(axs[1], plot_set, 'release_obs', 'release_pred', 'prcp', 'Release')
    
    path = path_t + f'/{fig_name}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_result(args, model, path_t, plot=True):
    """
    Plot evaluation results for training, validation, and testing periods.
    
    Args:
        args: Arguments containing data paths and parameters
        model: The trained model
        path_t (str): Path to save the plots
        plot (bool): Whether to generate plots
    
    Returns:
        tuple: Evaluation results for training, validation, and testing periods
    """
    # Preprocess data
    hydrodata = data_preprocess(args)
    train_x_phy, val_x_phy, test_x_phy, train_set, val_set, test_set = plot_data_preprocess_phy(args, hydrodata)
    train_x_lstm, _, val_x_lstm, _, test_x_lstm, _, train_max_lstm, train_min_lstm = traindata_watershed(args)
    
    # Load reservoir data
    reser = pd.read_csv(args.datapath + '/reservoir/ankang_operation.csv', index_col=0)
    reser.index = pd.to_datetime(reser.index)

    # Set model to evaluation mode
    model.cpu()
    model.eval()
    
    # Generate predictions for each dataset
    train_output_lstm, train_output_phy, train_analyze = model(train_x_lstm, train_x_phy)
    val_output_lstm, val_output_phy, val_analyze = model(val_x_lstm, val_x_phy)
    test_output_lstm, test_output_phy, test_analyze = model(test_x_lstm, test_x_phy)
    
    # Process model outputs
    train_output_lstm = (F.relu(train_output_lstm) * (train_max_lstm[-1] - train_min_lstm[-1]) + train_min_lstm[-1]).cpu().detach().numpy()
    val_output_lstm = (F.relu(val_output_lstm) * (train_max_lstm[-1] - train_min_lstm[-1]) + train_min_lstm[-1]).cpu().detach().numpy()
    test_output_lstm = (F.relu(test_output_lstm) * (train_max_lstm[-1] - train_min_lstm[-1]) + train_min_lstm[-1]).cpu().detach().numpy()
    
    train_output_phy = train_output_phy.cpu().detach().numpy()
    val_output_phy = val_output_phy.cpu().detach().numpy()
    test_output_phy = test_output_phy.cpu().detach().numpy()
    
    train_analyze = train_analyze.cpu().detach().numpy()
    val_analyze = val_analyze.cpu().detach().numpy()
    test_analyze = test_analyze.cpu().detach().numpy()

    # Prepare evaluation datasets
    eval_training = prepare_eval_data(train_set, reser, train_output_lstm, train_output_phy, train_analyze, model)
    eval_validating = prepare_eval_data(val_set, reser, val_output_lstm, val_output_phy, val_analyze, model)
    eval_testing = prepare_eval_data(test_set, reser, test_output_lstm, test_output_phy, test_analyze, model)
    
    # Generate plots if required
    if plot:
        evaluation_plot(eval_training, 'Training Period', path_t)
        evaluation_plot(eval_validating, 'Validating Period', path_t)
        evaluation_plot(eval_testing, 'Testing Period', path_t)
    
    return (eval_training,), (eval_validating,), (eval_testing,)

def prepare_eval_data(data_set, reser, output_lstm, output_phy, analyze, model):
    """
    Prepare evaluation data for a given dataset.
    
    Args:
        data_set (pandas.DataFrame): The original dataset
        reser (pandas.DataFrame): Reservoir data
        output_lstm (numpy.array): Watershed LSTM output
        output_phy (numpy.array): Reservoir LSTM output
        s2 (numpy.array): Storage data
        model: The trained model
    
    Returns:
        pandas.DataFrame: Prepared evaluation data
    """
    eval_data = data_set.copy()
    eval_data['storage_obs'] = reser['storage'] / model.reservoir.reser_maxstgNF.numpy()
    eval_data['inflow_obs'] = eval_data['inflow'] * 35200 * 1000 / 86400
    eval_data['release_obs'] = eval_data['release'] * 35200 * 1000 / 86400
    eval_data['inflow_pred'] = output_lstm[:, 0] * 35200 * 1000 / 86400
    eval_data['release_pred'] = output_phy[0, :, :] * 35200 * 1000 / 86400
    eval_data['storage_pred'] = analyze[0, :, 0] / model.reservoir.reser_maxstgNF
    eval_data['res_prcp_pred'] = analyze[0, :, 1] * 35200 * 1000
    eval_data['res_eva_pred'] = analyze[0, :, 2] * 35200 * 1000
    eval_data['res_vertical_pred']=np.cumsum(eval_data['res_prcp_pred']-eval_data['res_eva_pred'])
    return eval_data.iloc[365:, :]


