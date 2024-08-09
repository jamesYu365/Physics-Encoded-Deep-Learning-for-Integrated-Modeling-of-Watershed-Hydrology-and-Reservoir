# -*- coding: utf-8 -*-

import torch
import numpy as np
import pickle
import os

def NSE_loss_BTF(y_pred, y_true, spinup=0):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) loss for input shape [Batch, Time, Feature].
    
    Args:
        y_pred (torch.Tensor): Predicted values
        y_true (torch.Tensor): True values
        spinup (int): Number of days to omit for spinup period (default: 0)
    
    Returns:
        torch.Tensor: Mean NSE loss
    """
    # Omit values in the spinup period
    y_true = y_true[:, spinup:, :]
    y_pred = y_pred[:, spinup:, -1:]
    
    numerator = torch.sum((y_pred - y_true)**2, dim=1)
    denominator = torch.sum((y_true - y_true.mean(dim=1, keepdims=True))**2, dim=1)
    rNSE = numerator / denominator
    return torch.mean(rNSE)

def NSE_loss_BF(y_pred, y_true, spinup=0):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) loss for input shape [Batch, Feature].
    
    Args:
        y_pred (torch.Tensor): Predicted values
        y_true (torch.Tensor): True values
        spinup (int): Number of days to omit for spinup period (default: 0)
    
    Returns:
        torch.Tensor: Mean NSE loss
    """
    # Omit values in the spinup period
    y_true = y_true[spinup:, :]
    y_pred = y_pred[spinup:, -1:]
    
    numerator = torch.sum((y_pred - y_true)**2, dim=0)
    denominator = torch.sum((y_true - y_true.mean(dim=0, keepdims=True))**2, dim=0)
    rNSE = numerator / denominator
    return torch.mean(rNSE)

def save_model(args, best_epoch, epoch, best_model, epoch_model, model, path_t):
    """
    Save arguments and model, then reload the best model.
    
    Args:
        args: Arguments object
        best_epoch (int): Best epoch number
        epoch (int): Current epoch number
        best_model: Best model state dict
        epoch_model: Current epoch model state dict
        model: Model object to load the best state dict
        path_t (str): Path to save the model
    """
    if best_model:
        model_path = os.path.join(path_t, f"{args.model_name}_{best_epoch}.pkl")
        torch.save(best_model, model_path)
    else:
        model_path = os.path.join(path_t, f"{args.model_name}_{epoch}.pkl")
        torch.save(epoch_model, model_path)
    
    # Save arguments
    with open(os.path.join(path_t, f"{args.model_name}_args.pkl"), 'wb') as f:
        pickle.dump(args, f)
    
    # Load the best model
    model.load_state_dict(torch.load(model_path))

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cal_nse(y_pred, y_true):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) score.
    
    Args:
        y_pred (np.array): Predicted values
        y_true (np.array): True values
    
    Returns:
        float: NSE score or np.nan if all true values are NaN
    """
    if (~np.isnan(y_true)).sum() == 0:
        return np.nan
    
    # Remove NaN values
    mask = ~np.isnan(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    
    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - numerator / denominator

def cal_metrics(y_pred, y_true):
    """
    Calculate various performance metrics: Pearson correlation, NSE, KGE, RMSE.
    
    Args:
        y_pred (np.array): Predicted values
        y_true (np.array): True values
    
    Returns:
        tuple: (r, nse, kge, rmse) or (np.nan, np.nan, np.nan, np.nan, np.nan) if all true values are NaN
    """
    if (~np.isnan(y_true)).sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Remove NaN values
    mask = ~np.isnan(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    
    # Calculate metrics
    r = np.corrcoef(y_pred, y_true)[0, 1]
    nse = cal_nse(y_pred, y_true)
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    # Calculate KGE components
    alpha = y_pred.std() / y_true.std()
    beta = y_pred.mean() / y_true.mean()
    kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    
    return r, nse, kge, rmse