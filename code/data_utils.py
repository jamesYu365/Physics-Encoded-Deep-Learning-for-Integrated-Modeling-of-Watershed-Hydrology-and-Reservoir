# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch

def generate_train_val_test_phy(train_set, val_set, test_set, wrap_length):
    """
    Generate training, validation, and test input and target data for reservoir LSTM.
    
    Args:
        train_set (pd.DataFrame): Training dataset
        val_set (pd.DataFrame): Validation dataset
        test_set (pd.DataFrame): Test dataset
        wrap_length (int): Length of each sample sequence
    
    Returns:
        tuple: Numpy arrays of input and target data for training, validation, and test sets
               Shape: [Batch_size, time_len, feature]
    """
    # Separate input features and target variable
    train_x_np = train_set.values[:, :-1]
    train_y_np = train_set.values[:, -1:]
    val_x_np = val_set.values[:, :-1]
    val_y_np = val_set.values[:, -1:]
    test_x_np = test_set.values[:, :-1]
    test_y_np = test_set.values[:, -1:]
    
    # Calculate number of samples for training set
    # Each sample has length wrap_length, with a gap of 365 days between starts
    wrap_number_train = (train_set.shape[0] - wrap_length) // 365 + 1
    
    # Initialize empty arrays for training data
    train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))

    # Prepare validation and test data (single batch)
    val_x = np.expand_dims(val_x_np, axis=0)
    val_y = np.expand_dims(val_y_np, axis=0)
    test_x = np.expand_dims(test_x_np, axis=0)
    test_y = np.expand_dims(test_y_np, axis=0)
    
    # Generate training samples
    for i in range(wrap_number_train):
        start_idx = i * 365
        end_idx = wrap_length + i * 365
        train_x[i, :, :] = train_x_np[start_idx:end_idx, :]
        train_y[i, :, :] = train_y_np[start_idx:end_idx, :]
             
    return train_x, train_y, val_x, val_y, test_x, test_y


def get_wrapped_data(data_set, target_var, wrap_length=365):
    """
    Generate input and target data from dataframe for watershed LSTM.
    
    Args:
        data_set (pd.DataFrame): Input dataset
        target_var (str): Name of the target variable column
        wrap_length (int): Length of each sample sequence (default: 365)
    
    Returns:
        tuple: Numpy arrays of input and target data
               Shape: [Batch_size * time_len, feature]
    """
    # Separate input features and target variable
    x = data_set.drop(columns=target_var).values
    y = np.expand_dims(data_set[target_var].values, axis=-1)
    
    new_x, new_y = [], []
    
    # Generate samples using sliding window approach
    for i in range(x.shape[0] - wrap_length):
        # Input: sequence of wrap_length days, starting from day i+1
        new_x.append(x[i+1:i+wrap_length+1, :])
        # Target: value on the day following the input sequence
        new_y.append(y[i+wrap_length, :])
    
    # Stack all samples into 3D arrays
    return np.stack(new_x), np.stack(new_y)

