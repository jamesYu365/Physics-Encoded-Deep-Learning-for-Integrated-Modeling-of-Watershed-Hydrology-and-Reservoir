# -*- coding: utf-8 -*-


import torch
import numpy as np
import pandas as pd
import datetime

from data_utils import generate_train_val_test_phy,get_wrapped_data


def data_preprocess(args):
    """
    Preprocess reservoir operation records and meteorological data.
    
    Args:
        args: Configuration arguments containing datapath and other settings.
    
    Returns:
        pd.DataFrame: Processed hydrodata with selected features.
    """
    # Read the CSV file
    hydrodata = pd.read_csv(args.datapath + 'processed_ankang_inflow_release_meteo_1991_2014.csv', index_col=0)
    hydrodata.index = pd.to_datetime(hydrodata.index)
    
    # Convert flow units from m³/s to mm/day
    hydrodata['inflow'] = hydrodata['inflow'] * 86400 / 35200 / 1000 
    hydrodata['release'] = hydrodata['release'] * 86400 / 35200 / 1000
    
    # Calculate seasonality factors
    hydrodata['doy'] = hydrodata['doy'] / 365
    hydrodata['cosdoy'] = (np.cos(hydrodata['doy'] * 2 * np.pi) + 1) / 2
    hydrodata['doy'] = hydrodata['doy'] * 365
    
    # Select relevant features
    hydrodata = hydrodata[['prcp', 'tmean', 'doy', 'cosdoy', 'inflow', 'release']]
    
    return hydrodata


def traindata_watershed(args):
    """
    Prepare data for watershed LSTM model.
    
    Args:
        args: Configuration arguments containing training parameters.
    
    Returns:
        tuple: Processed data for training, validation, and testing, along with normalization parameters.
    """
    hydrodata = data_preprocess(args)
    
    # Exclude release and doy for watershed LSTM
    hydrodata = hydrodata.drop(['release', 'doy'], axis=1)
    
    # Define date ranges for train, validation, and test sets
    training_start = pd.to_datetime(args.training_start) - datetime.timedelta(days=args.wrap_inflow)
    training_end = pd.to_datetime(args.training_end)
    validating_start = pd.to_datetime(args.validating_start) - datetime.timedelta(days=args.wrap_inflow + args.spinup)
    validating_end = pd.to_datetime(args.validating_end)
    testing_start = pd.to_datetime(args.testing_start) - datetime.timedelta(days=args.wrap_inflow + args.spinup)
    testing_end = pd.to_datetime(args.testing_end)

    # Split data into train, validation, and test sets
    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    val_set = hydrodata[hydrodata.index.isin(pd.date_range(validating_start, validating_end))]
    test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]

    # Apply min-max normalization
    train_min_lstm = train_set.min(axis=0).values
    train_max_lstm = train_set.max(axis=0).values
    train_set = (train_set - train_min_lstm) / (train_max_lstm - train_min_lstm)
    val_set = (val_set - train_min_lstm) / (train_max_lstm - train_min_lstm)
    test_set = (test_set - train_min_lstm) / (train_max_lstm - train_min_lstm)

    # Wrap data for LSTM input
    train_x_lstm, train_y_lstm = get_wrapped_data(train_set, 'inflow', args.wrap_inflow)
    val_x_lstm, val_y_lstm = get_wrapped_data(val_set, 'inflow', args.wrap_inflow)
    test_x_lstm, test_y_lstm = get_wrapped_data(test_set, 'inflow', args.wrap_inflow)

    # Perform timeline alignment checks
    assert (train_x_lstm[:10, -1, :] - train_set.iloc[args.wrap_inflow:args.wrap_inflow+10, :args.input_channels]).sum().sum() == 0
    assert (train_y_lstm[:10, -1] - train_set.iloc[args.wrap_inflow:args.wrap_inflow+10, -1]).sum() == 0
    assert train_set.index[args.wrap_inflow] == pd.to_datetime(args.training_start)
    assert (val_x_lstm[:10, -1, :] - val_set.iloc[args.wrap_inflow:args.wrap_inflow+10, :args.input_channels]).sum().sum() == 0
    assert (val_y_lstm[:10, -1] - val_set.iloc[args.wrap_inflow:args.wrap_inflow+10, -1]).sum() == 0
    assert val_set.index[args.wrap_inflow+args.spinup] == pd.to_datetime(args.validating_start)
    assert (test_x_lstm[:10, -1, :] - test_set.iloc[args.wrap_inflow:args.wrap_inflow+10, :args.input_channels]).sum().sum() == 0
    assert (test_y_lstm[:10, -1] - test_set.iloc[args.wrap_inflow:args.wrap_inflow+10, -1]).sum() == 0
    assert test_set.index[args.wrap_inflow+args.spinup] == pd.to_datetime(args.testing_start)
    
    # Convert numpy arrays to PyTorch tensors
    train_x_lstm, train_y_lstm = torch.FloatTensor(train_x_lstm), torch.FloatTensor(train_y_lstm)
    val_x_lstm, val_y_lstm = torch.FloatTensor(val_x_lstm), torch.FloatTensor(val_y_lstm)
    test_x_lstm, test_y_lstm = torch.FloatTensor(test_x_lstm), torch.FloatTensor(test_y_lstm)
    train_max_lstm, train_min_lstm = torch.FloatTensor(train_max_lstm), torch.FloatTensor(train_min_lstm)
    
    return train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, test_x_lstm, test_y_lstm, train_max_lstm, train_min_lstm


def traindata_reservoir(args):
    """
    Prepare data for reservoir LSTM model.
    
    Args:
        args: Configuration arguments containing training parameters.
    
    Returns:
        tuple: Processed data for training, validation, and testing, along with normalization parameters.
    """
    hydrodata = data_preprocess(args)
    
    # Exclude observed inflow for reservoir LSTM
    hydrodata = hydrodata.drop(['inflow'], axis=1)

    # Define date ranges for train, validation, and test sets
    training_start = pd.to_datetime(args.training_start)
    training_end = pd.to_datetime(args.training_end)
    validating_start = pd.to_datetime(args.validating_start) - datetime.timedelta(days=args.spinup)
    validating_end = pd.to_datetime(args.validating_end)
    testing_start = pd.to_datetime(args.testing_start) - datetime.timedelta(days=args.spinup)
    testing_end = pd.to_datetime(args.testing_end)
    
    # Split data into train, validation, and test sets
    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))].copy()
    val_set = hydrodata[hydrodata.index.isin(pd.date_range(validating_start, validating_end))].copy()
    test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))].copy()
    
    train_mean_phy = train_set.mean(axis=0).values
    train_std_phy = train_set.std(axis=0).values
    
    # Apply min-max normalization to specific features
    norm_feature = ['tmean', 'prcp']
    for feature in norm_feature:
        min_val = train_set[feature].min()
        max_val = train_set[feature].max()
        val_set[feature] = (val_set[feature] - min_val) / (max_val - min_val)
        test_set[feature] = (test_set[feature] - min_val) / (max_val - min_val)
        train_set[feature] = (train_set[feature] - min_val) / (max_val - min_val)

    # Generate wrapped data for LSTM input
    wrap_length = args.wrap_release
    train_x, train_y, val_x, val_y, test_x, test_y = generate_train_val_test_phy(train_set, val_set, test_set, wrap_length=wrap_length)
    
    # Convert numpy arrays to PyTorch tensors
    train_x_phy, train_y_phy = torch.FloatTensor(train_x), torch.FloatTensor(train_y)
    val_x_phy, val_y_phy = torch.FloatTensor(val_x), torch.FloatTensor(val_y)
    test_x_phy, test_y_phy = torch.FloatTensor(test_x), torch.FloatTensor(test_y)
    train_mean_phy, train_std_phy = torch.FloatTensor(train_mean_phy), torch.FloatTensor(train_std_phy)
    
    return train_x_phy, train_y_phy, val_x_phy, val_y_phy, test_x_phy, test_y_phy, train_mean_phy, train_std_phy


def plot_data_preprocess_phy(args,hydrodata):
    """
    Prepare data for plotting. Similar to traindata_reservoir, but with single batch processing.
    
    Args:
        args: Configuration arguments containing training parameters.
        hydrodata: Preprocessed hydrodata.
    
    Returns:
        tuple: Processed data for training, validation, and testing, along with original DataFrames.
    """
    
    # Define date ranges for train, validation, and test sets
    training_start = pd.to_datetime(args.training_start)
    training_end = pd.to_datetime(args.training_end)
    validating_start = pd.to_datetime(args.validating_start) - datetime.timedelta(days=args.spinup)
    validating_end = pd.to_datetime(args.validating_end)
    testing_start = pd.to_datetime(args.testing_start) - datetime.timedelta(days=args.spinup)
    testing_end = pd.to_datetime(args.testing_end)
    
    # Split data into train, validation, and test sets
    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))].copy()
    val_set = hydrodata[hydrodata.index.isin(pd.date_range(validating_start, validating_end))].copy()
    test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))].copy()

    # Apply min-max normalization to specific features
    norm_feature = ['prcp', 'tmean']
    for feature in norm_feature:
        min_val = train_set[feature].min()
        max_val = train_set[feature].max()
        val_set[feature] = (val_set[feature] - min_val) / (max_val - min_val)
        test_set[feature] = (test_set[feature] - min_val) / (max_val - min_val)
        train_set[feature] = (train_set[feature] - min_val) / (max_val - min_val)

    # Prepare data for LSTM input (excluding observed inflow and release)
    train_x_phy = np.expand_dims(train_set.iloc[:, :-2].values, axis=0)
    val_x_phy = np.expand_dims(val_set.iloc[:, :-2].values, axis=0)
    test_x_phy = np.expand_dims(test_set.iloc[:, :-2].values, axis=0)

    # Convert numpy arrays to PyTorch tensors
    train_x_phy = torch.FloatTensor(train_x_phy)
    val_x_phy = torch.FloatTensor(val_x_phy)
    test_x_phy = torch.FloatTensor(test_x_phy)

    return train_x_phy, val_x_phy, test_x_phy, train_set, val_set, test_set
