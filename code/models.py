# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Watershed_LSTM_Layer, Reservoir_LSTM_Layer_Jit, PolynomialDecayLR

class Base_Model(nn.Module):
    '''
    Base model which provides a template for all DL-Res models.
    This class sets up common parameters and the optimizer configuration.
    '''
    def __init__(self, warmup_updates, tot_updates, peak_lr, end_lr, power, weight_decay):
        super(Base_Model, self).__init__()
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.power = power
        self.weight_decay = weight_decay
    
    def configure_optimizers(self):
        '''
        Configures optimizers for the DL-Res model:
        1. optimizer_inflow: For training Watershed LSTM with constant learning rate
        2. optimizer_all: For training both Watershed and Reservoir LSTMs with polynomial scheduled learning rate
        
        Returns:
            tuple: ((optimizer_inflow, optimizer_all), (1, scheduler_all))
        '''
        # Optimizer for Watershed LSTM
        optimizer_inflow = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.watershed.parameters()),
            lr=5e-3,
            weight_decay=self.weight_decay
        )
        
        # Optimizer for the entire model
        optimizer_all = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.peak_lr,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler for optimizer_all
        scheduler_all = PolynomialDecayLR(
            optimizer_all, 
            warmup_updates=self.warmup_updates,
            tot_updates=self.tot_updates,
            lr=self.peak_lr,
            end_lr=self.end_lr,
            power=self.power
        )

        return (optimizer_inflow, optimizer_all), (1, scheduler_all)

class DL_Res_LSTM(Base_Model):
    '''
    DL-Res model:
    Consists of two main components:
    1. Watershed LSTM: Predicts inflow
    2. Reservoir LSTM: Takes predicted inflow and other inputs to make release predictions
    '''
    def __init__(self, input_channels, hidden_channels, output_channels, dropout, delay,
                 input_size, hidden_size, warmup_updates, tot_updates, peak_lr, end_lr, power, weight_decay,
                 jit, train_max_lstm, train_min_lstm):
        super(DL_Res_LSTM, self).__init__(warmup_updates, tot_updates, peak_lr, end_lr, power, weight_decay)
        
        # Register buffers for normalization
        self.register_buffer('train_max_lstm', train_max_lstm)
        self.register_buffer('train_min_lstm', train_min_lstm)

        # Initialize Watershed and Reservoir LSTM layers
        self.watershed = Watershed_LSTM_Layer(input_channels, hidden_channels, output_channels, dropout)
        self.reservoir = Reservoir_LSTM_Layer_Jit(input_size, hidden_size, dropout)

    def forward(self, inputs_lstm, inputs_phy_p):
        """
        Forward pass of the DL-Res LSTM model.
        
        Args:
            inputs_lstm (torch.Tensor): Input for Watershed LSTM, shape [Batch_size * Time_len, Feature]
            inputs_phy_p (torch.Tensor): Inputs for Reservoir LSTM, shape [Batch_size, Time_len, Feature]
        
        Returns:
            tuple: (x, all_release, analyze)
                x: Output from Watershed LSTM
                all_release: Final output from Reservoir LSTM
                analyze: Additional analysis data from Reservoir LSTM
        """
        # Process inputs through Watershed LSTM
        x = self.watershed(inputs_lstm)
        
        # Denormalize predicted inflow
        inflow = x * (self.train_max_lstm[-1] - self.train_min_lstm[-1]) + self.train_min_lstm[-1]
        inflow = F.relu(inflow)
        
        # Reshape inflow to match inputs_phy_p dimensions
        lstm_len = inputs_phy_p.shape[1]
        inflow_batch = torch.empty(inputs_phy_p.shape[0], inputs_phy_p.shape[1], inflow.shape[1]).to(device=inputs_phy_p.device)
        for i in range(inflow_batch.shape[0]):
            inflow_batch[i] = inflow[i*365:i*365+lstm_len]
        
        # Concatenate reshaped inflow with other predictors
        inputs_phy = torch.cat([inputs_phy_p, inflow_batch], dim=-1)
        
        # Process combined inputs through Reservoir LSTM
        all_release, analyze = self.reservoir(inputs_phy)
        
        return x, all_release, analyze