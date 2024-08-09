# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Tuple
import pandas as pd

class Watershed_LSTM_Layer(nn.Module):
    """
    Watershed LSTM Layer: A standard LSTM layer for watershed modeling.
    """
    def __init__(self, input_channels, hidden_channels, output_channels, dropout):
        super(Watershed_LSTM_Layer, self).__init__()
        
        # LSTM layer
        self.lstm1 = nn.LSTM(input_channels, hidden_channels, batch_first=True, num_layers=1)
        self._initialize_lstm_parameters()
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully connected layer
        self.readout = nn.Linear(hidden_channels, output_channels)
        self._initialize_readout_parameters()

    def _initialize_lstm_parameters(self):
        for p in self.lstm1.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p)
            else:
                nn.init.normal_(p, 0, 0.1)

    def _initialize_readout_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for p in self.readout.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain)
            else:
                nn.init.normal_(p, 0, 0.1)

    def forward(self, inputs):
        """
        Perform forward propagation.
        """
        x, _ = self.lstm1(inputs)
        x = x[:, -1, :]  # Take the last time step
        x = self.dropout(x)
        x = self.readout(x)
        x = F.gelu(x)
        return x

class Reservoir_LSTM_Layer_Jit(jit.ScriptModule):
    """
    Reservoir LSTM Layer: Implementation of the reservoir LSTM layer using PyTorch JIT.
    
    This LSTM layer is implemented with PyTorch JIT, which is a powerful acceleration tool
    to speed up custom PyTorch models.
    
    Reference: https://residentmario.github.io/pytorch-training-performance-guide/jit.html
    
    Attributes:
        s2: reservoir storage (mm)
        inflow: reservoir inflow (mm/d)
        release: reservoir release (mm/d)
        reser_minstg: reservoir minimum storage (mm)
        reser_maxstgF: reservoir maximum storage during flood period (mm)
        reser_minstgNF: reservoir maximum storage during non-flood period (mm)
        flood_sday: start day of year for flood period 
        flood_eday: end day of year for flood period 
        reser_minre: reservoir minimum release (mm)
        reser_maxre: reservoir maximum release (mm)
        reser_normflow: value to normalize reservoir inflow and outflow (about 90% quantile of inflow) (mm)
    """
    def __init__(self, input_size, hidden_size, dropout):
        super(Reservoir_LSTM_Layer_Jit, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM cell, fully connected layer, and dropout
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout)

        self._initialize_parameters()
        self._initialize_reservoir_properties()

    def _initialize_parameters(self):
        # Initialize LSTM cell
        for p in self.lstm_cell.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, 0, 0.1)
            else:
                nn.init.zeros_(p)
        
        # Initialize fully connected layer
        for p in self.readout.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, 0, 0.5)
            else:
                nn.init.zeros_(p)

    def _initialize_reservoir_properties(self):
        # Define reservoir properties encoded in the reservoir LSTM layer
        self.register_buffer('reser_minstg', torch.tensor(9.1*10**8/35200/10**3))
        self.register_buffer('reser_maxstgF', torch.tensor(22.25*10**8/35200/10**3))
        self.register_buffer('reser_maxstgNF', torch.tensor(25.85*10**8/35200/10**3))
        self.register_buffer('flood_sday', torch.tensor(pd.to_datetime('2001-5-1').dayofyear))
        self.register_buffer('flood_eday', torch.tensor(pd.to_datetime('2001-9-30').dayofyear))
        self.register_buffer('reser_minre', torch.tensor(0*3600*24/35200/10**3))
        self.register_buffer('reser_maxre', torch.tensor(20000*3600*24/35200/10**3))
        self.register_buffer('reser_normflow', torch.tensor(4000*3600*24/35200/10**3))
    
    def heaviside(self, x):
        """
        A smooth approximation of Heaviside step function
            if x < 0: heaviside(x) ~= 0
            if x > 0: heaviside(x) ~= 1
        """
        return (torch.tanh(5*x) + 1) / 2

    def reservoir_bucket(self, s2, inflow, inflow_pf, p, t, cosdoy, hidden_state):
        """
        Reservoir bucket model for a single time step.
        
        Args:
            s2, inflow, inflow_pf, p, t, cosdoy: shape is [batch_size, feature]
        
        Note: Inflow and storage are normalized here for computation.
        """
        s2 = s2 / self.reser_maxstgNF  # normalize reservoir storage
        inflow = inflow / self.reser_normflow  # normalize inflow
        
        x = torch.cat([s2, inflow, p, t, cosdoy], dim=1)
        new_hidden, new_cell = self.lstm_cell(x, (hidden_state[0], hidden_state[1]))
        new_re = self.readout(self.dropout(new_hidden))
        new_re = new_re * self.reser_normflow  # unnormalize reservoir release
        
        # Constrain reservoir release between maximum and minimum
        new_re = self.reser_maxre * self.heaviside(new_re - self.reser_maxre) + \
                 self.reser_minre * self.heaviside(self.reser_minre - new_re) + \
                 new_re * self.heaviside(self.reser_maxre - new_re) * self.heaviside(new_re - self.reser_minre)
        
        new_state = torch.stack([new_hidden, new_cell], dim=0)
        return new_re, new_state
    
    def rnncell(self, inputs, states, hidden_state):
        """
        RNN cell for a single time step.
        
        Args:
            inputs: shape is [batch_size, feature]
            
        Note: Inflow and storage here are unnormalized to perform water balance.
        """
        p=inputs[:,0:1]
        t=inputs[:,1:2]
        doy=inputs[:,2:3]
        cosdoy=inputs[:,3:4]
        inflow_pred=inputs[:,4:5]
        inflow_pf=inputs[:,5:7]
        s2=states[:,0:1]

        # Make release prediction
        new_re, new_hidden = self.reservoir_bucket(s2, inflow_pred, inflow_pf, p, t, cosdoy, hidden_state)
        
        # Get the maximum reservoir storage according to day of year
        reser_maxstg = self.heaviside(doy - self.flood_eday) * self.reser_maxstgNF + \
                       self.heaviside(self.flood_sday - doy) * self.reser_maxstgNF + \
                       self.heaviside(self.flood_eday - doy) * self.heaviside(doy - self.flood_sday) * self.reser_maxstgF
        
        # Perform water balance calculation
        new_s2 = s2 - new_re + inflow_pred
        
        # Check the storage range
        cstred_s2 = self.heaviside(new_s2 - reser_maxstg) * reser_maxstg + \
                    self.heaviside(self.reser_minstg - new_s2) * self.reser_minstg + \
                    self.heaviside(reser_maxstg - new_s2) * self.heaviside(new_s2 - self.reser_minstg) * new_s2
        
        # Check the water balance again 
        release = s2 + inflow_pred - cstred_s2
        
        # Collect release and storage
        new_state = torch.cat([cstred_s2, release], dim=1)
        return new_state, new_hidden
    
    @jit.script_method
    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Reservoir LSTM Layer.
        
        Args:
            inputs: shape is [batch_size, time_steps, features]
        
        Returns:
            all_release: Reservoir release for all time steps
            s2: Final reservoir storage
        """
        new_inputs = inputs.unbind(1)
        
        init_states = torch.ones((inputs.shape[0], 2), device=inputs.device) * 20
        hidden_state = torch.zeros((2, inputs.shape[0], self.hidden_size), device=inputs.device)
        temp_states = init_states
        all_states = torch.jit.annotate(List[Tensor], [])
        
        for i in range(len(new_inputs)):
            temp_states, hidden_state = self.rnncell(new_inputs[i], temp_states, hidden_state)
            all_states += [temp_states]
        
        all_states = torch.stack(all_states, dim=1)
        
        s2 = all_states[:, :, 0:1]
        all_release = all_states[:, :, 1:2]

        return all_release, s2

class PolynomialDecayLR(_LRScheduler):
    """
    Polynomial Decay Learning Rate Scheduler.
    
    This scheduler is modified from https://github.com/diggerdu/Graphormer
    """

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr,
                 power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (self.tot_updates - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr
        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        raise NotImplementedError("Closed form LR not implemented for PolynomialDecayLR")