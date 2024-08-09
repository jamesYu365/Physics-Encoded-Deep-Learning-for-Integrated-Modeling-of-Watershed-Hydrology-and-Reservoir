# -*- coding: utf-8 -*-

import argparse
import torch

from models import DL_Res_LSTM
from preparedataset import traindata_reservoir, traindata_watershed
from utils import set_seed
from train_utils import run_exp

#%% Configuration

def setup_argparse():
    """Set up command-line argument parser with all necessary parameters."""
    parser = argparse.ArgumentParser(description='DL-Res-LSTM model configuration')
    
    # File paths
    parser.add_argument("--datapath", type=str, default='../../data/', help="Data path")
    parser.add_argument("--exp", type=str, default='exp0/', help="Configuration result path")
    
    # Data settings
    parser.add_argument("--training_start", type=str, default='1992-01-01', help="Training set start time")
    parser.add_argument("--training_end", type=str, default='2005-12-31', help="Training set end time")
    parser.add_argument("--validating_start", type=str, default='2006-01-01', help="Validation set start time")
    parser.add_argument("--validating_end", type=str, default='2008-12-31', help="Validation set end time")
    parser.add_argument("--testing_start", type=str, default='2009-01-01', help="Testing set start time")
    parser.add_argument("--testing_end", type=str, default='2014-01-01', help="Testing set end time")
    parser.add_argument("--delay", type=int, default=0, help="Inflow delay time")
    parser.add_argument("--wrap_inflow", type=int, default=180, help="Look back size")
    parser.add_argument("--wrap_release", type=int, default=365*4, help="Time length for one batch size")
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='DL-Res(AK)', help='Model name')
    parser.add_argument('--jit', type=bool, default=True, help='Whether to use jit Reservoir LSTM')
    parser.add_argument('--input_size', type=int, default=5, help='Input channel size of watershed LSTM')
    parser.add_argument('--hidden_size', type=int, default=3, help='Hidden channel size of watershed LSTM')
    parser.add_argument('--input_channels', type=int, default=3, help='Input channel size of reservoir LSTM')
    parser.add_argument('--hidden_channels', type=int, default=3, help='Hidden channel size of reservoir LSTM')
    parser.add_argument('--output_channels', type=int, default=1, help='Output channel size of reservoir LSTM')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')
    
    # Training settings
    parser.add_argument('--seed', type=int, default=42, help='Set environment seed')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use CUDA')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number')
    parser.add_argument('--spinup', type=int, default=365, help='Set warmup periods as one year')
    parser.add_argument('--tot_evals', type=int, default=10, help='Number of repeat experiments')
    parser.add_argument('--batch_size_phy', type=int, default=5, help='Batch size')
    parser.add_argument('--patience', type=int, default=100, help='Early stop patience')
    parser.add_argument('--warmup_updates', type=int, default=10, help='Learning rate warmup updates')
    parser.add_argument('--tot_updates', type=int, default=400, help='Total updates')
    parser.add_argument('--peak_lr', type=float, default=1e-2, help='Peak learning rate')
    parser.add_argument('--end_lr', type=float, default=1e-3, help='End learning rate')
    parser.add_argument('--power', type=int, default=1, help='Learning rate decay power')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for Adam')
    
    return parser.parse_args()

#%% Main execution

def main():
    """Main function to run the DL-Res-LSTM model."""
    
    # Parse command-line arguments
    args = setup_argparse()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load and preprocess data
    train_x_phy, train_y_phy, val_x_phy, val_y_phy, test_x_phy, test_y_phy, train_mean_phy, train_std_phy = traindata_reservoir(args)
    train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, test_x_lstm, test_y_lstm, train_max_lstm, train_min_lstm = traindata_watershed(args)
    
    # Set up CUDA if available
    device = torch.device(f'cuda:{args.cuda_device}') if args.cuda else torch.device('cpu')
    
    # Run the training process for specified number of evaluations
    for i in range(args.tot_evals):
        print("-"*10, f"Experiment {i+1}/{args.tot_evals}", "-"*10)
        
        # Initialize model
        model = DL_Res_LSTM(
            args.input_channels, args.hidden_channels, args.output_channels, args.dropout,
            args.delay, args.input_size, args.hidden_size, args.warmup_updates, args.tot_updates,
            args.peak_lr, args.end_lr, args.power, args.weight_decay,
            args.jit, train_max_lstm, train_min_lstm
        )
        
        # Configure optimizer and learning rate scheduler
        optimizer, lr_scheduler = model.configure_optimizers()
        model = model.to(device=device)
        
        # Print model information
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('*'*60)
        print(f'{args.exp} model {i+1} has {num_parameters} learnable parameters')
        print('*'*60)
        
        # Train the model
        exit_flag = run_exp(
            args, args.tot_updates, model, optimizer, lr_scheduler,
            train_x_phy, train_y_phy, val_x_phy, val_y_phy, test_x_phy, test_y_phy,
            train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, test_x_lstm, test_y_lstm,
            device, args.exp, i
        )
        
        # Check for training termination
        if exit_flag == 0:
            print('Iteration terminated due to NaN')
        else:
            print('Iteration terminated with no error')

if __name__ == "__main__":
    main()
