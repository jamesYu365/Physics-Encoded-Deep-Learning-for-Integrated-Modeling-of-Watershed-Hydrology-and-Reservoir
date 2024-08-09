# -*- coding: utf-8 -*-

from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import copy
import time

from utils import NSE_loss_BF, NSE_loss_BTF, save_model
from plot_utils import plot_result

def test_model(args, model, test_x_lstm, test_y_lstm, test_x_phy, test_y_phy, device):
    """
    Run model on test set and calculate NSE scores.

    Args:
        args: Argument object containing model parameters
        model: Trained model
        test_x_lstm, test_y_lstm: Watershed LSTM input and output test data
        test_x_phy, test_y_phy: Reservoir LSTM input and output test data
        device: Computation device (CPU/GPU)

    Returns:
        None, but prints final test NSE scores
    """
    model.eval()
    inputs_lstm = Variable(test_x_lstm.to(device=device))
    labels_lstm = Variable(test_y_lstm.to(device=device))
    inputs_phy = Variable(test_x_phy.to(device=device))
    labels_phy = Variable(test_y_phy.to(device=device))
    
    outputs_lstm, outputs_phy, _ = model(inputs_lstm, inputs_phy)
    
    inflow_nse = 1 - NSE_loss_BF(outputs_lstm, labels_lstm, args.spinup)
    release_nse = 1 - NSE_loss_BTF(outputs_phy, labels_phy, args.spinup)
    
    print(f'Final test inflow NSE: {inflow_nse.item():.4f}')
    print(f'Final test release NSE: {release_nse.item():.4f}')

def val_model(args, epoch, model, writer, val_x_lstm, val_y_lstm, val_x_phy, val_y_phy, device):
    """
    Run model on validation set and log results.

    Args:
        args: Argument object containing model parameters
        epoch: Current epoch number
        model: Current model
        writer: TensorBoard writer object
        val_x_lstm, val_y_lstm: Watershed LSTM input and output validation data
        val_x_phy, val_y_phy: Reservoir LSTM input and output validation data
        device: Computation device (CPU/GPU)

    Returns:
        tuple: (inflow_nse, release_nse) validation NSE scores
    """
    model.eval()
    inputs_lstm = Variable(val_x_lstm.to(device=device))
    labels_lstm = Variable(val_y_lstm.to(device=device))
    inputs_phy = Variable(val_x_phy.to(device=device))
    labels_phy = Variable(val_y_phy.to(device=device))
    
    outputs_lstm, outputs_phy, _ = model(inputs_lstm, inputs_phy)
    
    # Calculate NSE scores for validation set and log them
    inflow_nse = 1 - NSE_loss_BF(outputs_lstm, labels_lstm, args.spinup)
    release_nse = 1 - NSE_loss_BTF(outputs_phy, labels_phy, args.spinup)
    writer.add_scalar('loss/val_inflow_nse', inflow_nse.item(), epoch)
    writer.add_scalar('loss/val_release_nse', release_nse.item(), epoch)
    
    return inflow_nse.item(), release_nse.item()

def train_epoch(args, epoch, model, optimizer, lr_scheduler, writer, stime,
                train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm,
                train_x_phy, train_y_phy, val_x_phy, val_y_phy, device,
                val_inflow_nse, val_release_nse, path_t):
    """
    Train the model for one epoch.

    Args:
        args: Argument object containing model parameters
        epoch: Current epoch number
        model: Current model
        optimizer: Optimizer object
        lr_scheduler: Learning rate scheduler
        writer: TensorBoard writer object
        stime: Start time of training
        train_x_lstm, train_y_lstm: Watershed LSTM input and output training data
        val_x_lstm, val_y_lstm: Watershed LSTM input and output validation data
        train_x_phy, train_y_phy: Reservoir LSTM input and output training data
        val_x_phy, val_y_phy: Reservoir LSTM input and output validation data
        device: Computation device (CPU/GPU)
        val_inflow_nse, val_release_nse: Previous validation NSE scores
        path_t: Path to save results

    Returns:
        tuple: (train_nseloss, val_inflow_nse, val_release_nse)
    """
    model.train()
    train_inflowloss, train_releaseloss, train_nseloss = [], [], []
    
    # Run all samples in batches
    for i in range(0, train_x_phy.shape[0], args.batch_size_phy):
        lstm_len = (args.batch_size_phy - 1) * 365 + args.wrap_release
        inputs_lstm = Variable(train_x_lstm[i*365:i*365+lstm_len].to(device=device))
        labels_lstm = Variable(train_y_lstm[i*365:i*365+lstm_len].to(device=device))
        inputs_phy = Variable(train_x_phy[i:i+args.batch_size_phy].to(device=device))
        labels_phy = Variable(train_y_phy[i:i+args.batch_size_phy].to(device=device))
        
        outputs_lstm, outputs_phy, analyze = model(inputs_lstm, inputs_phy)
        
        # Calculate NSE loss for inflow and release
        loss_inflow = NSE_loss_BF(outputs_lstm, labels_lstm)
        loss_release = NSE_loss_BTF(outputs_phy, labels_phy, args.spinup)
        
        # Training strategy: watershed LSTM for first 100 epochs, then both LSTMs
        if epoch < 100:
            loss_nse = loss_inflow
            optimizer[0].zero_grad()
            loss_nse.backward()
            optimizer[0].step()
        else:
            loss_nse = loss_inflow + loss_release
            optimizer[1].zero_grad()
            loss_nse.backward()
            optimizer[1].step()

        train_inflowloss.append(loss_inflow.item())
        train_releaseloss.append(loss_release.item())
        train_nseloss.append(loss_nse.item())
    
    # Update learning rate and log it
    if epoch < 100:
        writer.add_scalar('learning_rate', optimizer[0].state_dict()['param_groups'][0]['lr'], epoch)
    else:
        lr_scheduler[1].step()
        writer.add_scalar('learning_rate', optimizer[1].state_dict()['param_groups'][0]['lr'], epoch)

    # Log average losses
    train_inflowloss = np.mean(train_inflowloss)
    train_releaseloss = np.mean(train_releaseloss)
    train_nseloss = np.mean(train_nseloss)
    writer.add_scalar('loss/inflowloss', train_inflowloss, epoch)
    writer.add_scalar('loss/releaseloss', train_releaseloss, epoch)
    writer.add_scalar('loss/train_nseloss', train_nseloss, epoch)
    
    # Print progress every 10 epochs
    if epoch % 10 == 0:
        etime = time.time()
        val_inflow_nse, val_release_nse = val_model(args, epoch, model, writer, val_x_lstm, val_y_lstm, val_x_phy, val_y_phy, device)
        print(f'Epoch: {epoch:04d}, '
              f'Train NSE Loss: {train_nseloss:.4f}, '
              f'Val Inflow NSE: {val_inflow_nse:.4f}, '
              f'Val Release NSE: {val_release_nse:.4f}, '
              f'Time: {etime-stime:.1f}s')
    
    return train_nseloss, val_inflow_nse, val_release_nse

def run_exp(args, num_epoch, model, optimizer, lr_scheduler,
            train_x_phy, train_y_phy, val_x_phy, val_y_phy, test_x_phy, test_y_phy, 
            train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, test_x_lstm, test_y_lstm, 
            device, exp, i):
    """
    Run a single experiment (one repeat).

    Args:
        args: Argument object containing model parameters
        num_epoch: Number of epochs to train
        model: Model to train
        optimizer: Optimizer object
        lr_scheduler: Learning rate scheduler
        train_x_phy, train_y_phy: Reservoir LSTM input and output training data
        val_x_phy, val_y_phy: Reservoir LSTM input and output validation data
        test_x_phy, test_y_phy: Reservoir LSTM input and output test data
        train_x_lstm, train_y_lstm: Watershed LSTM input and output training data
        val_x_lstm, val_y_lstm: Watershed LSTM input and output validation data
        test_x_lstm, test_y_lstm: Watershed LSTM input and output test data
        device: Computation device (CPU/GPU)
        exp: Experiment name
        i: Repeat number

    Returns:
        int: Flag indicating if training completed successfully (1) or not (0)
    """
    # Initialize early stopping parameters
    stime = time.time()
    train_nselosses = []
    val_inflow_nse, val_release_nse = -np.inf, -np.inf
    bad_count, best_nse, best_epoch = 0, -np.inf, 0
    best_model, flag = None, 1
    
    # Set up TensorBoard writer
    path_t = f'./result/{exp}repeat{i}/'
    writer = SummaryWriter(path_t)
    
    for epoch in range(num_epoch):
        # Run one epoch
        train_nseloss, val_inflow_nse, val_release_nse = train_epoch(
            args, epoch, model, optimizer, lr_scheduler, writer, stime,
            train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm,
            train_x_phy, train_y_phy, val_x_phy, val_y_phy, device,
            val_inflow_nse, val_release_nse, path_t
        )
        
        # Save current epoch results
        train_nselosses.append(train_nseloss)
        epoch_model = copy.deepcopy(model.state_dict())
        
        # Check for NaN loss
        if np.isnan(train_nseloss):
            flag = 0
            save_model(args, best_epoch, epoch, best_model, epoch_model, model, path_t)
            test_model(args, model, test_x_lstm, test_y_lstm, test_x_phy, test_y_phy, device)
            plot_result(args, model, path_t)
            return flag
        
        # Check for potential best model
        if train_nseloss < 1 and val_inflow_nse > 0.7 and epoch > 250:
            if val_release_nse > best_nse:
                best_nse, best_epoch, best_model, bad_count = val_release_nse, epoch, epoch_model, 0
            else:
                bad_count += 1
        else:
            bad_count = 0

        # Early stopping check
        if bad_count == args.patience:
            print(f'Early stopping! Best NSE: {best_nse:.4f}')
            break
    
    # Save final model, run test, and plot results
    save_model(args, best_epoch, epoch, best_model, epoch_model, model, path_t)
    test_model(args, model, test_x_lstm, test_y_lstm, test_x_phy, test_y_phy, device)
    plot_result(args, model, path_t)
    return flag