#!/usr/bin/env python
# Stock Prediction Model using Transformer Architecture

import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import from our modules
from utils import load_config, setup_logging, set_seed, get_device, create_output_dirs
from data_utils import load_and_prepare_features, prepare_data_for_pytorch
from model_utils import create_model

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Stock Prediction Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "evaluate"], default="train", 
                      help="Mode to run: train, predict, or evaluate")
    parser.add_argument("--tickers", type=str, nargs="+", help="Override tickers in config")
    parser.add_argument("--start_date", type=str, help="Override start date in config")
    parser.add_argument("--end_date", type=str, help="Override end date in config")
    parser.add_argument("--output_dir", type=str, help="Override output directory in config")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint to load for evaluation/prediction")
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, target_indices):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_l1_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for data in progress_bar:
        input_features, ticker_ids, targets = data
        input_features = input_features.to(device)
        ticker_ids = ticker_ids.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predicted_full_features, latent = model(input_features, ticker_ids)
        
        # We're interested in the prediction for the last timestep
        last_timestep_predictions = predicted_full_features[:, -1, :]
        
        # We need to extract only the target features from the full feature predictions
        predicted_targets = last_timestep_predictions[:, target_indices]
        
        # Reconstruction Loss
        reconstruction_loss = criterion(predicted_targets, targets)
        
        # Sparsity Loss
        l1_loss = model.l1_activity_loss(latent)
        
        # Total Loss
        loss = reconstruction_loss + l1_loss
        
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_recon_loss += reconstruction_loss.item()
        total_l1_loss += l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss
        
        # Update progress bar
        progress_bar.set_postfix(loss=loss.item(), recon=reconstruction_loss.item(), l1=l1_loss.item())
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    
    return avg_loss, avg_recon_loss, avg_l1_loss


def evaluate_model(model, dataloader, criterion, target_indices, target_feature_names, ticker_map_inv, device):
    """Evaluate the model on validation/test data."""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_l1_loss = 0.0
    
    all_targets = []
    all_preds = []
    
    # Additional metrics
    metrics = {}
    
    # Track per-ticker metrics
    ticker_metrics = {}
    
    # Direction prediction counters
    correct_direction = 0
    total_direction_preds = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Unpack batch
            x, ticker_ids, y = batch
            x, y = x.to(device), y.to(device)
            ticker_ids = ticker_ids.to(device)
            
            # Forward pass
            outputs, latent = model(x, ticker_ids)
            
            # Get predictions for the last time step
            outputs_last = outputs[:, -1, :]
            
            # Extract target features
            target_outputs = outputs_last[:, target_indices]
            
            # Calculate losses
            recon_loss = criterion(target_outputs, y)
            l1_loss = model.l1_activity_loss(latent)
            loss = recon_loss + l1_loss
            
            total_loss += loss.item() * x.size(0)
            total_recon_loss += recon_loss.item() * x.size(0)
            total_l1_loss += l1_loss.item() * x.size(0)
            
            # Collect predictions and targets for metrics
            predictions = target_outputs.cpu().numpy()
            targets = y.cpu().numpy()
            
            all_targets.append(targets)
            all_preds.append(predictions)
            
            # Calculate per-ticker metrics
            for i, ticker_id in enumerate(ticker_ids.cpu().numpy()):
                ticker_id = int(ticker_id)
                ticker_symbol = ticker_map_inv.get(ticker_id, f"ID_{ticker_id}")
                
                if ticker_symbol not in ticker_metrics:
                    ticker_metrics[ticker_symbol] = {
                        'targets': [],
                        'preds': [],
                    }
                
                ticker_metrics[ticker_symbol]['targets'].append(targets[i:i+1])
                ticker_metrics[ticker_symbol]['preds'].append(predictions[i:i+1])
            
            # Calculate direction prediction accuracy for Close price
            if 'Close' in target_feature_names:
                close_idx = target_feature_names.index('Close')
                
                # Get previous close (from input sequence)
                prev_close = x[:, -1, target_indices[close_idx]].cpu().numpy()
                
                # Get actual and predicted close
                actual_close = targets[:, close_idx]
                predicted_close = predictions[:, close_idx]
                
                # Calculate directions
                actual_direction = actual_close > prev_close
                predicted_direction = predicted_close > prev_close
                
                # Count correct predictions
                correct = np.sum(actual_direction == predicted_direction)
                correct_direction += correct
                total_direction_preds += len(actual_direction)
    
    # Concatenate batch data
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = total_recon_loss / len(dataloader.dataset)
    avg_l1_loss = total_l1_loss / len(dataloader.dataset)
    
    # Import metric functions here to avoid circular imports
    from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                               explained_variance_score, median_absolute_error, max_error)
    
    # Calculate overall metrics
    metrics['mse'] = mean_squared_error(all_targets, all_preds)
    metrics['mae'] = mean_absolute_error(all_targets, all_preds)
    metrics['r2'] = r2_score(all_targets, all_preds)
    metrics['explained_variance'] = explained_variance_score(all_targets, all_preds)
    
    # Calculate MAPE with handling for zeros
    non_zero_mask = np.abs(all_targets) > 1e-10
    if np.any(non_zero_mask):
        metrics['mape'] = np.mean(np.abs((all_targets[non_zero_mask] - all_preds[non_zero_mask]) / all_targets[non_zero_mask])) * 100
    else:
        metrics['mape'] = float('nan')
    
    metrics['median_abs_error'] = median_absolute_error(all_targets, all_preds)
    
    # Calculate max_error per target feature
    for i, feature_name in enumerate(target_feature_names):
        feature_targets = all_targets[:, i]
        feature_preds = all_preds[:, i]
        try:
            metrics[f'max_error_{feature_name}'] = max_error(feature_targets, feature_preds)
        except ValueError as e:
            logging.warning(f"Could not calculate max_error for {feature_name}: {e}")
            metrics[f'max_error_{feature_name}'] = float('nan')
    
    # Direction accuracy
    if total_direction_preds > 0:
        metrics['direction_accuracy'] = correct_direction / total_direction_preds
    
    return avg_loss, avg_recon_loss, avg_l1_loss, metrics


def train_model(config, data_dict, device):
    """Train the model with early stopping and logging."""
    logging.info("Starting model training...")
    
    # Create model
    model = create_model(
        config=config,
        num_input_features=len(data_dict['input_feature_names']),
        num_tickers=data_dict['num_tickers']
    )
    model = model.to(device)
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Setup optimizer
    optimizer_name = config['training']['optimizer']
    lr = config['training']['learning_rate']
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        logging.warning(f"Unknown optimizer '{optimizer_name}'. Using Adam.")
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Setup learning rate scheduler if enabled
    scheduler = None
    if config['training']['lr_scheduler']['enabled']:
        scheduler_config = config['training']['lr_scheduler']
        if scheduler_config['type'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=scheduler_config['step_size'], 
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=scheduler_config['gamma'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
            )
        elif scheduler_config['type'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs'],
                eta_min=scheduler_config['min_lr']
            )
    
    # Setup gradient clipping if enabled
    clip_grad = False
    if config['training']['gradient_clipping']['enabled']:
        clip_grad = True
        clip_value = config['training']['gradient_clipping']['max_norm']
    
    # Setup early stopping
    early_stopping = config['training']['early_stopping']['enabled']
    patience = config['training']['early_stopping']['patience']
    delta = config['training']['early_stopping']['delta']
    best_val_loss = float('inf')
    no_improve_counter = 0
    best_model_state = None
    
    # Setup TensorBoard
    dirs = create_output_dirs(config)
    writer = SummaryWriter(dirs['tensorboard'])
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_l1_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_l1_loss': [],
        'val_metrics': []
    }
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        # Training phase
        train_loss, train_recon_loss, train_l1_loss = train_epoch(
            model=model,
            dataloader=data_dict['train_loader'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            target_indices=data_dict['target_indices']
        )
        
        # Validation phase
        val_loss, val_recon_loss, val_l1_loss, val_metrics = evaluate_model(
            model=model,
            dataloader=data_dict['test_loader'],
            criterion=criterion,
            target_indices=data_dict['target_indices'],
            target_feature_names=data_dict['target_feature_names'],
            ticker_map_inv=data_dict['ticker_map_inv'],
            device=device
        )
        
        # If using ReduceLROnPlateau, step with validation loss
        if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler:
            scheduler.step()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_recon_loss'].append(train_recon_loss)
        history['train_l1_loss'].append(train_l1_loss)
        history['val_loss'].append(val_loss)
        history['val_recon_loss'].append(val_recon_loss)
        history['val_l1_loss'].append(val_l1_loss)
        history['val_metrics'].append(val_metrics)
        
        # Log to console
        logging.info(f"Epoch {epoch+1}/{config['training']['epochs']}:")
        logging.info(f"  Train Loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, L1: {train_l1_loss:.6f})")
        logging.info(f"  Val Loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f}, L1: {val_l1_loss:.6f})")
        logging.info(f"  Val MSE: {val_metrics['mse']:.6f}, Val MAE: {val_metrics['mae']:.6f}")
        
        if 'direction_accuracy' in val_metrics:
            logging.info(f"  Direction Accuracy: {val_metrics['direction_accuracy']*100:.2f}%")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train/Total', train_loss, epoch)
        writer.add_scalar('Loss/Train/Reconstruction', train_recon_loss, epoch)
        writer.add_scalar('Loss/Train/L1', train_l1_loss, epoch)
        writer.add_scalar('Loss/Val/Total', val_loss, epoch)
        writer.add_scalar('Loss/Val/Reconstruction', val_recon_loss, epoch)
        writer.add_scalar('Loss/Val/L1', val_l1_loss, epoch)
        writer.add_scalar('Metrics/Val/MSE', val_metrics['mse'], epoch)
        writer.add_scalar('Metrics/Val/MAE', val_metrics['mae'], epoch)
        writer.add_scalar('Metrics/Val/R2', val_metrics['r2'], epoch)
        
        if 'direction_accuracy' in val_metrics:
            writer.add_scalar('Metrics/Val/DirectionAccuracy', val_metrics['direction_accuracy'], epoch)
        
        # Check for improvement (early stopping)
        if early_stopping:
            if val_loss < best_val_loss - delta:
                logging.info(f"  Validation loss improved from {best_val_loss:.6f} to {val_loss:.6f}")
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                logging.info(f"  No improvement in validation loss for {no_improve_counter} epochs")
                
                if no_improve_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
    # Load best model
    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    # Save model
    model_path = os.path.join(dirs['base'], config['paths']['model_save_file'])
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'history': history,
        'config': config
    }, model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    return model, history


def predict_future(model, ticker_data, ticker_id, config, data_dict, device):
    """Generate future predictions for a ticker."""
    future_days = config['prediction']['future_days']
    window_size = config['data']['window_size']
    ticker_to_id = {v: k for k, v in data_dict['ticker_map_inv'].items()}
    ticker_symbol = data_dict['ticker_map_inv'].get(ticker_id, f"ID_{ticker_id}")
    
    logging.info(f"Generating {future_days} days of future predictions for {ticker_symbol} (ID: {ticker_id})...")
    
    model.eval()
    
    # Make a copy of the input data
    data = ticker_data.copy()
    
    # Get only the input features that were used during training
    input_features = data_dict['input_feature_names']
    data_for_scaling = data[input_features].copy()
    
    # Get the original values for scaling
    orig_values = data_for_scaling.values
    
    # Find the scaler for this ticker
    scaler = data_dict['scalers'].get(ticker_id)
    if scaler is None:
        logging.error(f"No scaler found for ticker ID {ticker_id}")
        return None
    
    # Scale the input data
    scaled_data = scaler.transform(orig_values)
    
    # Take the last window_size records for prediction
    last_input_window = scaled_data[-window_size:]
    
    # Prepare for prediction
    all_preds = []
    
    # Generate future predictions one step at a time
    for i in tqdm(range(future_days), desc=f"Predicting {ticker_symbol}"):
        # Prepare the input
        if i == 0:
            # For the first prediction, use the last window from historical data
            current_window = torch.tensor(last_input_window, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            # For subsequent predictions, shift the window forward and add the last prediction
            current_window = torch.cat([current_window[:, 1:, :], next_pred.unsqueeze(1)], dim=1)
        
        # Make prediction
        with torch.no_grad():
            next_pred_full, _ = model(current_window, torch.tensor([ticker_id], dtype=torch.long).to(device))
            next_pred = next_pred_full[:, -1, :]  # Take the last timestep prediction
        
        # Store the prediction
        all_preds.append(next_pred.cpu().numpy()[0])
    
    # Convert predictions to numpy array
    all_preds = np.array(all_preds)
    
    # Inverse transform predictions to original scale
    unscaled_preds = scaler.inverse_transform(all_preds)
    
    # Create a DataFrame for the predictions with the same columns as input features
    date_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')
    future_df = pd.DataFrame(unscaled_preds, index=date_index, columns=input_features)
    
    # Add indicator calculations for better visualization
    full_df = pd.concat([data, future_df])
    
    # Calculate technical indicators
    result = full_df.copy()
    
    # Simple Moving Averages
    result['SMA5'] = full_df['Close'].rolling(window=5).mean()
    result['SMA20'] = full_df['Close'].rolling(window=20).mean()
    
    # Exponential Moving Averages
    result['EMA12'] = full_df['Close'].ewm(span=12, adjust=False).mean()
    result['EMA26'] = full_df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    result['MACD'] = result['EMA12'] - result['EMA26']
    result['Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['MACD_Hist'] = result['MACD'] - result['Signal']
    
    # RSI
    delta = full_df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    result['BB_Middle'] = full_df['Close'].rolling(window=20).mean()
    std_dev = full_df['Close'].rolling(window=20).std()
    result['BB_Upper'] = result['BB_Middle'] + (std_dev * 2)
    result['BB_Lower'] = result['BB_Middle'] - (std_dev * 2)
    
    # Plot the predictions
    plot_predictions(ticker_symbol, data, future_df, result, window_size, 
                   config['paths']['output_dir'] + "/" + config['paths']['plot_subdir'])
    
    return result


def plot_predictions(ticker, historical_data, future_data, technical_df, window_size, plot_dir):
    """Create visualization of historical data and predictions with technical indicators."""
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(16, 12))
    
    # Setup subplot grid
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Historical end index
    historical_end_idx = len(historical_data) - 1
    
    # 1. Price plot
    ax1 = plt.subplot(gs[0])
    
    # Plot historical and predicted Close prices
    ax1.plot(technical_df.index[:historical_end_idx+1], technical_df['Close'][:historical_end_idx+1], 
            label='Historical Close', color='blue', linewidth=2)
    ax1.plot(technical_df.index[historical_end_idx:], technical_df['Close'][historical_end_idx:], 
            label='Predicted Close', color='red', linestyle='--', linewidth=2)
    
    # Plot moving averages
    ax1.plot(technical_df.index, technical_df['SMA20'], label='SMA20', color='green', alpha=0.7)
    ax1.plot(technical_df.index, technical_df['EMA12'], label='EMA12', color='purple', alpha=0.7)
    
    # Plot Bollinger Bands
    ax1.plot(technical_df.index, technical_df['BB_Upper'], label='BB Upper', color='gray', alpha=0.5)
    ax1.plot(technical_df.index, technical_df['BB_Middle'], color='gray', alpha=0.5)
    ax1.plot(technical_df.index, technical_df['BB_Lower'], label='BB Lower', color='gray', alpha=0.5)
    ax1.fill_between(technical_df.index, technical_df['BB_Upper'], technical_df['BB_Lower'], alpha=0.1, color='gray')
    
    # Add prediction start vertical line
    ax1.axvline(x=historical_data.index[-1], color='black', linestyle='--', alpha=0.7, label='Prediction Start')
    
    # Format the plot
    ax1.set_title(f'Price Prediction with Technical Indicators for {ticker}', fontsize=16)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # 2. RSI plot
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # Plot RSI
    ax2.plot(technical_df.index[:historical_end_idx+1], technical_df['RSI'][:historical_end_idx+1], 
            label='Historical RSI', color='blue')
    ax2.plot(technical_df.index[historical_end_idx:], technical_df['RSI'][historical_end_idx:], 
            label='Predicted RSI', color='red', linestyle='--')
    
    # Add overbought/oversold lines
    ax2.axhline(y=70, color='red', linestyle='-', alpha=0.5)
    ax2.axhline(y=30, color='green', linestyle='-', alpha=0.5)
    ax2.fill_between(technical_df.index, 70, 100, alpha=0.1, color='red')
    ax2.fill_between(technical_df.index, 0, 30, alpha=0.1, color='green')
    
    # Add prediction start vertical line
    ax2.axvline(x=historical_data.index[-1], color='black', linestyle='--', alpha=0.7)
    
    # Format the plot
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # 3. MACD plot
    ax3 = plt.subplot(gs[2], sharex=ax1)
    
    # Plot MACD and signal lines
    ax3.plot(technical_df.index[:historical_end_idx+1], technical_df['MACD'][:historical_end_idx+1], 
            label='Historical MACD', color='blue')
    ax3.plot(technical_df.index[:historical_end_idx+1], technical_df['Signal'][:historical_end_idx+1], 
            label='Historical Signal', color='orange')
    
    ax3.plot(technical_df.index[historical_end_idx:], technical_df['MACD'][historical_end_idx:], 
            label='Predicted MACD', color='red', linestyle='--')
    ax3.plot(technical_df.index[historical_end_idx:], technical_df['Signal'][historical_end_idx:], 
            label='Predicted Signal', color='purple', linestyle='--')
    
    # Add prediction start vertical line
    ax3.axvline(x=historical_data.index[-1], color='black', linestyle='--', alpha=0.7)
    
    # Format the plot
    ax3.set_ylabel('MACD', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # Adjust layout and format x-axis dates
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{ticker}_prediction.png'))
    plt.close()
    logging.info(f"Saved prediction plot to {os.path.join(plot_dir, f'{ticker}_prediction.png')}")


def main():
    """Main function to run the workflow."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.tickers:
        config['data']['tickers'] = args.tickers
    if args.start_date:
        config['data']['start_date'] = args.start_date
    if args.end_date:
        config['data']['end_date'] = args.end_date
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    
    # Create output directories
    dirs = create_output_dirs(config)
    
    # Setup logging
    setup_logging(config['logging'], dirs['logs'])
    
    # Set random seed
    set_seed(config['environment']['seed'])
    
    # Get device
    device = get_device(config['environment']['device'])
    
    logging.info(f"Starting Stock Prediction Model in {args.mode} mode")

    # Load and prepare data
    logging.info("Loading and preparing data...")
    data, feature_names, ticker_map = load_and_prepare_features(config)
    
    if data is None:
        logging.error("Failed to load data. Exiting.")
        return 1
    
    logging.info(f"Data loaded with {len(data)} records and {len(feature_names)} features")
    
    # Prepare data for PyTorch
    data_dict = prepare_data_for_pytorch(data, config, ticker_map, feature_names)
    
    # Run in appropriate mode
    if args.mode == 'train':
        # Train the model
        model, history = train_model(config, data_dict, device)
        
        # Visualize training history
        logging.info("Visualizing training results...")
        # Create training visualization (you can add this as a separate function)
        
    elif args.mode == 'predict':
        # Load checkpoint
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(dirs['base'], config['paths']['model_save_file'])
        
        logging.info(f"Loading model from {checkpoint_path}")
        try:
            # First try loading with weights_only=True (safer)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception as e:
            logging.warning("Failed to load with weights_only=True. Attempting with weights_only=False...")
            logging.warning("This is less secure but necessary for older checkpoints.")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create model and load weights
        model = create_model(
            config=config,
            num_input_features=len(data_dict['input_feature_names']),
            num_tickers=data_dict['num_tickers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Generate predictions for selected tickers
        predict_tickers = config['prediction']['predict_for_tickers']
        if predict_tickers == ['ALL']:
            predict_tickers = config['data']['tickers']
        
        for ticker in predict_tickers:
            if ticker not in ticker_map:
                logging.warning(f"Ticker {ticker} not found in data. Skipping prediction.")
                continue
                
            ticker_id = ticker_map[ticker]
            
            # Get the most recent data for this ticker
            ticker_data = data[data['Ticker_ID'] == ticker_id]
            
            if len(ticker_data) < config['data']['window_size']:
                logging.warning(f"Not enough data for ticker {ticker}. Skipping prediction.")
                continue
            
            # Use only the needed window for prediction
            recent_data = ticker_data.iloc[-config['data']['window_size']:].copy()
            
            # Make future prediction
            prediction_df = predict_future(model, recent_data, ticker_id, config, data_dict, device)
        
    elif args.mode == 'evaluate':
        # Load checkpoint
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(dirs['base'], config['paths']['model_save_file'])
        
        logging.info(f"Loading model from {checkpoint_path}")
        try:
            # First try loading with weights_only=True (safer)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception as e:
            logging.warning("Failed to load with weights_only=True. Attempting with weights_only=False...")
            logging.warning("This is less secure but necessary for older checkpoints.")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create model and load weights
        model = create_model(
            config=config,
            num_input_features=len(data_dict['input_feature_names']),
            num_tickers=data_dict['num_tickers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Evaluate the model
        _, _, _, metrics = evaluate_model(
            model=model,
            dataloader=data_dict['test_loader'],
            criterion=nn.MSELoss(),
            target_indices=data_dict['target_indices'],
            target_feature_names=data_dict['target_feature_names'],
            ticker_map_inv=data_dict['ticker_map_inv'],
            device=device
        )
        
        # Print detailed metrics
        logging.info("Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"  {metric_name}: {metric_value}")
    
    logging.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 