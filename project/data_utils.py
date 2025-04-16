import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch

def load_and_prepare_features(config):
    """
    Loads stock data using yfinance based on configuration,
    calculates technical indicators and prepares features.
    
    Args:
        config: Dictionary containing data configuration
        
    Returns:
        tuple: (pd.DataFrame, list, dict):
               Processed DataFrame with combined features,
               list of feature names (including 'Ticker_ID'),
               dictionary mapping ticker symbol to integer ID.
    """
    target_tickers = config['data']['tickers']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    interval = config['data']['interval']
    feature_config = config['data']['features']
    
    all_data_frames = []
    ticker_to_id = {ticker: i for i, ticker in enumerate(target_tickers)}
    final_feature_names = []  # Will be determined after processing first successful ticker

    logging.info(f"Loading and processing data for {len(target_tickers)} tickers...")

    for ticker_id, ticker in enumerate(tqdm(target_tickers, desc="Loading Tickers")):
        logging.info(f"Processing {ticker} ({ticker_id + 1}/{len(target_tickers)})...")
        try:
            # --- Load Data for Current Ticker ---
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
            if data.empty:
                logging.warning(f"No data found for {ticker}. Skipping.")
                continue
            logging.info(f"Loaded {len(data)} records for {ticker}")

            # Clean column names
            data.columns = [(col[0] if isinstance(col, tuple) else col).replace(' ', '_') for col in data.columns]
            base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']  # Adjust if needed
            present_base_cols = [col for col in base_cols if col in data.columns]
            if 'Close' not in present_base_cols:
                logging.warning(f"'Close' price missing for {ticker}. Skipping.")
                continue
            data = data[present_base_cols].copy()

            # --- Feature Calculation (Per Ticker) ---
            # SMA (Simple Moving Averages)
            for window in feature_config.get('sma_windows', [5, 10, 20]):
                data[f'SMA{window}'] = data['Close'].rolling(window=window, min_periods=1).mean()
                
            # RSI (Relative Strength Index)
            rsi_window = feature_config.get('rsi_window', 14)
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=rsi_window, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0.0).rolling(window=rsi_window, min_periods=1).mean()
            rs = gain / loss
            rsi_values = 100.0 - (100.0 / (1.0 + rs))
            # Fix chained assignment warnings
            rsi_values = rsi_values.replace([np.inf, -np.inf], 100.0)
            rsi_values = rsi_values.fillna(50.0)
            data['RSI'] = rsi_values

            # MACD (Moving Average Convergence Divergence)
            macd_params = feature_config.get('macd_params', {'fast_ema': 12, 'slow_ema': 26, 'signal_ema': 9})
            fast_ema = data['Close'].ewm(span=macd_params['fast_ema'], adjust=False).mean()
            slow_ema = data['Close'].ewm(span=macd_params['slow_ema'], adjust=False).mean()
            data['MACD'] = fast_ema - slow_ema
            data['MACD_Signal'] = data['MACD'].ewm(span=macd_params['signal_ema'], adjust=False).mean()
            data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

            # Bollinger Bands
            bb_window = feature_config.get('bollinger_window', 20)
            bb_std = feature_config.get('bollinger_std_dev', 2)
            data['BB_Middle'] = data['Close'].rolling(window=bb_window).mean()
            rolling_std = data['Close'].rolling(window=bb_window).std()
            data['BB_Upper'] = data['BB_Middle'] + (rolling_std * bb_std)
            data['BB_Lower'] = data['BB_Middle'] - (rolling_std * bb_std)

            # Return calculation
            if feature_config.get('calculate_return', True):
                returns = data['Close'].pct_change()
                data['Return'] = returns.fillna(0.0)  # Fix chained assignment warning

            # Add Ticker ID
            data['Ticker_ID'] = ticker_to_id[ticker]

            # Drop initial NaNs created by rolling windows/diffs
            initial_len = len(data)
            data = data.dropna()
            final_len = len(data)
            if final_len < 1:
                logging.warning(f"Data for {ticker} became empty after NaN drop. Skipping.")
                continue
            logging.info(f"Dropped {initial_len - final_len} rows with NaNs for {ticker}. Final length: {final_len}")

            # Store feature names from the first successful ticker
            if not final_feature_names:
                final_feature_names = list(data.columns)

            # Ensure columns match standard feature list
            current_cols = list(data.columns)
            if current_cols != final_feature_names:
                logging.warning(f"Column mismatch for {ticker}. Expected {final_feature_names}, got {current_cols}. Reindexing.")
                data = data.reindex(columns=final_feature_names)

            all_data_frames.append(data)

        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue  # Skip this ticker

    if not all_data_frames:
        logging.error("No data successfully processed for any ticker.")
        return None, None, None

    # --- Concatenate DataFrames ---
    logging.info("Concatenating data from all tickers...")
    combined_data = pd.concat(all_data_frames, axis=0)
    logging.info(f"Combined data shape: {combined_data.shape}")

    # --- Final NaN Check ---
    initial_len = len(combined_data)
    combined_data = combined_data.dropna()
    final_len = len(combined_data)
    logging.info(f"Dropped {initial_len - final_len} rows with NaNs from combined data. Final shape: {combined_data.shape}")

    if combined_data.empty:
        logging.error("Combined DataFrame became empty after final NaN drop.")
        return None, None, None

    # Ensure Ticker_ID is integer type
    combined_data['Ticker_ID'] = combined_data['Ticker_ID'].astype(int)

    logging.info(f"Final feature names: {final_feature_names}")
    return combined_data, final_feature_names, ticker_to_id


def prepare_data_for_pytorch(data_pd, config, ticker_to_id, feature_names=None):
    """
    Prepare data for PyTorch model training with enhanced progress tracking.
    This function creates sequences where each input is a window of data, and 
    the target is the next time step's values for the target features.
    
    Args:
        data_pd: Processed DataFrame with all features and Ticker_ID
        config: Configuration dictionary
        ticker_to_id: Dictionary mapping ticker symbols to IDs
        feature_names: Optional list of feature names to use (if None, use all columns except Ticker_ID)
        
    Returns:
        Dictionary containing train/test DataLoaders, feature names, and scalers
    """
    target_features = config['data']['target_features']
    window_size = config['data']['window_size']
    batch_size = config['data']['batch_size']
    test_size = config['data']['test_split_ratio']
    
    logging.info("Preparing data for PyTorch model...")
    
    # Separate Ticker_ID from features
    if 'Ticker_ID' not in data_pd.columns:
        raise ValueError("Ticker_ID column missing from input DataFrame")
    
    feature_data = data_pd.copy()
    ticker_ids = feature_data.pop('Ticker_ID').values
    
    # Get unique ticker IDs
    unique_ticker_ids = np.unique(ticker_ids)
    num_tickers = len(unique_ticker_ids)
    logging.info(f"Found {num_tickers} unique tickers")
    
    # Create inverse mapping (ID to ticker symbol)
    ticker_map_inv = {v: k for k, v in ticker_to_id.items()}
    
    # Adjust feature names (remove Ticker_ID if present)
    if feature_names is None:
        input_feature_names = list(feature_data.columns)
    else:
        input_feature_names = [f for f in feature_names if f != 'Ticker_ID']
    
    # Ensure all target features are in the input features
    if not all(tf in input_feature_names for tf in target_features):
        missing = [tf for tf in target_features if tf not in input_feature_names]
        raise ValueError(f"Target features {missing} not found in input features {input_feature_names}")
    
    # Get indices of target features in input features list for later use
    target_indices = [input_feature_names.index(f) for f in target_features]
    
    # Split data by ticker to ensure no sequence crosses ticker boundaries
    sequences_by_ticker = {}
    targets_by_ticker = {}
    ids_by_ticker = {}
    scalers = {}
    
    logging.info("Preparing sequences by ticker...")
    for ticker_id in tqdm(unique_ticker_ids, desc="Processing tickers"):
        # Extract data for this ticker
        ticker_mask = ticker_ids == ticker_id
        ticker_data = feature_data.iloc[ticker_mask]
        
        if len(ticker_data) <= window_size:
            logging.warning(f"Ticker ID {ticker_id} has only {len(ticker_data)} samples, which is <= window_size {window_size}. Skipping.")
            continue
        
        # Scale the data
        ticker_values = ticker_data.values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(ticker_values)
        scalers[ticker_id] = scaler
        
        # Create sequences for this ticker
        X, y, ids = [], [], []
        for i in range(len(scaled_values) - window_size):
            # Input sequence: window_size days of all features
            X.append(scaled_values[i:i+window_size])
            
            # Target: the next day's (day after window) target features only
            next_day_values = scaled_values[i+window_size]
            # Extract only target feature values using target_indices
            target_values = next_day_values[target_indices]
            y.append(target_values)
            
            # Store ticker ID for each sequence
            ids.append(ticker_id)
        
        # Store sequences for this ticker
        if X:  # Ensure we have sequences
            sequences_by_ticker[ticker_id] = np.array(X)
            targets_by_ticker[ticker_id] = np.array(y)
            ids_by_ticker[ticker_id] = np.array(ids)
    
    # Combine sequences from all tickers
    all_sequences = []
    all_targets = []
    all_ids = []
    
    for ticker_id in sequences_by_ticker:
        all_sequences.append(sequences_by_ticker[ticker_id])
        all_targets.append(targets_by_ticker[ticker_id])
        all_ids.append(ids_by_ticker[ticker_id])
    
    if not all_sequences:
        raise ValueError("No valid sequences created. Check if window_size is too large for your data.")
    
    X = np.vstack(all_sequences)
    y = np.vstack(all_targets)
    ids = np.concatenate(all_ids)
    
    logging.info(f"Created {len(X)} sequences across {len(sequences_by_ticker)} tickers")
    logging.info(f"Input shape: {X.shape}, Target shape: {y.shape}")
    logging.info(f"Each input is a sequence of {window_size} timesteps with {X.shape[2]} features")
    logging.info(f"Each target is the next timestep's {len(target_indices)} target features")
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    ids_tensor = torch.LongTensor(ids)
    
    # Split into train and test
    logging.info("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_tensor, y_tensor, ids_tensor, test_size=test_size, random_state=config['environment']['seed']
    )
    
    # Create inverse scaler mapping function
    def inverse_transform_fn(scaled_data, ticker_id):
        """Transform scaled data back to original scale for a specific ticker."""
        # scaled_data shape: (batch_size, num_target_features)
        if ticker_id not in scalers:
            raise ValueError(f"No scaler found for ticker ID {ticker_id}")
        
        scaler = scalers[ticker_id]
        
        # Create a dummy array with zeros for all features
        dummy_full_features = np.zeros((scaled_data.shape[0], len(input_feature_names)))
        
        # Place the target values in their correct positions
        for i, target_idx in enumerate(target_indices):
            dummy_full_features[:, target_idx] = scaled_data[:, i]
        
        # Inverse transform the full feature array
        unscaled_full = scaler.inverse_transform(dummy_full_features)
        
        # Extract just the target features again
        result = np.zeros_like(scaled_data)
        for i, target_idx in enumerate(target_indices):
            result[:, i] = unscaled_full[:, target_idx]
        
        return result
    
    # Create DataLoaders
    logging.info("Creating DataLoader objects...")
    train_dataset = TensorDataset(X_train, ids_train, y_train)
    test_dataset = TensorDataset(X_test, ids_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logging.info(f"Train set: {len(train_dataset)} sequences")
    logging.info(f"Test set: {len(test_dataset)} sequences")
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'input_feature_names': input_feature_names,
        'target_feature_names': target_features,
        'target_indices': target_indices,
        'num_tickers': num_tickers,
        'window_size': window_size,
        'scalers': scalers,
        'inverse_transform_fn': inverse_transform_fn,
        'ticker_map_inv': ticker_map_inv
    } 