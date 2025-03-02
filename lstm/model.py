"""
LSTM Model for Price Prediction

This module provides functions to build, train, and evaluate LSTM models for price prediction.
Using IBKR data and common processing for any symbol.
"""

import joblib
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as ke
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Import from project modules
from data.processing import calculate_technical_indicators, calculate_trend_indicator, normalize_indicators
from ibkr_api.interface import IBKRInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define Keras layers for easier access
Sequential = ke.models.Sequential
LSTM = ke.layers.LSTM
Dense = ke.layers.Dense
Dropout = ke.layers.Dropout
Bidirectional = ke.layers.Bidirectional
Adam = ke.optimizers.Adam
EarlyStopping = ke.callbacks.EarlyStopping
ReduceLROnPlateau = ke.callbacks.ReduceLROnPlateau

def connect_to_ibkr(host='127.0.0.1', port=4002, client_id=1):
    """Connect to Interactive Brokers."""
    try:
        ibkr = IBKRInterface(host=host, port=port, client_id=client_id)
        connected = ibkr.connect()
        if connected:
            logger.info(f"Connected to IBKR at {host}:{port}")
            return ibkr
        else:
            logger.error("Failed to connect to IBKR")
            return None
    except Exception as e:
        logger.error(f"Error connecting to IBKR: {e}")
        return None

def fetch_historical_data(ibkr, symbol, lookback_days=730):
    """
    Fetch historical data from IBKR.
    
    Args:
        ibkr: IBKR interface
        symbol (str): Symbol to fetch data for
        lookback_days (int): Number of days to look back
        
    Returns:
        pd.DataFrame: Historical data
    """
    try:
        logger.info(f"Fetching {lookback_days} days of historical data for {symbol}")
        
        # Format duration string based on days
        if lookback_days <= 30:
            duration = f"{lookback_days} D"
        elif lookback_days <= 365:
            duration = f"{lookback_days // 30 + 1} M"
        else:
            duration = f"{lookback_days // 365 + 1} Y"
            
        # Fetch hourly data
        df = ibkr.get_historical_data(
            symbol=symbol,
            duration=duration,
            bar_size="1 hour",
            what_to_show="TRADES",
            use_rth=True
        )
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
            
        logger.info(f"Fetched {len(df)} data points for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def save_raw_data(df, symbol, data_dir="data"):
    """Save raw data to CSV file."""
    try:
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        file_path = os.path.join(raw_dir, f"{symbol}.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Raw data saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving raw data: {e}")
        return None

def preprocess_data(df, symbol, seq_length=60, train_split=0.8, output_dir=None):
    """
    Preprocess data for LSTM model.
    
    Args:
        df (pd.DataFrame): Raw data
        symbol (str): Symbol name
        seq_length (int): Sequence length for LSTM
        train_split (float): Proportion of data to use for training
        output_dir (str): Directory to save processed data
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler, feature_cols)
    """
    try:
        logger.info(f"Preprocessing data for {symbol}")
        
        # Handle datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            logger.warning("No datetime column found, creating one")
            df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq='H')
            
        # Sort by datetime
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df, include_all=True)
        
        # Calculate trend indicator
        df = calculate_trend_indicator(df)
        
        # Normalize indicators
        df = normalize_indicators(df)
        
        # Save processed data if output_dir is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{symbol}_processed.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Processed data saved to {output_file}")
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        # Split data into training and testing sets
        train_size = int(len(df) * train_split)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # Use normalized columns for features
        feature_cols = [col for col in df.columns if col.endswith('_norm')]
        
        # Ensure close_norm is included
        if 'close_norm' not in feature_cols and 'close_norm' in df.columns:
            feature_cols.append('close_norm')
        
        # Create sequences for LSTM
        X_train, y_train = create_sequences(
            train_df[feature_cols].values,
            train_df['close_norm'].values,
            seq_length
        )
        
        X_test, y_test = create_sequences(
            test_df[feature_cols].values,
            test_df['close_norm'].values,
            seq_length
        )
        
        # Create and fit scaler for later inverse transform
        scaler = MinMaxScaler()
        # For scaling back to original values, we need additional columns
        additional_cols = [col for col in df.columns if col in ['open', 'high', 'low', 'volume'] and f"{col}_norm" in feature_cols]
        all_cols = additional_cols + ['close']
        scaler.fit(df[all_cols])
        
        logger.info(f"Preprocessing complete. X_train shape: {X_train.shape}, feature_cols: {len(feature_cols)}")
        
        return X_train, y_train, X_test, y_test, scaler, feature_cols, train_df, test_df
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def create_sequences(data, target, seq_length):
    """
    Create sequences for LSTM model.
    
    Args:
        data (np.ndarray): Input features
        target (np.ndarray): Target values
        seq_length (int): Sequence length
        
    Returns:
        tuple: (X sequences, y targets)
    """
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)

def build_lstm_model(input_shape, config):
    """
    Build LSTM model with given configuration.
    
    Args:
        input_shape (tuple): Shape of input data
        config (dict): Model configuration
        
    Returns:
        keras.Model: LSTM model
    """
    model = Sequential([
        ke.Input(shape=input_shape)  # Explicit input layer
    ])
    
    # First LSTM layer
    if config.get('bidirectional', False):
        model.add(Bidirectional(LSTM(
            units=config['units'], 
            return_sequences=True,
            kernel_regularizer=ke.regularizers.l2(config.get('l2_reg', 0))
        )))
    else:
        model.add(LSTM(
            units=config['units'], 
            return_sequences=True,
            kernel_regularizer=ke.regularizers.l2(config.get('l2_reg', 0))
        ))
    
    model.add(Dropout(config['dropout']))
    
    # Middle LSTM layers
    for _ in range(config.get('num_layers', 2) - 1):
        if config.get('bidirectional', False):
            model.add(Bidirectional(LSTM(
                units=config['units'], 
                return_sequences=True,
                kernel_regularizer=ke.regularizers.l2(config.get('l2_reg', 0))
            )))
        else:
            model.add(LSTM(
                units=config['units'], 
                return_sequences=True,
                kernel_regularizer=ke.regularizers.l2(config.get('l2_reg', 0))
            ))
        model.add(Dropout(config['dropout']))
    
    # Final LSTM layer (no return_sequences)
    if config.get('bidirectional', False):
        model.add(Bidirectional(LSTM(
            units=config['units'],
            kernel_regularizer=ke.regularizers.l2(config.get('l2_reg', 0))
        )))
    else:
        model.add(LSTM(
            units=config['units'],
            kernel_regularizer=ke.regularizers.l2(config.get('l2_reg', 0))
        ))
    
    model.add(Dropout(config['dropout']))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

def train_model(model, x_train, y_train, x_test, y_test, config):
    """
    Train LSTM model.
    
    Args:
        model (keras.Model): LSTM model
        x_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        x_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing targets
        config (dict): Training configuration
        
    Returns:
        keras.callbacks.History: Training history
    """
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['patience'],
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.get('lr_factor', 0.2),
        patience=config['patience'] // 2,
        min_lr=config.get('min_lr', 0.00001)
    )
    
    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(x_test, y_test),
        shuffle=False,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history

def evaluate_model(model, x_test, y_test, scaler=None, x_test_unscaled=None):
    """
    Evaluate LSTM model.
    
    Args:
        model (keras.Model): LSTM model
        x_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler for inverse transform
        x_test_unscaled (np.ndarray): Unscaled test features for inverse transform
        
    Returns:
        tuple: (rmse, mae, r2)
    """
    # Make predictions
    y_pred = model.predict(x_test)
    
    # If scaler is provided, inverse transform predictions and targets
    if scaler and x_test_unscaled is not None:
        # Create dummy arrays for inverse transform
        y_test_inverse = np.zeros((len(y_test), scaler.n_features_in_))
        y_pred_inverse = np.zeros((len(y_pred), scaler.n_features_in_))
        
        # Set the close column (assuming it's the last column)
        y_test_inverse[:, -1] = y_test.flatten()
        y_pred_inverse[:, -1] = y_pred.flatten()
        
        # Inverse transform
        y_test_descaled = scaler.inverse_transform(y_test_inverse)[:, -1]
        y_pred_descaled = scaler.inverse_transform(y_pred_inverse)[:, -1]
    else:
        # Use raw values if no scaler
        y_test_descaled = y_test
        y_pred_descaled = y_pred
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_descaled, y_pred_descaled))
    mae = mean_absolute_error(y_test_descaled, y_pred_descaled)
    r2 = r2_score(y_test_descaled, y_pred_descaled)
    
    logger.info(f"Model evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return rmse, mae, r2, y_test_descaled, y_pred_descaled

def plot_results(history, y_test_descaled, y_pred_descaled, config):
    """
    Plot training results.
    
    Args:
        history (keras.callbacks.History): Training history
        y_test_descaled (np.ndarray): True test values
        y_pred_descaled (np.ndarray): Predicted test values
        config (dict): Model configuration
    """
    # Create output directory
    plots_dir = os.path.join("plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Training and Validation Loss - {config['label']}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"loss_{config['label']}.png"))
    plt.close()
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_descaled, label='Actual')
    plt.plot(y_pred_descaled, label='Predicted')
    plt.title(f"Predictions vs Actual - {config['label']}")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"predictions_{config['label']}.png"))
    plt.close()

def save_experiment_results(config, rmse, mae, r2, results_df_path="experiment_results.csv"):
    """
    Save experiment results to CSV file.
    
    Args:
        config (dict): Model configuration
        rmse (float): Root mean squared error
        mae (float): Mean absolute error
        r2 (float): R-squared score
        results_df_path (str): Path to save results
        
    Returns:
        pd.DataFrame: Updated results dataframe
    """
    # Create new results row
    new_results = pd.DataFrame({
        'experiment': [config['label']],
        'symbol': [config['symbol']],
        'rmse': [rmse],
        'mae': [mae],
        'r2': [r2],
        'units': [config['units']],
        'num_layers': [config['num_layers']],
        'dropout': [config['dropout']],
        'bidirectional': [config['bidirectional']],
        'learning_rate': [config['learning_rate']],
        'batch_size': [config['batch_size']],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })
    
    # Load existing results if available
    if os.path.exists(results_df_path):
        results_df = pd.read_csv(results_df_path)
        results_df = pd.concat([results_df, new_results], ignore_index=True)
    else:
        results_df = new_results
    
    # Save to CSV
    results_df.to_csv(results_df_path, index=False)
    logger.info(f"Results saved to {results_df_path}")
    
    return results_df

def run_lstm_training(config, ibkr=None):
    """
    Run complete LSTM training process.
    
    Args:
        config (dict): Model configuration
        ibkr (IBKRInterface): IBKR interface (will create new if None)
        
    Returns:
        dict: Training results
    """
    try:
        symbol = config['symbol']
        logger.info(f"Starting LSTM training for {symbol} with config: {config['label']}")
        
        # Set up directories
        data_dir = os.path.join("data")
        models_dir = os.path.join("models", "lstm")
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Path del archivo crudo donde guardamos/cargamos el CSV
        raw_data_path = os.path.join(data_dir, "raw", f"{symbol}_raw_data.csv")

        # Si el archivo raw existe, lo cargamos en vez de descargarlo
        if os.path.exists(raw_data_path):
            logger.info(f"Found existing raw data for {symbol}. Loading from {raw_data_path}")
            df = pd.read_csv(raw_data_path, parse_dates=["Date"], index_col="Date")
        else:
            logger.info(f"No existing raw data found for {symbol}. Fetching from IBKR...")

            # Connect to IBKR if not provided
            if ibkr is None:
                ibkr = connect_to_ibkr()
                if ibkr is None:
                    logger.error("Failed to connect to IBKR. Exiting.")
                    return {"status": "error", "message": "Failed to connect to IBKR"}
            
            # Fetch historical data
            df = fetch_historical_data(ibkr, symbol, lookback_days=730)  # 2 years
            if df.empty:
                logger.error(f"No data fetched for {symbol}. Exiting.")
                return {"status": "error", "message": f"No data fetched for {symbol}"}
            
            # Save raw data
            save_raw_data(df, symbol, data_dir)
        
        # Preprocess data
        X_train, y_train, X_test, y_test, scaler, feature_cols, train_df, test_df = preprocess_data(
            df, 
            symbol, 
            seq_length=config['seq_length'],
            output_dir=os.path.join(data_dir, "processed")
        )
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape, config)
        
        # Train model
        history = train_model(model, X_train, y_train, X_test, y_test, config)
        
        # Evaluate model
        rmse, mae, r2, y_test_descaled, y_pred_descaled = evaluate_model(
            model, X_test, y_test, scaler, X_test
        )
        
        # Plot results
        plot_results(history, y_test_descaled, y_pred_descaled, config)
        
        # Save experiment results
        results_df = save_experiment_results(config, rmse, mae, r2)
        
        # Save model and scaler
        model_path = os.path.join(models_dir, f"lstm_model_universal.keras")
        scaler_path = os.path.join(models_dir, f"scaler_universal.pkl")
        
        # Save model
        ke.saving.save_model(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Model and scaler saved to {models_dir}")
        
        return {
            "status": "success",
            "symbol": symbol,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "model_path": model_path,
            "scaler_path": scaler_path
        }
    
    except Exception as e:
        logger.error(f"Error in LSTM training: {e}")
        return {"status": "error", "message": str(e)}

def load_model(model_path="models/lstm/lstm_model_universal.keras", scaler_path="models/lstm/scaler_universal.pkl"):
    """
    Load saved model and scaler.
    
    Args:
        model_path (str): Path to saved model
        scaler_path (str): Path to saved scaler
        
    Returns:
        tuple: (model, scaler)
    """
    try:
        # Load model
        model = ke.models.load_model(model_path)
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        
        logger.info(f"Model and scaler loaded from {os.path.dirname(model_path)}")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model and scaler: {e}")
        return None, None

def predict_future_prices(model, scaler, current_data, seq_length=60, steps=5):
    """
    Predict future prices using the LSTM model.
    
    Args:
        model (keras.Model): Trained LSTM model
        scaler (sklearn.preprocessing.MinMaxScaler): Fitted scaler
        current_data (np.ndarray): Current sequence data (normalized)
        seq_length (int): Sequence length
        steps (int): Number of steps to predict
        
    Returns:
        np.ndarray: Predicted prices
    """
    try:
        # Make sure current_data has the right shape
        if current_data.shape[0] < seq_length:
            logger.error(f"Not enough data points. Need at least {seq_length}, got {current_data.shape[0]}")
            return None
        
        # Get the last sequence
        last_sequence = current_data[-seq_length:].copy()
        
        # Initialize predictions array
        predictions = []
        
        # Predict future steps
        for _ in range(steps):
            # Reshape for prediction
            x_pred = last_sequence.reshape(1, seq_length, last_sequence.shape[1])
            
            # Predict next value
            next_pred = model.predict(x_pred, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence for next prediction (rolling window)
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, -1] = next_pred  # Assuming close is the last feature
        
        # Create dummy arrays for inverse transform
        pred_inverse = np.zeros((len(predictions), scaler.n_features_in_))
        
        # Set the close column (assuming it's the last column)
        pred_inverse[:, -1] = predictions
        
        # Inverse transform to get actual prices
        predicted_prices = scaler.inverse_transform(pred_inverse)[:, -1]
        
        return predicted_prices
    
    except Exception as e:
        logger.error(f"Error predicting future prices: {e}")
        return None

# If run as main script
if __name__ == "__main__":
    try:
        # Define the optimal configuration
        universal_config = {
            'label': 'universal_model',
            'symbol': 'AMZN',  # Initial training symbol, but model will work for any
            'seq_length': 60,
            'epochs': 50,
            'batch_size': 128,
            'units': 200,
            'dropout': 0.41,
            'learning_rate': 0.0001,
            'patience': 10,
            'num_layers': 2,
            'bidirectional': True,
            'l2_reg': 0,
            'lr_factor': 0.3,
            'min_lr': 1e-05
        }
        
        # Run training process
        results = run_lstm_training(universal_config)
        
        if results["status"] == "success":
            logger.info(f"Training successful. RMSE: {results['rmse']:.4f}, R²: {results['r2']:.4f}")
        else:
            logger.error(f"Training failed: {results['message']}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")