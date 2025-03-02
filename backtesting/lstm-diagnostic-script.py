"""
LSTM Model Diagnostic Script

This script helps diagnose issues with the LSTM model, particularly focusing
on the feature compatibility between the training data and prediction data.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import keras as ke
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from ibkr_api.interface import IBKRInterface
from data.processing import calculate_technical_indicators, calculate_trend_indicator, normalize_indicators

def analyze_model():
    """Analyze LSTM model structure and requirements."""
    try:
        # Load the universal model
        model_path = os.path.join('models', 'lstm', 'lstm_model_universal.keras')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return
        
        # Load the model
        model = ke.models.load_model(model_path)
        
        # Print model summary
        model.summary()
        
        # Get input shape details
        input_shape = model.input_shape
        logger.info(f"Model input shape: {input_shape}")
        
        # Extract specifically the feature dimension (last dimension of input)
        if input_shape and len(input_shape) > 2:
            feature_count = input_shape[-1]
            logger.info(f"Expected feature count: {feature_count}")
        else:
            logger.warning(f"Unable to determine feature count from input shape: {input_shape}")
        
        # Check saved training data if available
        processed_dir = os.path.join("data", "processed")
        if os.path.exists(processed_dir):
            for file in os.listdir(processed_dir):
                if file.endswith("_processed.csv"):
                    logger.info(f"Examining processed file: {file}")
                    df = pd.read_csv(os.path.join(processed_dir, file))
                    
                    # Count normalized features
                    feature_cols = [col for col in df.columns if col.endswith('_norm')]
                    logger.info(f"Found {len(feature_cols)} normalized features in {file}")
                    
                    # List all normalized features
                    logger.info(f"Normalized features: {feature_cols}")
                    
                    # Only process one file for brevity
                    break
        
        return {"status": "success", "feature_count": feature_count if 'feature_count' in locals() else None}
    
    except Exception as e:
        logger.error(f"Error analyzing model: {e}")
        return {"status": "error", "message": str(e)}

def analyze_real_time_data(symbol="AAPL"):
    """Analyze real-time data feature compatibility with model."""
    try:
        # Connect to IBKR
        ibkr = IBKRInterface()
        connected = ibkr.connect()
        
        if not connected:
            logger.error("Failed to connect to IBKR")
            return {"status": "error", "message": "Failed to connect to IBKR"}
        
        # Get market data
        data = ibkr.get_historical_data(
            symbol=symbol,
            duration="15 D",
            bar_size="1 hour",
            what_to_show="TRADES",
            use_rth=True
        )
        
        if data.empty:
            logger.error(f"No data retrieved for {symbol}")
            ibkr.disconnect()
            return {"status": "error", "message": f"No data retrieved for {symbol}"}
        
        # Process data
        data = calculate_technical_indicators(data, include_all=True)
        data = calculate_trend_indicator(data)
        data = normalize_indicators(data)
        
        # Analyze features
        feature_cols = [col for col in data.columns if col.endswith('_norm')]
        logger.info(f"Found {len(feature_cols)} normalized features in real-time data for {symbol}")
        logger.info(f"Real-time normalized features: {feature_cols}")
        
        # Compare with model requirements (from analyze_model)
        model_analysis = analyze_model()
        if model_analysis and model_analysis.get("status") == "success":
            expected_count = model_analysis.get("feature_count")
            if expected_count:
                if len(feature_cols) != expected_count:
                    logger.warning(f"Feature count mismatch: model expects {expected_count}, real-time data has {len(feature_cols)}")
                    
                    if len(feature_cols) < expected_count:
                        logger.warning(f"Missing {expected_count - len(feature_cols)} features")
                    else:
                        logger.warning(f"Extra {len(feature_cols) - expected_count} features")
                else:
                    logger.info(f"Feature count matches: {expected_count}")
        
        # Disconnect from IBKR
        ibkr.disconnect()
        
        return {
            "status": "success", 
            "feature_count": len(feature_cols),
            "features": feature_cols
        }
    
    except Exception as e:
        logger.error(f"Error analyzing real-time data: {e}")
        try:
            ibkr.disconnect()
        except:
            pass
        return {"status": "error", "message": str(e)}

def check_model_compatibility():
    """
    Check if the LSTM model is compatible with the real-time data processing.
    This helps identify issues with feature mismatches.
    """
    logger.info("Checking LSTM model compatibility with real-time data...")
    
    # Step 1: Analyze model
    model_info = analyze_model()
    
    # Step 2: Analyze real-time data
    data_info = analyze_real_time_data()
    
    # Step 3: Provide suggestions
    if (model_info and model_info.get("status") == "success" and 
        data_info and data_info.get("status") == "success"):
        
        model_features = model_info.get("feature_count", 0)
        data_features = data_info.get("feature_count", 0)
        
        if model_features != data_features:
            logger.warning("COMPATIBILITY ISSUE DETECTED!")
            logger.warning(f"Model expects {model_features} features but real-time data has {data_features} features")
            
            logger.info("\nSUGGESTIONS TO FIX THE ISSUE:")
            
            if data_features < model_features:
                logger.info("1. You need to add more features to the real-time data processing")
                logger.info("   - Ensure all technical indicators used during training are calculated")
                logger.info("   - Add missing normalized features with default values (e.g., 0.5)")
            else:
                logger.info("1. You need to limit the features used in real-time prediction")
                logger.info("   - Select only the most important features from the real-time data")
                logger.info("   - Use the same feature selection logic as during training")
            
            logger.info("2. Alternative: Retrain the model with the current feature set")
            logger.info("   - This ensures the model matches the current data processing pipeline")
        else:
            logger.info("FEATURE COUNT COMPATIBILITY CONFIRMED!")
            logger.info(f"Both model and real-time data use {model_features} features")
            
        return {
            "status": "success",
            "compatible": model_features == data_features,
            "model_features": model_features,
            "data_features": data_features
        }
    else:
        logger.error("Failed to complete compatibility check")
        return {"status": "error", "message": "Failed to complete compatibility check"}

if __name__ == "__main__":
    logger.info("Starting LSTM model diagnostic")
    
    # Check model and data compatibility
    result = check_model_compatibility()
    
    if result.get("status") == "success":
        if result.get("compatible"):
            logger.info("Diagnostic complete: Model and data are compatible")
        else:
            logger.warning("Diagnostic complete: Compatibility issues detected, see suggestions above")
    else:
        logger.error("Diagnostic failed, please check the error messages")