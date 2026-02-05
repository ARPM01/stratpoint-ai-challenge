"""
Solar Output Prediction Script

This script loads the trained models and makes predictions on new input data.
The input should contain the required weather features for prediction.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_models():
    """Load the trained models and feature columns."""
    with open("solar_rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    
    with open("solar_xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    
    with open("model_features.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    
    return rf_model, xgb_model, feature_columns


def prepare_input(input_data, feature_columns):
    """
    Prepare input data to match the expected feature format.
    
    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Input features. Should contain all required features.
    feature_columns : list
        List of expected feature column names.
    
    Returns:
    --------
    pd.DataFrame
        Prepared input data with correct columns and order.
    """
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Add cyclical month encoding if Month is provided
    if 'Month' in input_df.columns and 'month_sin' not in input_df.columns:
        input_df['month_sin'] = np.sin(2 * np.pi * input_df['Month'] / 12)
        input_df['month_cos'] = np.cos(2 * np.pi * input_df['Month'] / 12)
    
    # Ensure all required features are present
    missing_features = set(feature_columns) - set(input_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select and order columns to match training data
    input_df = input_df[feature_columns]
    
    return input_df


def predict(input_data, model_type='random_forest'):
    """
    Make predictions using the trained model.
    
    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Input features for prediction.
    model_type : str
        Model to use: 'random_forest' or 'xgboost' (default: 'random_forest')
    
    Returns:
    --------
    float or np.ndarray
        Predicted solar output in kWh/kWp
    """
    rf_model, xgb_model, feature_columns = load_models()
    
    # Prepare input
    prepared_input = prepare_input(input_data, feature_columns)
    
    # Make prediction
    if model_type.lower() in ['random_forest', 'rf']:
        prediction = rf_model.predict(prepared_input)
    elif model_type.lower() in ['xgboost', 'xgb']:
        prediction = xgb_model.predict(prepared_input)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'random_forest' or 'xgboost'")
    
    return prediction


def print_feature_info():
    """Print information about required features."""
    _, _, feature_columns = load_models()
    
    print("Required Features for Prediction:")
    print("=" * 70)
    print("\nWeather Features:")
    weather_features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday'
    ]
    for feat in weather_features:
        if feat in feature_columns:
            print(f"  - {feat}")
    
    print("\nLocation Features:")
    print("  - Latitude (degrees)")
    print("  - Longitude (degrees)")
    
    print("\nTemporal Features:")
    print("  - Month (1-12) - will be converted to cyclical encoding")
    print("  OR provide directly:")
    print("  - month_sin, month_cos")
    
    print("\n" + "=" * 70)


def example_prediction():
    """Run an example prediction."""
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    example_input = {
        'MinTemp': 7,
        'MaxTemp': 30,
        'Rainfall': 2,
        'Evaporation': 8.05,
        'Sunshine': 10,
        'WindGustSpeed': 40.0,
        'WindSpeed9am': 15.0,
        'WindSpeed3pm': 20.0,
        'Humidity9am': 65.0,
        'Humidity3pm': 50.0,
        'Pressure9am': 1013.0,
        'Pressure3pm': 1011.0,
        'Cloud9am': 3.0,
        'Cloud3pm': 4.0,
        'Temp9am': 30.0,
        'Temp3pm': 30.0,
        'RainToday': 0,
        'Latitude': -36.0806,
        'Longitude': 146.9158,
        'Month': 2  # Feb
    }
    
    print("\nInput data (Sydney, January - Summer):")
    for key, value in example_input.items():
        print(f"  {key}: {value}")
    
    print("\nPredictions:")
    rf_pred = predict(example_input, model_type='random_forest')
    xgb_pred = predict(example_input, model_type='xgboost')
    
    print(f"  Random Forest: {rf_pred[0]:.4f} kWh/kWp/day")
    print(f"  XGBoost:       {xgb_pred[0]:.4f} kWh/kWp/day")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SOLAR OUTPUT PREDICTION SCRIPT")
    print("=" * 70)
    
    # Print feature information
    print_feature_info()
    
    # Run example prediction
    example_prediction()