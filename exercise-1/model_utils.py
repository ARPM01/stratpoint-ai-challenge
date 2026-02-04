import pandas as pd
import pickle
import os


class SolarModels:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.feature_columns = []
        self.medians = {}
        self.loaded = False

    def load(self, base_path="."):
        print("Loading data and models...")

        # Load dataset medians
        try:
            df = pd.read_csv(os.path.join(base_path, "solar_weather_dataset.csv"))
            self.medians = df.median(numeric_only=True)
            print("Medians calculated.")
        except Exception as e:
            print(f"Warning: Could not load dataset for medians: {e}")
            self.medians = {}

        # Load feature names
        try:
            with open(os.path.join(base_path, "model_features.pkl"), "rb") as f:
                self.feature_columns = pickle.load(f)
            print(f"Features loaded: {self.feature_columns}")
        except Exception as e:
            print(f"Error loading model features: {e}")
            self.feature_columns = []

        # Load Models
        try:
            with open(os.path.join(base_path, "solar_rf_model.pkl"), "rb") as f:
                self.rf_model = pickle.load(f)
            print("Random Forest model loaded.")
        except Exception as e:
            print(f"Error loading RF model: {e}")

        try:
            with open(os.path.join(base_path, "solar_xgb_model.pkl"), "rb") as f:
                self.xgb_model = pickle.load(f)
            print("XGBoost model loaded.")
        except Exception as e:
            print(f"Error loading XGB model: {e}")

        self.loaded = True


# Global instance
resources = SolarModels()
