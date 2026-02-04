import numpy as np
import pandas as pd
from langchain_core.tools import tool
from model_utils import resources

# City coordinates mapping
# Check dataset-preparation.ipynb for more details on how these coordinates were sourced.
city_coords = {
    "Albury": (np.float64(-36.0806), np.float64(146.9158)),
    "Brisbane": (np.float64(-27.4678), np.float64(153.0281)),
    "Cairns": (np.float64(-16.92), np.float64(145.78)),
    "Canberra": (np.float64(-35.2931), np.float64(149.1269)),
    "Hobart": (np.float64(-42.8806), np.float64(147.325)),
    "Launceston": (np.float64(-41.4419), np.float64(147.145)),
    "Melbourne": (np.float64(-37.8142), np.float64(144.9631)),
    "Newcastle": (np.float64(-32.9167), np.float64(151.75)),
    "Perth": (np.float64(-31.9559), np.float64(115.8606)),
    "Sydney": (np.float64(-33.8678), np.float64(151.21)),
    "Adelaide": (np.float64(-34.9275), np.float64(138.6)),
    "Darwin": (np.float64(-12.4381), np.float64(130.8411)),
    "Townsville": (np.float64(-19.25), np.float64(146.8167)),
    "Ballarat": (np.float64(-37.5608), np.float64(143.8475)),
    "Bendigo": (np.float64(-36.75), np.float64(144.2667)),
    "Penrith": (np.float64(-33.7511), np.float64(150.6942)),
    "BadgerysCreek": (np.float64(-33.7511), np.float64(150.6942)),
    "Cobar": (np.float64(-31.4997), np.float64(145.8319)),
    "Moree": (np.float64(-29.4658), np.float64(149.8339)),
    "NorahHead": (np.float64(-33.3), np.float64(151.2)),
    "Richmond": (np.float64(-33.6), np.float64(150.75)),
    "SydneyAirport": (np.float64(-33.8678), np.float64(151.21)),
    "WaggaWagga": (np.float64(-35.1189), np.float64(147.3689)),
    "Williamtown": (np.float64(-32.9167), np.float64(151.75)),
    "Wollongong": (np.float64(-34.4331), np.float64(150.8831)),
    "Tuggeranong": (np.float64(-35.2931), np.float64(149.1269)),
    "MountGinini": (np.float64(-35.2931), np.float64(149.1269)),
    "MelbourneAirport": (np.float64(-37.8142), np.float64(144.9631)),
    "Watsonia": (np.float64(-37.8142), np.float64(144.9631)),
    "GoldCoast": (np.float64(-28.0167), np.float64(153.4)),
    "MountGambier": (np.float64(-37.8294), np.float64(140.7828)),
    "Nuriootpa": (np.float64(-34.4667), np.float64(138.9833)),
    "PearceRAAF": (np.float64(-31.9559), np.float64(115.8606)),
    "PerthAirport": (np.float64(-31.9559), np.float64(115.8606)),
    "NorfolkIsland": (-29.0278, 167.9486),
    "Nhil": (-36.2135, 141.933),
    "Dartmoor": (-37.99, 141.46),
    "Woomera": (-31.1667, 136.8167),
    "Witchcliffe": (-34.0532, 115.1583),
    "SalmonGums": (-32.0481, 121.9393),
    "AliceSprings": (-23.698, 133.8807),
    "Uluru": (-25.3444, 131.0369),
    "CoffsHarbour": (-30.2963, 153.1133),
    "Sale": (-38.108, 147.0663),
    "Mildura": (-34.185, 142.1629),
    "Portland": (-38.345, 141.6058),
    "Albany": (-35.0247, 117.884),
    "Walpole": (-34.9553, 116.7374),
    "Katherine": (-14.4658, 132.263),
}


@tool
def get_seasonal_weather_defaults(month: int = None) -> str:
    """
    Returns typical weather conditions for Australia based on the month/season.
    Use this tool to get default weather parameters when specific conditions are not provided.

    Args:
        month: Month number (1-12). If not provided, will ask user to specify.

    Returns:
        A string with seasonal information and typical weather parameter defaults for Australia.
    """
    if month is None:
        return "Please provide a month number (1-12, where 1=January, 12=December) to get seasonal weather defaults."

    if not isinstance(month, int) or month < 1 or month > 12:
        return (
            f"Invalid month: {month}. Please provide a month number between 1 and 12."
        )

    # Define seasonal defaults for Australia
    # Summer: Dec, Jan, Feb (12, 1, 2)
    # Autumn: Mar, Apr, May (3, 4, 5)
    # Winter: Jun, Jul, Aug (6, 7, 8)
    # Spring: Sep, Oct, Nov (9, 10, 11)

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Calculate cyclical encoding for the month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    if month in [12, 1, 2]:  # Summer
        season = "Summer"
        defaults = {
            "MinTemp": 20.0,
            "MaxTemp": 30.0,
            "Rainfall": 2.0,
            "Evaporation": 8.0,
            "Sunshine": 10.0,
            "WindGustSpeed": 40.0,
            "WindSpeed9am": 15.0,
            "WindSpeed3pm": 20.0,
            "Humidity9am": 65.0,
            "Humidity3pm": 50.0,
            "Pressure9am": 1013.0,
            "Pressure3pm": 1011.0,
            "Cloud9am": 3.0,
            "Cloud3pm": 4.0,
            "Temp9am": 24.0,
            "Temp3pm": 28.0,
            "RainToday": 0,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
    elif month in [3, 4, 5]:  # Autumn
        season = "Autumn"
        defaults = {
            "MinTemp": 14.0,
            "MaxTemp": 23.0,
            "Rainfall": 3.0,
            "Evaporation": 5.0,
            "Sunshine": 7.0,
            "WindGustSpeed": 35.0,
            "WindSpeed9am": 12.0,
            "WindSpeed3pm": 18.0,
            "Humidity9am": 70.0,
            "Humidity3pm": 55.0,
            "Pressure9am": 1015.0,
            "Pressure3pm": 1013.0,
            "Cloud9am": 4.0,
            "Cloud3pm": 5.0,
            "Temp9am": 18.0,
            "Temp3pm": 22.0,
            "RainToday": 0,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
    elif month in [6, 7, 8]:  # Winter
        season = "Winter"
        defaults = {
            "MinTemp": 8.0,
            "MaxTemp": 17.0,
            "Rainfall": 5.0,
            "Evaporation": 2.0,
            "Sunshine": 6.0,
            "WindGustSpeed": 35.0,
            "WindSpeed9am": 10.0,
            "WindSpeed3pm": 15.0,
            "Humidity9am": 75.0,
            "Humidity3pm": 60.0,
            "Pressure9am": 1020.0,
            "Pressure3pm": 1018.0,
            "Cloud9am": 5.0,
            "Cloud3pm": 6.0,
            "Temp9am": 12.0,
            "Temp3pm": 16.0,
            "RainToday": 0,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
    else:  # Spring (9, 10, 11)
        season = "Spring"
        defaults = {
            "MinTemp": 12.0,
            "MaxTemp": 22.0,
            "Rainfall": 3.0,
            "Evaporation": 6.0,
            "Sunshine": 8.0,
            "WindGustSpeed": 38.0,
            "WindSpeed9am": 13.0,
            "WindSpeed3pm": 19.0,
            "Humidity9am": 68.0,
            "Humidity3pm": 52.0,
            "Pressure9am": 1016.0,
            "Pressure3pm": 1014.0,
            "Cloud9am": 4.0,
            "Cloud3pm": 4.0,
            "Temp9am": 16.0,
            "Temp3pm": 21.0,
            "RainToday": 0,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }

    result = f"Season: {season} ({month_names[month-1]})\n\nTypical weather conditions for {season} in Australia:\n"
    for param, value in defaults.items():
        if param in ["month_sin", "month_cos"]:
            result += f"  {param}: {value:.6f}\n"
        else:
            result += f"  {param}: {value}\n"

    result += "\nNote: month_sin and month_cos are cyclical encodings of the month (preserves seasonal periodicity)."
    return result


@tool
def lookup_location(city: str) -> str:
    """
    Looks up coordinates for an Australian city.
    Use this tool FIRST to validate and get the coordinates for a location before making predictions.

    Args:
        city: Name of the city in Australia (e.g., "Sydney", "Melbourne", "Brisbane")

    Returns:
        A string with the city name and coordinates if found, or list of available cities if not found.
    """
    # Case-insensitive lookup
    city_map = {k.lower(): (k, v) for k, v in city_coords.items()}
    city_lower = city.lower().strip()

    if city_lower in city_map:
        original_name, coords = city_map[city_lower]
        lat, lon = coords
        return f"Location found: {original_name} at coordinates (Latitude: {lat:.4f}, Longitude: {lon:.4f})"
    else:
        # Find close matches
        matches = [name for name in city_coords.keys() if city_lower in name.lower()]
        if matches:
            return f"Location '{city}' not found. Did you mean one of these? {', '.join(matches[:10])}"
        else:
            available = ", ".join(list(city_coords.keys())[:20])
            return f"Location '{city}' not found. Available cities include: {available}... (and {len(city_coords) - 20} more)"


@tool
def predict_solar_output(
    Latitude: float = None,
    Longitude: float = None,
    MinTemp: float = None,
    MaxTemp: float = None,
    Rainfall: float = None,
    Evaporation: float = None,
    Sunshine: float = None,
    WindGustSpeed: float = None,
    WindSpeed9am: float = None,
    WindSpeed3pm: float = None,
    Humidity9am: float = None,
    Humidity3pm: float = None,
    Pressure9am: float = None,
    Pressure3pm: float = None,
    Cloud9am: float = None,
    Cloud3pm: float = None,
    Temp9am: float = None,
    Temp3pm: float = None,
    RainToday: int = None,
    month_sin: float = None,
    month_cos: float = None,
) -> str:
    """
    Predicts daily solar PV output (kWh/kWp) based on weather conditions using XGBoost model.

    Args:
        Latitude: Location latitude (use lookup_location tool to get this from city name).
        Longitude: Location longitude (use lookup_location tool to get this from city name).
        MinTemp: Minimum temperature (°C) during a particular day.
        MaxTemp: Maximum temperature (°C) during a particular day.
        Rainfall: Precipitation (mm) during a particular day.
        Evaporation: Class A pan evaporation (mm) during a particular day.
        Sunshine: Number of hours of bright sunshine.
        WindGustSpeed: The speed (km/h) of the strongest wind gust during a particular day.
        WindSpeed9am: Wind speed (km/h) averaged over 10 minutes prior to 9am.
        WindSpeed3pm: Wind speed (km/h) averaged over 10 minutes prior to 3pm.
        Humidity9am: Humidity (percent) at 9am.
        Humidity3pm: Humidity (percent) at 3pm.
        Pressure9am: Atmospheric pressure (hPa) reduced to mean sea level at 9am.
        Pressure3pm: Atmospheric pressure (hPa) reduced to mean sea level at 3pm.
        Cloud9am: Fraction of sky obscured by cloud at 9am (0-8 scale).
        Cloud3pm: Fraction of sky obscured by cloud at 3pm (0-8 scale).
        Temp9am: Temperature (°C) at 9am.
        Temp3pm: Temperature (°C) at 3pm.
        RainToday:  If today is rainy then 1 (Yes). If today is not rainy then 0 (No).
        month_sin: Cyclical encoding of month: sin(2π × month / 12). Captures seasonal patterns.
        month_cos: Cyclical encoding of month: cos(2π × month / 12). Captures seasonal patterns.

    Returns:
        JSON string containing XGBoost prediction (kWh/kWp) and the input parameters used.
    """
    # Lazily load resources if not loaded
    if not resources.loaded:
        resources.load()

    # Create a dictionary of inputs
    inputs = {
        "Latitude": Latitude,
        "Longitude": Longitude,
        "MinTemp": MinTemp,
        "MaxTemp": MaxTemp,
        "Rainfall": Rainfall,
        "Evaporation": Evaporation,
        "Sunshine": Sunshine,
        "WindGustSpeed": WindGustSpeed,
        "WindSpeed9am": WindSpeed9am,
        "WindSpeed3pm": WindSpeed3pm,
        "Humidity9am": Humidity9am,
        "Humidity3pm": Humidity3pm,
        "Pressure9am": Pressure9am,
        "Pressure3pm": Pressure3pm,
        "Cloud9am": Cloud9am,
        "Cloud3pm": Cloud3pm,
        "Temp9am": Temp9am,
        "Temp3pm": Temp3pm,
        "RainToday": RainToday,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }

    # Check for missing required values
    missing_params = [
        col for col in resources.feature_columns if inputs.get(col) is None
    ]
    if missing_params:
        return str(
            {
                "error": f"Missing required parameters: {', '.join(missing_params)}. Please use get_seasonal_weather_defaults tool to get complete weather parameters."
            }
        )

    # Create DataFrame for prediction
    X_input = pd.DataFrame([inputs], columns=resources.feature_columns)

    results = {}

    if resources.xgb_model:
        try:
            xgb_pred = resources.xgb_model.predict(X_input)[0]
            results["prediction_kWh_kWp"] = round(xgb_pred, 3)
        except Exception as e:
            results["error"] = str(e)
    else:
        results["error"] = "XGBoost model not available"

    results["input_parameters"] = inputs

    return str(results)
