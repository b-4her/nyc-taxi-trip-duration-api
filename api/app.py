from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Literal

import traceback
import pandas as pd
import numpy as np
import re
import joblib

import os, sys
sys.path.append(os.path.abspath('../scripts'))

from saved_models_evaluator import load_model

MODEL_PATH = '../models/final_ridge_pipeline.pkl'


app = FastAPI()


class TripInput(BaseModel):
    store_and_fwd_flag: Literal['Y', 'N'] = Field(title="Store and Forward Flag ('Y' or 'N')")

    @field_validator('store_and_fwd_flag', mode='before')
    @classmethod
    def capitalize_flag(cls, v: str) -> str:
        return v.upper()
    
    vendor_id: int = Field(
        title="Vendor ID (1 or 2)",
        gt=0,
        lt=3
    )

    passenger_count: int = Field(
        title="Number of Passengers (1 to 6)",
        gt=0,
        lt=7
    )

    pickup_longitude: float = Field(
        title="Pickup Longitude (-75 to -73)",
        gt=-75,
        lt=-73
    )

    pickup_latitude: float = Field(
        title="Pickup Latitude (40 to 43)",
        gt=40,
        lt=43
    )

    dropoff_longitude: float = Field(
        title="Dropoff Longitude (-75 to -73)",
        gt=-75,
        lt=-73
    )

    dropoff_latitude: float = Field(
        title="Dropoff Latitude (40 to 43)",
        gt=40,
        lt=43
    )

    pickup_date: str = Field(
        title="Pickup Date (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )

    pickup_time: str = Field(
        title="Pickup Time (HH:MM in 24-hour format)",
        pattern=r"^([01]\d|2[0-3]):([0-5]\d)$"
    )

@app.post("/predict")
def predict(trip_data: TripInput):
    """
    Predict the taxi trip duration in seconds based on user-provided trip details.

    Parameters:
        trip_data (TripInput): Input data including vendor ID, passenger count,
                               pickup/dropoff coordinates, datetime info, etc.

    Returns:
        JSON response containing the predicted trip duration.
    """
    try:
        # parse data into df
        trip = pd.DataFrame([trip_data.model_dump()])

        # create datetime feature
        trip["pickup_datetime"] = trip["pickup_date"] + " " + trip["pickup_time"] + ":00"
        trip["pickup_datetime"] = pd.to_datetime(trip["pickup_datetime"])

        # Using distance formula:
        # https://www.chegg.com/homework-help/questions-and-answers/point-latitude-373198-point-longitude-121936-point-b-latitude-373185-point-b-longitude-121-q56508606

        R = 6356  # radius of Earth in km

        # Convert degrees to radians
        lat1 = np.radians(trip["pickup_latitude"])
        lat2 = np.radians(trip["dropoff_latitude"])
        lon1 = np.radians(trip["pickup_longitude"])
        lon2 = np.radians(trip["dropoff_longitude"])

        # x and y components of distance
        x = R * (lat1 - lat2)
        y = R * (lon1 - lon2) * np.cos(lat2)

        # Euclidean distance approximation
        trip["trip_distance"] = np.sqrt(x**2 + y**2)
        trip["trip_distance_sqrt"] = np.sqrt(np.sqrt(x**2 + y**2))
        trip["trip_distance_square"] = x**2 + y**2
        trip["trip_distance_cube"] = (np.sqrt(x**2 + y**2))**3

        trip["log_trip_distance"] = np.log1p(np.sqrt(x**2 + y**2))
        trip["log_trip_distance_sqrt"] = np.log1p(np.sqrt(np.sqrt(x**2 + y**2)))
        trip["log_trip_distance_square"] = np.log1p(x**2 + y**2)
        trip["log_trip_distance_cube"] = np.log1p((np.sqrt(x**2 + y**2))**3)


        # Coordinates are taken from Google Maps
        JFK_LATITUDE_RANGE = [40.620998, 40.683139]
        JFK_LONGITUDE_RANGE = [-73.841476, -73.729188]

        LG_LATITUDE_RANGE = [40.763557, 40.787499]
        LG_LONGITUDE_RANGE = [-73.899899, -73.848085]

        # JFK bounding box
        trip["is_jfk_airport"] = (
            ((trip["pickup_latitude"].between(JFK_LATITUDE_RANGE[0], JFK_LATITUDE_RANGE[1])) &
            (trip["pickup_longitude"].between(JFK_LONGITUDE_RANGE[0], JFK_LONGITUDE_RANGE[1])))
            |
            ((trip["dropoff_latitude"].between(JFK_LATITUDE_RANGE[0], JFK_LATITUDE_RANGE[1])) &
            (trip["dropoff_longitude"].between(JFK_LONGITUDE_RANGE[0], JFK_LONGITUDE_RANGE[1])))
        ).astype("int")

        # LaGuardia bounding box
        trip["is_lg_airport"] = (
            ((trip["pickup_latitude"].between(LG_LATITUDE_RANGE[0], LG_LATITUDE_RANGE[1])) &
            (trip["pickup_longitude"].between(LG_LONGITUDE_RANGE[0], LG_LONGITUDE_RANGE[1])))
            |
            ((trip["dropoff_latitude"].between(LG_LATITUDE_RANGE[0], LG_LATITUDE_RANGE[1])) &
            (trip["dropoff_longitude"].between(LG_LONGITUDE_RANGE[0], LG_LONGITUDE_RANGE[1])))
        ).astype("int")


        from scipy.stats import hmean

        geo_columns = geo_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

        # Ensure geo_features is a 2D NumPy array (shape: [n_samples, n_features])
        geo_array = trip[geo_columns].to_numpy()

        # axis = 1 ensures row wise operations
        trip['coord_arithmetic_mean'] = np.mean(geo_array, axis=1)
        trip['coord_harmonic_mean'] = hmean(np.abs(geo_array), axis=1)
        trip['coord_square_sum'] = np.sum(geo_array ** 2, axis=1)

        trip['month'] = trip.pickup_datetime.dt.month
        trip['weekday'] = trip.pickup_datetime.dt.weekday
        trip['hour'] = trip.pickup_datetime.dt.hour
        trip['minute'] = trip.pickup_datetime.dt.minute


        def get_season(month):
            if month in [12, 1, 2]:
                return 0  # Winter
            elif month in [3, 4, 5]:
                return 1  # Spring
            elif month in [6, 7, 8]:
                return 2  # Summer
            else:
                return 3  # Fall (September, October, November)

        trip['season'] = trip['month'].apply(get_season)


        trip["is_summer"] = (trip["season"] == 2).astype("int")
        trip["is_rush_hour"] = ((trip["hour"].between(7, 9)) | (trip["hour"].between(13, 19))).astype("int")
        trip["is_night"] = ((trip["hour"] > 1) & (trip["hour"] < 6)).astype("int")
        trip["is_weekend"] =  ((trip["weekday"] // 5) == 1).astype("int")


        BASE_SPEED = 32

        trip['virtual_speed'] = BASE_SPEED / (2 ** (
                                (trip['is_jfk_airport'] | trip["is_lg_airport"]).astype("int") + # cast bool to int
                                (trip['is_rush_hour']).astype("int") +
                                (trip['is_summer']).astype("int") + 
                                (trip['store_and_fwd_flag'] == 'Y').astype("int")
                                ))

        trip['virtual_time'] = trip['log_trip_distance'] / trip['virtual_speed']

        # Adding the cubes
        trip["virtual_time_dist_sqrt"] = trip['trip_distance_sqrt'] / trip['virtual_speed']


        def drop_col(df, col):
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
            else:
                print(f"[Warning] Column not found: {col}")
            return df

        cols_to_drop = [ 
                            'pickup_time', 'pickup_date', 'pickup_datetime', 'store_and_fwd_flag',
                            'dropoff_latitude', 'dropoff_longitude', 'pickup_latitude', 'pickup_longitude', 
                            'is_weekend', 'is_rush_hour', 'is_summer', 'is_night', 'virtual_speed', 
                        ] 

        for col in cols_to_drop:
            trip = drop_col(trip, col)


        model_pipeline, _ = load_model(MODEL_PATH)
        log_trip_duration = model_pipeline.predict(trip)[0]
        trip_duration = np.expm1(log_trip_duration).round()

        return {"trip_duration": trip_duration}
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    

class UncheckedTripInput(BaseModel):
    store_and_fwd_flag: str = Field(
        title="Store and Forward Flag ('Y' or 'N')",
    )
    
    vendor_id: int = Field(
        title="Vendor ID (1 or 2)",
    )

    passenger_count: int = Field(
        title="Number of Passengers (1 to 6)",
    )

    pickup_longitude: float = Field(
        title="Pickup Longitude (-75 to -73)",
    )

    pickup_latitude: float = Field(
        title="Pickup Latitude (40 to 43)",
    )

    dropoff_longitude: float = Field(
        title="Dropoff Longitude (-75 to -73)",
    )

    dropoff_latitude: float = Field(
        title="Dropoff Latitude (40 to 43)",
    )

    pickup_date: str = Field(
        title="Pickup Date (YYYY-MM-DD)",
    )

    pickup_time: str = Field(
        title="Pickup Time (HH:MM in 24-hour format)",
    )

@app.post("/validate")
def validate(trip_data :UncheckedTripInput):
    '''
    Validate the input trip data without performing prediction.

    This endpoint checks if the provided input conforms to the expected
    structure and value constraints. It is useful for debugging or confirming
    input validity before using the prediction endpoint.

    Args:
        trip_data (UncheckedTripInput): Raw trip data from the user.

    Returns:
        dict: A success message if all checks pass, or detailed errors if not.
    '''
    try:
        trip = pd.DataFrame([trip_data.model_dump()])
        errors = {}

        # Store and forward flag
        flag = trip["store_and_fwd_flag"][0].strip().upper()
        if flag not in {"Y", "N"}:
            errors["store_and_fwd_flag"] = "Value must be 'Y' or 'N' (case-insensitive)."

        # Vendor ID
        vendor_id = trip["vendor_id"][0]
        if vendor_id not in {1, 2}:
            errors["vendor_id"] = "Vendor ID must be either 1 or 2."

        # Passenger count
        passenger_count = trip["passenger_count"][0]
        if not (1 <= passenger_count <= 6):
            errors["passenger_count"] = "Passenger count must be between 1 and 6."

        # Coordinate validations
        coords = {
            "pickup_longitude": (-75, -73),
            "dropoff_longitude": (-75, -73),
            "pickup_latitude": (40, 43),
            "dropoff_latitude": (40, 43),
        }

        for key, (low, high) in coords.items():
            val = trip[key][0]
            if not (low <= val <= high):
                label = key.replace('_', ' ').capitalize()
                errors[key] = f"{label} must be within the expected range for NYC: between {low} and {high}."

        # Pickup date format (YYYY-MM-DD)
        pickup_date = trip["pickup_date"][0]
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", pickup_date):
            errors["pickup_date"] = "Pickup date must follow the format YYYY-MM-DD."

        # Pickup time format (HH:MM in 24-hour format)
        pickup_time = trip["pickup_time"][0]
        if not re.match(r"^([01]\d|2[0-3]):([0-5]\d)$", pickup_time):
            errors["pickup_time"] = "Pickup time must follow the format HH:MM (24-hour clock)."

        if not errors:
            return {"success": "Input is valid and ready for prediction."}
        else:
            return {"errors": errors}
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/features")
def get_features():
    """
    Returns a dictionary describing the expected features and their descriptions
    required for making a trip duration prediction.
    """
    features = {
        "features": {
            "store_and_fwd_flag": "Whether the trip record was stored and forwarded ('Y' or 'N')",
            "vendor_id": "ID of the vendor (1 or 2)",
            "passenger_count": "Number of passengers (1 to 6)",
            "pickup_longitude": "Longitude where the trip started (-75 to -73)",
            "pickup_latitude": "Latitude where the trip started (40 to 43)",
            "dropoff_longitude": "Longitude where the trip ended (-75 to -73)",
            "dropoff_latitude": "Latitude where the trip ended (40 to 43)",
            "pickup_date": "Date of the pickup in 'YYYY-MM-DD' format",
            "pickup_time": "Time of the pickup in 'HH:MM' 24-hour format"
        }
    }

    return features


@app.get("/features/sample")
def get_sample():
    '''
    Returns a sample input dictionary to guide users on the expected input format.
    '''
    sample = {
        "sample": {
            "vendor_id": 1,
            "passenger_count": 1,
            "pickup_longitude": -73.988609,
            "pickup_latitude": 40.748977,
            "dropoff_longitude": -73.992797,
            "dropoff_latitude": 40.763408,
            "pickup_date": "2016-03-23",
            "pickup_time": "02:24",
            "store_and_fwd_flag": "N"
        }
    }
    return sample


@app.get("/about")
def get_about():
    """
    Returns a general description of the model, its purpose, inputs, and outputs.
    """
    return {
        "about": (
            "This API predicts the duration (in seconds) of NYC taxi trips using a Ridge Regression model.\n"
            "It focuses on feature engineering to improve prediction quality.\n"
            "The model predicts log-transformed trip duration (`log_trip_duration`),\n"
            "which is later exponentiated to return the actual trip time in seconds.\n"
            "Predictions are based on trip metadata such as:\n"
            "- Vendor ID\n"
            "- Pickup/Dropoff coordinates\n"
            "- Passenger count\n"
            "- Pickup date and time\n"
            "\n"
            "The features used can be viewed at the `/features` endpoint,\n"
            "and a valid sample input is available at `/features/sample`.\n"
            "The output is a single integer representing the estimated trip duration in seconds.\n"
            "\n"
            "This project was developed by Baher Alabbar\n"
            "as part of a learning initiative to understand end-to-end deployment of ML models."
        )
    }


@app.get("/version")
def get_version():
    """
    Returns version information and model performance details.

    NOTE: These values must be updated manually if a new model is trained or deployed.
    """
    return {
        "model_type": "Ridge Regression",
        "alpha": 1,
        "model_path": MODEL_PATH,
        "train_rmse": 0.3931,
        "train_r2": 0.6946,
        "val_rmse": 0.3930,
        "val_r2": 0.6949,
        "target_variable": "log_trip_duration (converted back to seconds)",
    }


@app.get("/help")
def get_help():
    """
    Returns a summary of all available API endpoints with short descriptions.
    Helpful for quickly navigating the available routes.

    NOTE: Update this manually if new endpoints are added or removed.
    """
    return {
        "endpoints": [
            {
                "method": "POST",
                "endpoint": "/predict",
                "description": "Predicts trip duration based on user-provided trip features."
            },
            {
                "method": "POST",
                "endpoint": "/validate",
                "description": "Validates a user input JSON against the expected input schema."
            },
            {
                "method": "GET",
                "endpoint": "/features",
                "description": "Returns a dictionary of required input features for prediction."
            },
            {
                "method": "GET",
                "endpoint": "/features/sample",
                "description": "Returns a sample input dictionary for guidance."
            },
            {
                "method": "GET",
                "endpoint": "/about",
                "description": "Provides general information about the model and its purpose."
            },
            {
                "method": "GET",
                "endpoint": "/version",
                "description": "Returns version details of the model, API, and performance metrics."
            },
            {
                "method": "GET",
                "endpoint": "/help",
                "description": "Returns a list of all available API endpoints with descriptions."
            },
        ]
    }