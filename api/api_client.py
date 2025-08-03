import requests
import json
import os

class TripDurationPredictor:
    '''
    API Client for the NYC Taxi Trip Duration Prediction service.

    This class provides methods to interact with all supported API endpoints:
    - `predict()`: Make a single prediction using required trip features.
    - `predict_batch()`: Make predictions for a batch of trips using JSON input (from file or string).
    - `validate()`: Validate a user input dictionary against the expected schema.
    - `get_features()`: Retrieve a list of required input features.
    - `get_sample_features()`: Get a sample input dictionary for guidance.
    - `get_about()`: Return information about the model and how predictions are made.
    - `get_version()`: Get version details of the model, API, and key libraries.
    - `get_help()`: Get an overview of all available API endpoints.

    Attributes:
        BASE_URL (str): The base URL of the prediction API. Defaults to 'http://127.0.0.1:8000' or the API_URL environment variable.

    Example:
        client = TripDurationPredictor()
        prediction = client.predict(2, 1, -73.99, 40.75, -73.97, 40.74, "2024-08-02", "08:00:00", "N")
        print(prediction)
    '''

    def __init__(self, base_url=None):
        self.BASE_URL = base_url or os.getenv("API_URL", "http://127.0.0.1:8000")

    def _get(self, endpoint):
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint, payload):
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def predict(self, vendor_id, passenger_count, 
                pickup_longitude, pickup_latitude, 
                dropoff_longitude, dropoff_latitude, 
                pickup_date, pickup_time, store_and_fwd_flag):
        
        payload = {
            "vendor_id": vendor_id,
            "passenger_count": passenger_count,
            "pickup_longitude": pickup_longitude,
            "pickup_latitude": pickup_latitude,
            "dropoff_longitude": dropoff_longitude,
            "dropoff_latitude": dropoff_latitude,
            "pickup_date": pickup_date,
            "pickup_time": pickup_time,
            "store_and_fwd_flag": store_and_fwd_flag,
        }

        return self._post("predict", payload)

    def validate(self, **kwargs):
        required = [
            "vendor_id", "passenger_count", 
            "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude",
            "pickup_date", "pickup_time", 
            "store_and_fwd_flag"
        ]
        missing = [key for key in required if key not in kwargs]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        return self._post("validate", kwargs)

    def predict_batch(self, input_json):
        """
        Accepts a path to a JSON file or a raw JSON string (list of objects).
        """
        try:
            if os.path.exists(input_json):
                with open(input_json, "r") as f:
                    batch_data = json.load(f)
            else:
                batch_data = json.loads(input_json)
        except Exception as e:
            raise ValueError(f"Failed to load batch input: {e}")

        return self._post("predict/batch", batch_data)

    def get_features(self):
        return self._get("features")

    def get_sample_features(self):
        return self._get("features/sample")

    def get_about(self):
        return self._get("about")

    def get_version(self):
        return self._get("version")

    def get_help(self):
        return self._get("help")