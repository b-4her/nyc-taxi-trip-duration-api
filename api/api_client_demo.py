from api_client import TripDurationPredictor
import json

def main():
    client = TripDurationPredictor()

    # GET request
    help = client.get_help()
    print(json.dumps(help, indent=4))

    # Single prediction (POST request)
    result = client.predict(
        vendor_id=2,
        passenger_count=1,
        pickup_longitude=-73.993,
        pickup_latitude=40.751,
        dropoff_longitude=-73.975,
        dropoff_latitude=40.749,
        pickup_date="2024-08-02",
        pickup_time="08:00",
        store_and_fwd_flag="N"
    )
    print(json.dumps(result, indent=4))

    # Batch prediction from file or string
    batch = client.predict_batch("sample_batch.json")  # or raw JSON string
    print(json.dumps(batch, indent=4))

if __name__ == "__main__":
    main()