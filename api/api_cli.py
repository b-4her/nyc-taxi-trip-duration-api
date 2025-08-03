import requests
import argparse
import os, sys
import json

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")  # FastAPI must be running  

def main():
    parser = argparse.ArgumentParser(description="CLI for Taxi Trip API")

    parser.add_argument(
        "--endpoint", 
        choices=[
            "predict", 
            "predict/batch", 
            "validate", 
            "features", 
            "features/sample", 
            "about", 
            "version", 
            "help"
        ], 
        default="help", 
        help="API endpoint to call (default: 'help')"
    )

    # Optional positional arguments (required only for predict/validate)
    parser.add_argument("vendor_id", type=int, nargs="?", help="Vendor ID (e.g., 1 or 2)")
    parser.add_argument("passenger_count", type=int, nargs="?", help="Number of passengers in the taxi (integer)")
    parser.add_argument("pickup_longitude", type=float, nargs="?", help="Longitude of pickup location (decimal degrees)")
    parser.add_argument("pickup_latitude", type=float, nargs="?", help="Latitude of pickup location (decimal degrees)")
    parser.add_argument("dropoff_longitude", type=float, nargs="?", help="Longitude of dropoff location (decimal degrees)")
    parser.add_argument("dropoff_latitude", type=float, nargs="?", help="Latitude of dropoff location (decimal degrees)")
    parser.add_argument("pickup_date", type=str, nargs="?", help="Pickup date in YYYY-MM-DD format")
    parser.add_argument("pickup_time", type=str, nargs="?", help="Pickup time in HH:MM:SS format")
    parser.add_argument("store_and_fwd_flag", type=str, nargs="?", help="Store and forward flag ('Y' or 'N').")
    parser.add_argument("--input-json",type=str, help="Batch input as a JSON file path or raw JSON string (used only with 'predict/batch')")
  
    args = parser.parse_args() # activates the parser

    request_url = BASE_URL + "/" + args.endpoint  

    # validate entered arguments
    if args.endpoint in ["validate", "predict"]:
        required_args = ["vendor_id", "passenger_count", 
                         "pickup_longitude", "pickup_latitude",
                         "dropoff_longitude", "dropoff_latitude",
                         "pickup_date", "pickup_time",
                         "store_and_fwd_flag"]

        # if a required arg is not given its added to the list
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]

        if missing_args: # if not empty
            print(f"Error: Missing required arguments for '{args.endpoint}' endpoint: {', '.join(missing_args)}")
            sys.exit(1)

    try:
        match args.endpoint:
            case "predict" | "validate":  # post methods
                request_body = {
                    "vendor_id": args.vendor_id,
                    "passenger_count": args.passenger_count,
                    "pickup_longitude": args.pickup_longitude,
                    "pickup_latitude": args.pickup_latitude,
                    "dropoff_longitude": args.dropoff_longitude,
                    "dropoff_latitude": args.dropoff_latitude,
                    "pickup_date": args.pickup_date,
                    "pickup_time": args.pickup_time,
                    "store_and_fwd_flag": args.store_and_fwd_flag,
                }
                response = requests.post(request_url, json=request_body)
                print(json.dumps(response.json(), indent=4))

            case "features" | "features/sample" | "version" | "help":  # get methods
                response = requests.get(request_url)
                print(json.dumps(response.json(), indent=4))

            case "predict/batch":  # POST method with batch input from JSON file or raw string
                if not args.input_json:
                    raise ValueError("The --input_json argument is required for 'predict/batch'")

                try:
                    # Try to read as file path
                    if os.path.exists(args.input_json):
                        with open(args.input_json, "r") as f:
                            batch_data = json.load(f)
                    else:
                        # Assume it's a raw JSON string
                        batch_data = json.loads(args.input_json)
                except Exception as e:
                    raise ValueError(f"Failed to parse batch input: {e}")

                response = requests.post(request_url, json=batch_data)
                print(json.dumps(response.json(), indent=4))

            case "about":  # get method
                # Dump to JSON without escaping newlines
                response = requests.get(request_url)
                json_str = json.dumps(response.json(), indent=4, ensure_ascii=False)

                # Load again and print cleanly to preserve newlines
                parsed = json.loads(json_str)
                print("\n" + parsed["about"] + "\n")  

            case _:
                raise ValueError(f"Invalid endpoint: {args.endpoint}")

    except ConnectionError:
        print("Failed to connect to the server.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
    except requests.exceptions.Timeout:
        print("Request timed out.")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")


if __name__ == "__main__":
    main()