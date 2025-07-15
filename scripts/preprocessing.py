import numpy as np
import pandas as pd


def column_transformation(df):
    # Shrinking columns values
    df["trip_duration"] = np.log1p(df["trip_duration"]) 
    df["trip_distance"] = np.log1p(df["trip_distance"])

    return df


def replace_numerical_by_categorical(df):
    # Casting to object
    df["vendor_id"] = df["vendor_id"].astype("object") 
    
    return df


def clean_outliers(df, train_iqr=-1):
    df = df[(df["passenger_count"] != 0)]

    def clean_outliers(df, col):
        MULTIPLIER = 5

        if train_iqr == -1:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
        else:
            iqr = train_iqr
        
        # You should not make the outliers range so narrow so that you don't drop much data.
        lower_bound = q1 - MULTIPLIER * iqr
        upper_bound = q3 + MULTIPLIER * iqr
        
        df_no_outliers = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df_no_outliers
    
    df = clean_outliers(df, "trip_distance")

    return df


def engineer_feature(df):
    # Using distance formula:
    # https://www.chegg.com/homework-help/questions-and-answers/point-latitude-373198-point-longitude-121936-point-b-latitude-373185-point-b-longitude-121-q56508606

    R = 6356  # radius of Earth in km

    # Convert degrees to radians
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    # x and y components of distance
    x = R * (lat1 - lat2)
    y = R * (lon1 - lon2) * np.cos(lat2)

    # Euclidean distance approximation
    df["trip_distance"] = np.sqrt(x**2 + y**2)

    # converting datetime into 4 new features
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dayofweek'] = df.pickup_datetime.dt.dayofweek
    df['hour'] = df.pickup_datetime.dt.hour
    df['month'] = df.pickup_datetime.dt.month

    # Some summary stats for coordinates
    df["coord_sum"] = df["dropoff_latitude"] + df["dropoff_longitude"] + df["pickup_latitude"] + df["pickup_longitude"]
    df["coord_square_sum"] = df["dropoff_latitude"]**2 + df["dropoff_longitude"]**2 + df["pickup_latitude"]**2 + df["pickup_longitude"]**2

    df["isNight"] = df["hour"] // 18
    df["isWeekend"] = df.pickup_datetime.dt.weekday // 5

    return df


def drop_cols(df):
    df.drop("id", axis=1, inplace=True)  # dropping id

    # # dropping coordinates
    # df.drop("dropoff_latitude", axis=1, inplace=True)
    # df.drop("dropoff_longitude", axis=1, inplace=True)
    # df.drop("pickup_latitude", axis=1, inplace=True)
    # df.drop("pickup_longitude", axis=1, inplace=True)

    # dropping datetime object
    df.drop("pickup_datetime", axis=1, inplace=True)

    return df


def preprocessing_pipeline(df: pd.DataFrame):
    print("Preprocessing started...")
    print(f"Initial shape: {df.shape}")

    print("Replacing Numerical Values...")
    df = replace_numerical_by_categorical(df)

    print("Feature Engineering...")
    df = engineer_feature(df)

    print("Doing column transformation...")
    df = column_transformation(df)

    df = clean_outliers(df)
    print(f"After cleaning outliers: {df.shape}")

    print("Dropping columns...")
    df = drop_cols(df)

    print("Final shape:", df.shape, "\n")

    return df