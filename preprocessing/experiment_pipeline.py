import numpy as np
import pandas as pd


def column_transformation(df):
    # Shrinking column values
    df["log_trip_duration"] = np.log1p(df["trip_duration"])
    df.drop("trip_duration", axis=1, inplace=True)
    return df


def fix_datatypes(df):
    # Fixing Data Types
    df['vendor_id'] = df['vendor_id'].astype('int')
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].astype('category')
    df['passenger_count'] = df['passenger_count'].astype('int')
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    return df


def clean_numeric_outliers(df, col, train_iqr=-1):
    MULTIPLIER = 1.5

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1 if train_iqr == -1 else train_iqr

    lower_bound = q1 - MULTIPLIER * iqr
    upper_bound = q3 + MULTIPLIER * iqr
    
    df_no_outliers = df[df[col].between(lower_bound, upper_bound)]
    return df_no_outliers, iqr


def clean_outliers(df):
    df = df[(df['passenger_count'] != 7) & (df['passenger_count'] != 0)]

    df = df[df["dropoff_latitude"].between(40, 43)]
    df = df[df["pickup_latitude"].between(40, 43)]
    df = df[df["pickup_longitude"].between(-75, -73)]
    df = df[df["dropoff_longitude"].between(-75, -73)]
    
    # Blizzard Anomaly
    df = df[~(df["pickup_datetime"].between('2016-01-22', '2016-01-25'))]
    
    return df


def engineer_feature(df):
    df['requires_large_vehicle'] = ((df['passenger_count'] == 5) | (df['passenger_count'] == 6)).astype("int")

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
    df["trip_distance_sqrt"] = np.sqrt(np.sqrt(x**2 + y**2))
    df["trip_distance_square"] = x**2 + y**2
    df["trip_distance_cube"] = (np.sqrt(x**2 + y**2))**3
    
    df["log_trip_distance"] = np.log1p(np.sqrt(x**2 + y**2))
    df["log_trip_distance_sqrt"] = np.log1p(np.sqrt(np.sqrt(x**2 + y**2)))
    df["log_trip_distance_square"] = np.log1p(x**2 + y**2)
    df["log_trip_distance_cube"] = np.log1p((np.sqrt(x**2 + y**2))**3)


    # Coordinates are taken from Google Maps
    JFK_LATITUDE_RANGE = [40.620998, 40.683139]
    JFK_LONGITUDE_RANGE = [-73.841476, -73.729188]

    LG_LATITUDE_RANGE = [40.763557, 40.787499]
    LG_LONGITUDE_RANGE = [-73.899899, -73.848085]

    # JFK bounding box
    df["is_jfk_airport"] = (
        ((df["pickup_latitude"].between(JFK_LATITUDE_RANGE[0], JFK_LATITUDE_RANGE[1])) &
        (df["pickup_longitude"].between(JFK_LONGITUDE_RANGE[0], JFK_LONGITUDE_RANGE[1])))
        |
        ((df["dropoff_latitude"].between(JFK_LATITUDE_RANGE[0], JFK_LATITUDE_RANGE[1])) &
        (df["dropoff_longitude"].between(JFK_LONGITUDE_RANGE[0], JFK_LONGITUDE_RANGE[1])))
    ).astype("int")

    # LaGuardia bounding box
    df["is_lg_airport"] = (
        ((df["pickup_latitude"].between(LG_LATITUDE_RANGE[0], LG_LATITUDE_RANGE[1])) &
        (df["pickup_longitude"].between(LG_LONGITUDE_RANGE[0], LG_LONGITUDE_RANGE[1])))
        |
        ((df["dropoff_latitude"].between(LG_LATITUDE_RANGE[0], LG_LATITUDE_RANGE[1])) &
        (df["dropoff_longitude"].between(LG_LONGITUDE_RANGE[0], LG_LONGITUDE_RANGE[1])))
    ).astype("int")


    from scipy.stats import gmean, hmean

    geo_columns = geo_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

    # Ensure geo_features is a 2D NumPy array (shape: [n_samples, n_features])
    geo_array = df[geo_columns].to_numpy()

    # axis = 1 ensures row wise operations
    df['coord_arithmetic_mean'] = np.mean(geo_array, axis=1)
    df['coord_geometric_mean'] = gmean(np.abs(geo_array), axis=1)
    df['coord_harmonic_mean'] = hmean(np.abs(geo_array), axis=1)
    df['coord_square_sum'] = np.sum(geo_array ** 2, axis=1)


    df['dayofyear'] = df.pickup_datetime.dt.dayofyear
    df['dayofweek'] = df.pickup_datetime.dt.dayofweek
    df['month'] = df.pickup_datetime.dt.month
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour
    df['minute'] = df.pickup_datetime.dt.minute

    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall (September, October, November)

    df['season'] = df['month'].apply(get_season)

    df["is_summer"] = (df["season"] == 2).astype("int")
    df["is_rush_hour"] = ((df["hour"].between(7, 9)) | (df["hour"].between(13, 19))).astype("int")
    df["is_night"] = ((df["hour"] > 1) & (df["hour"] < 6)).astype("int")
    df["is_weekend"] =  ((df["weekday"] // 5) == 1).astype("int")


    BASE_SPEED = 32

    df['virtual_speed'] = BASE_SPEED / (2 ** (
                            (df['is_jfk_airport'] | df["is_lg_airport"]).astype("int") + # cast bool to int
                            (df['is_rush_hour']).astype("int") +
                            (df['is_summer']).astype("int") + 
                            (df['store_and_fwd_flag'] == 'Y').astype("int")
                            ))
    
    df['virtual_time'] = df['log_trip_distance'] / df['virtual_speed']

    # Adding the cubes
    df['virtual_speed_cube'] = df['virtual_speed'] ** 3
    df['virtual_time_cube'] = df['virtual_time'] ** 3
    df["virtual_time_dist_sqrt"] = df['trip_distance_sqrt'] / df['virtual_speed']
    
    return df


def drop_cols(df):
    def drop_col(df, col):
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
        else:
            print(f"[Warning] Column not found: {col}")
        return df

    cols_to_drop = [ 
                        'pickup_datetime', 'id',
                        'store_and_fwd_flag', # 'vendor_id', 'passenger_count', 
                        'coord_geometric_mean', # 'coord_square_sum', 'coord_arithmetic_mean', 'coord_harmonic_mean',
                        'dropoff_latitude', 'dropoff_longitude', 'pickup_latitude', 'pickup_longitude', 
                        'is_weekend', 'is_rush_hour', 'is_summer', 'is_night','requires_large_vehicle',
                        'dayofyear', 'dayofweek', # 'hour', 'season', 'weekday', 'month', 
                        # 'virtual_speed', 'virtual_speed_cube',
                        'virtual_time_cube', # 'virtual_time', 'virtual_time_dist_sqrt',
                        # 'trip_distance_sqrt', 'trip_distance_square', 'trip_distance_cube', 'trip_distance',
                        # 'log_trip_distance_sqrt', 'log_trip_distance_square', 'log_trip_distance_cube', 'log_trip_distance',
                        # 'is_jfk_airport', 'is_lg_airport',
                     ] 

    for col in cols_to_drop:
        df = drop_col(df, col)

    return df


def preprocessing_pipeline(df: pd.DataFrame, iqr=-1):
    print("Preprocessing started...")
    print(f"Initial shape: {df.shape}")

    print("Replacing Numerical Values...")
    df = fix_datatypes(df)

    print("Doing column transformation...")
    df = column_transformation(df)

    df = clean_outliers(df)
    df, iqr = clean_numeric_outliers(df, "log_trip_duration", iqr)
    print(f"After cleaning outliers: {df.shape}")

    print("Feature Engineering...")
    df = engineer_feature(df)  

    print("Dropping columns...")
    df = drop_cols(df)

    print("Final shape:", df.shape, "\n")
    # print(df.columns)

    return df, iqr