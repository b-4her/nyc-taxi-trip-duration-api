import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from preprocessing import preprocessing_pipeline


def main():
    train = pd.read_csv('../data/split/train.csv')
    val = pd.read_csv('../data/split/val.csv')

    train, iqr = preprocessing_pipeline(train)
    val, _ = preprocessing_pipeline(val, iqr)

    # Separating target
    train_target = train["log_trip_duration"]
    val_target = val["log_trip_duration"]
    train.drop("log_trip_duration", axis=1, inplace=True)
    val.drop("log_trip_duration", axis=1, inplace=True)

    approach1(train, val, train_target, val_target)
    # approach2(train, val, train_target, val_target)
    


def approach1(train, val, train_target, val_target):
    # encoding 
    categorical_features = ['hour', 'season', 'weekday', 'vendor_id', 'passenger_count', 'month']
    # scaling
    numeric_features = []
    
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
        ]
        , remainder = 'passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('regression', Ridge())
    ])

    # To pickle
    model = pipeline.fit(train, train_target)
    predict_eval(model, train, train_target, "train")
    predict_eval(model, val, val_target, "validation")


# using polynomial featuring
def approach2(train, val, train_target, val_target):
    # Define feature groups
    categorical_features = ['store_and_fwd_flag', 'vendor_id', 'dayofweek', 'month', "passenger_count", 'hour', 'isNight']
    poly_features = ['dropoff_latitude', 'dropoff_longitude', 'pickup_latitude', 'pickup_longitude']
    numeric_features = ['trip_distance']

    # Preprocessing transformers
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()
    poly_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    # Column transformer
    column_transformer = ColumnTransformer([
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features),
        ('poly', poly_transformer, poly_features)
    ], remainder='drop')  # drop unused columns

    # Final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('regression', Ridge())
    ])

    # To pickle
    model = pipeline.fit(train, train_target)
    predict_eval(model, train, train_target, "train")
    predict_eval(model, val, val_target, "validation")


def predict_eval(model, train, train_target, name):
    y_train_pred = model.predict(train)
    rmse = root_mean_squared_error(train_target, y_train_pred)
    r2 = r2_score(train_target, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


if __name__ == "__main__":
    main()