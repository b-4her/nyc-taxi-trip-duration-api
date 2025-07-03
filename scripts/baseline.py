import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from preprocessing import preprocessing_pipeline


def main():
    train = pd.read_csv('../data/split_sample/train.csv')
    val = pd.read_csv('../data/split_sample/val.csv')

    train = preprocessing_pipeline(train)
    val = preprocessing_pipeline(val)

    # Separating target
    train_target = train["trip_duration"]
    val_target = val["trip_duration"]
    train.drop("trip_duration", axis=1, inplace=True)
    val.drop("trip_duration", axis=1, inplace=True)

    approach1(train, val, train_target, val_target)


def approach1(train, val, train_target, val_target):
    # encoding 
    categorical_features = ['store_and_fwd_flag', 'vendor_id', 'dayofweek',  'month',  'hour',  'dayofyear']
    # scaling
    numeric_features = ['trip_distance']

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
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


def predict_eval(model, train, train_target, name):
    y_train_pred = model.predict(train)
    rmse = mean_squared_error(train_target, y_train_pred)
    r2 = r2_score(train_target, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


if __name__ == "__main__":
    main()