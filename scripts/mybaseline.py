import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from preprocessing import preprocessing_pipeline

def main():
    train = pd.read_csv('../data/split_sample/train.csv')
    val = pd.read_csv('../data/split_sample/val.csv')

    train = preprocessing_pipeline(train)
    val = preprocessing_pipeline(val)

    # One hot encoding
    one_hot_columns = ['store_and_fwd_flag', 'vendor_id']
    train = pd.get_dummies(train, columns=one_hot_columns)
    val = pd.get_dummies(val, columns=one_hot_columns)

    # Separating target
    train_target = train["trip_duration"]
    val_target = val["trip_duration"]

    train.drop("trip_duration", axis=1, inplace=True)
    val.drop("trip_duration", axis=1, inplace=True)


    # Scaling
    scaler = StandardScaler()

    scaler.fit_transform(train)

    print(train.head())



if __name__ == "__main__":
    main()