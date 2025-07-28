import pandas as pd
import sys, os
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

sys.path.append(os.path.abspath('../preprocessing/'))

from experiment_pipeline import preprocessing_pipeline
from helper import predict_eval

MODEL_NAME = 'experiment_ridge_pipeline'
SAVE_MODEL = False

def main():
    train = pd.read_csv('../data/split/train.csv')
    train_2 = pd.read_csv('../data/split/val.csv')
    
    val = pd.read_csv('../data/split/test.csv')

    # Combine train and val before testing (stack rows)
    train = pd.concat([train, train_2], ignore_index=True)
 
    # ensure you select the correct pipeline file you want
    train, train_iqr = preprocessing_pipeline(train)
    val, _ = preprocessing_pipeline(val, train_iqr)

    # Separating target
    train_target = train["log_trip_duration"]
    val_target = val["log_trip_duration"]
    train.drop("log_trip_duration", axis=1, inplace=True)
    val.drop("log_trip_duration", axis=1, inplace=True)

    train_model(train, val, train_target, val_target, train_iqr, SAVE_MODEL) 
    

def train_model(train, val, train_target, val_target, train_iqr=-1, save_it=False, model_path=f"../models/{MODEL_NAME}.pkl"):
    """
    Trains a Ridge Regression model using a pipeline and evaluates it on training and validation sets.

    Parameters:
    - train: Training feature DataFrame.
    - val: Validation feature DataFrame.
    - train_target: Target values for training set.
    - val_target: Target values for validation set.
    - train_iqr: Optional IQR value for saving (used in outlier filtering during inference).
    - save_it: If True, saves the trained model and IQR info using joblib.
    - model_path: File path to save the model.

    Returns:
    - If save_it is False, the function only prints evaluation scores.
      If save_it is True, saves the pipeline and IQR in a dictionary to the specified path.
    """
    
    # Features that are included here should match the split made in the preprocessing pipeline

    # encoding 
    categorical_features = ['hour', 'season', 'passenger_count', 'month', 'is_jfk_airport', 'is_lg_airport', 'virtual_speed', 'virtual_speed_cube'] 
    # scaling
    numeric_features = [ 'coord_square_sum', 'coord_arithmetic_mean', 
                        'coord_harmonic_mean', 
                        ]

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
        ]
        , remainder = 'passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('regression', Ridge(alpha=1))
    ])
    
    # training
    model = pipeline.fit(train, train_target)
    predict_eval(model, train, train_target, "train")
    predict_eval(model, val, val_target, "validation")

    if not save_it:
        return

    to_save = {
        "model": model,
        "train_iqr": train_iqr,
    }

    # pickle
    joblib.dump(to_save, model_path)
    print(f"Model saved to path {model_path}")


if __name__ == "__main__":
    main()