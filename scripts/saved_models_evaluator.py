import joblib
import argparse
import pandas as pd
import sys, os

sys.path.append(os.path.abspath('../preprocessing/'))

# Select the model pipeline:
from final_pipeline import preprocessing_pipeline
from helper import predict_eval

MODEL_NAME = 'final_ridge_pipeline'
TARGET_VARIABLE = 'log_trip_duration'

DEFAULT_MODEL_PATH = f'../models/{MODEL_NAME}.pkl'
DEFAULT_TEST_PATH = '../data/split/test.csv'

def load_model(path):
    saved = joblib.load(path)
    model = saved["model"]
    train_iqr = saved["train_iqr"]
    return model, train_iqr

def load_data(path):
    return pd.read_csv(path)

def main():
    parser = argparse.ArgumentParser(description="Load model and test data for prediction")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the pickled model and IQR file')
    parser.add_argument('--test_path', type=str, default=DEFAULT_TEST_PATH,
                        help='Path to the test CSV file')

    args = parser.parse_args()


    model, train_iqr = load_model(args.model_path)
    test = load_data(args.test_path)
    
    # Preparing data
    test, _ = preprocessing_pipeline(test, train_iqr)
    test_target = test[TARGET_VARIABLE]
    test.drop(TARGET_VARIABLE, axis=1, inplace=True)

    '''
        Note: Ensure that you import the correct preprocessing pipeline
        that matches the one used during model training.

        Name of target variable should also be matching to the one used here
    '''

    predict_eval(model, test, test_target, "TestSet")

if __name__ == "__main__":
    main()
