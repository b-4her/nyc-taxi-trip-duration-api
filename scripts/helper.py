from sklearn.metrics import r2_score, root_mean_squared_error

def predict_eval(model, test, target, name):
    """
    Evaluates a regression model using RMSE and R² score.

    Parameters:
    - model: Trained scikit-learn model with a .predict() method.
    - test: Features (X) to make predictions on.
    - target: True target values (y) corresponding to the test set.
    - name: String identifier for labeling the output (e.g., model name).

    Prints:
    - Root Mean Squared Error (RMSE) and R² score formatted to 4 decimal places.
    """

    # Generate predictions from the model
    pred = model.predict(test)

    # evaluate
    rmse = root_mean_squared_error(target, pred)
    r2 = r2_score(target, pred)
    
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")
