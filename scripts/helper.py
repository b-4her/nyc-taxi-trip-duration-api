from sklearn.metrics import r2_score, root_mean_squared_error

def predict_eval(model, test, target, name):
    pred = model.predict(test)

    rmse = root_mean_squared_error(target, pred)
    r2 = r2_score(target, pred)
    
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")
