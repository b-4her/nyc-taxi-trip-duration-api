### Model Comparison Table

| Model      | Description                                       | Train RMSE | Train R²  | Val RMSE | Val R²   | Notes |
|------------|---------------------------------------------------|------------|-----------|----------|----------|-------|
| Baseline   | No engineered features; raw data only             | 0.6672     | 0.1200    | 0.6673   | 0.1205   | Establishes a reference to highlight the impact of feature engineering |
| Model 1    | Key engineered features + minimal transformation  | 0.4906     | 0.5244    | 0.4891   | 0.5276   | Feature engineering improved performance significantly |
| Model 2    | No column transformations; log on target only     | 270.8359   | 0.6347    | 271.5070 | 0.6355   | Model uses unscaled input features (raw scale); log-transform boosts performance |
| Model 3    | No one-hot encoding                               | 0.4178     | 0.6550    | 0.4178   | 0.6552   | Surprisingly good even without encoding, but one-hot improves final model |
| Model 4    | Removed outliers before log transform             | 0.4297     | 0.6383    | 0.4305   | 0.6379   | Removing outliers early slightly degraded performance |
| Model 5    | Trained only on training set before final eval    | 0.3933     | 0.6941    | 0.3932   | 0.6947   | Clean evaluation; near-best generalization |
| Model 6    | No scaling (MinMax/Standard)                      | 0.3939     | 0.6933    | 0.3940   | 0.6934   | Model still performs well; tree-based models less affected by scaling |
| Model 7    | MinMax scaling instead of StandardScaler          | 0.3960     | 0.6901    | 0.3959   | 0.6905   | Slightly worse than standardization |
| Final      | One-hot encoded, log target, outliers removed after log | 0.3931     | 0.6946    | 0.3930   | 0.6949   | Best model overall with clean preprocessing pipeline |

---

> **Note:**  
> All models in the table above were trained using a fixed Ridge Regression model with `alpha=1`. The objective was not to optimize the model itself, but rather to isolate and evaluate the impact of feature engineering. While more advanced models such as XGBoost or Neural Networks could significantly improve performance, this experiment demonstrates that careful feature understanding and design can lead to substantial gains on their own.  
>
> As shown, the R² score improved from **0.12 (baseline)** to **0.69 (final model)**—highlighting how powerful feature engineering can be when grounded in thorough data analysis.  
>
> All trained models are saved in the `models/` directory and can be evaluated using the `saved_models_evaluator.py` script found in the `scripts/` folder.  
>
> Detailed insights and commentary on model changes, feature choices, and observations throughout the modeling process are available in the 📄 [`final report`](nyc-taxi-trip-summary-report.pdf) located in the `summary/` directory.