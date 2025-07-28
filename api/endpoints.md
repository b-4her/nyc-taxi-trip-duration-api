| Method | Endpoint        | Description                                                                 | Input Format | Response Format               |
|--------|-----------------|-----------------------------------------------------------------------------|--------------|--------------------------------|
| POST   | /predict        | Predicts trip duration based on user-provided trip features.                | JSON         | JSON (e.g., {"duration": 437}) |
| GET    | /features       | Returns a list of required input features for prediction.                   | None         | JSON (list of features)        |
| GET    | /features/sample | Returns a sample input dictionary to guide the user.                        | None         | JSON (sample trip_dict)        |
| GET    | /about          | Provides basic information about the model and how the prediction works.    | None         | JSON (text/info)               |
| POST   | /validate       | Validates a user input JSON against the expected schema.                    | JSON         | JSON (valid or errors)         |
| GET    | /version        | Returns version details of the model, API, and key libraries used.          | None         | JSON (version info)            |
| GET    | /help           | Returns a list of all endpoints with short descriptions.                    | None         | JSON (endpoint overview)       |