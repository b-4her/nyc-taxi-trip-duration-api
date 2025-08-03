<a id="readme-top"></a>

<!-- PROJECT TITLE -->
<br />
<div align="center">
  <h1 align="center"><b>NYC Taxi Trip Duration Prediction</b></h1>

  <p align="center">
    <i>end-to-end machine learning project that explores, builds, and deploys a predictive model for NYC taxi trip durations — from raw data analysis to API integration.</i>
    <br />
    <a href="https://youtu.be/your-demo-link"><strong>Quick Demo</strong></a>
  </p>
</div>

---
<!-- TABLE of CONTENTS -->
<details>
<summary><strong>Table of Contents</strong></summary>

- [Overview](#overview)
- [Repo Structure and File Descriptions](#repo-structure-and-file-descriptions)
- [Development Process](#development-process)
  - [EDA](#eda)
  - [Feature Selection](#feature-selection)
  - [Modeling and Results](#modeling-and-results)
  - [Inference Input Formatting](#Inference-input-formatting)
  - [Docker](#docker)
  - [API](#api)
  - [CLI](#cli)
- [Project Report](#project-report)
- [Lessons Learned](#lessons-learned)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Docker Usage](#docker-usage)
- [API Usage](#api-usage)
- [CLI Usage](#cli-usage)
- [Contact Information](#contact-information)
- [Resources and Credits](#resources-and-credits)
  - [Libraries and Frameworks](#libraries-and-frameworks)
  - [Development Tools](#development-tools)


</details>

---

## Overview

The main idea of this project is to understand the end-to-end deployment process of a machine learning model, focusing on each step in the pipeline and gaining hands-on experience throughout.

The project starts with a comprehensive exploratory data analysis (EDA) and emphasizes feature engineering and data preprocessing. Multiple models were trained and evaluated to predict taxi trip durations based on trip-related features.

The final model is deployed using a FastAPI service, and a command-line interface (CLI) tool was built to interact with the API — simulating how such a system could be used in production.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Repo Structure and File Descriptions

```
project-root/
├── README.md
├── LICENSE
├── requirements.txt               # Python dependencies
├── .gitignore
│
├── models/                        # Saved Ridge models
│   ├── final_ridge_pipeline.pkl
│   └── ridge_pipeline_5.pkl
│
├── notebooks/                     # Notebooks for EDA and input prep
│   ├── EDA.ipynb
│   └── inference-input-prep.ipynb
│
├── preprocessing/                # Pipeline scripts
│   ├── final_pipeline.py
│   └── pipeline_5.py
│
├── scripts/                      # Training and evaluation
│   ├── helper.py
│   ├── model_trainer.py
│   └── saved_models_evaluator.py
│
├── summary/                      # Results and report
│   ├── model_results.md
│   └── nyc-taxi-trip-summary-report.pdf
│
└── api/                          # API and CLI
    ├── app.py
    ├── endpoints.md
    └── predict_cli.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Development Process

### EDA

📄 [`EDA.ipynb`](notebooks/EDA.ipynb)

The exploratory data analysis (EDA) phase focused on understanding the dataset's structure and identifying patterns to guide feature engineering. Key actions included:

- Cleaning invalid coordinates and removing outliers.
- Extracting temporal features (e.g., hour, weekday, month).
- Mapping spatial patterns using pickup/dropoff clusters.
- Identifying longer trip durations near JFK and LaGuardia airports.
- Transforming the target variable (`trip_duration`) using `log1p` to reduce skew.

Engineered features included `is_rush_hour`, `is_summer`, `requires_large_vehicle`, and distance-based transformations, among others.

### Feature Selection

Feature selection was guided by correlation analysis and experimentation. Highlights:

- Not all highly correlated features led to better performance, and some low-correlation features helped when encoded.
- Custom-engineered features like `virtual_time` (derived from `trip_distance` and `virtual_speed`) improved prediction performance.
- Final selection balanced statistical strength and domain knowledge.

### Modeling and Results

Modeling was done using Ridge Regression with standardization and one-hot encoding. Key points:

- The baseline R² was ~0.12. With feature engineering and transformations, it improved to ~0.69.
- Multiple pipeline variants were tested to isolate the effect of each transformation (e.g., scaling, outlier removal, encoding).
- Final pipeline steps included log-transforming the target, outlier removal post-transform, feature engineering, encoding, and scaling.

Model tracking showed that even simple changes (like transformation order or encoding strategy) impacted results significantly.

Detailed model results and performance comparisons are available in 📄 [`summary/model_results.md`](summary/model_results.md).

### Inference Input Formatting 

A dedicated notebook (📄 [`inference-input-prep.ipynb`](notebooks/inference-input-prep.ipynb)) was created to:

- Format new trip data to match training features.
- Apply consistent preprocessing logic.
- Ensure compatibility with the final serialized pipeline.

This bridged the gap between training and real-time inference.

### Docker

To simplify deployment and avoid manual installation of dependencies, a Docker setup was added. This allows users to run the API in a containerized environment without installing Python packages locally.

- The Docker image includes the trained model, all required dependencies, and the FastAPI app.
- Useful for users who do not wish to use the CLI or set up a virtual environment.

### API

A RESTful API was built using FastAPI to serve the model. Main endpoints:

- `POST /predict` — Returns trip duration prediction.
- `GET /features`, `GET /features/sample` — Show required fields and sample input.
- `GET /help`, `GET /about` — Describe API and usage.

A cURL usage guide is also included for users who prefer to interact with the API directly from the terminal without using the CLI tool.

### CLI

A CLI tool using `argparse` and `requests` was created to interact with the API directly from the terminal. Users can:

- Pass input features via command-line arguments.
- Hit endpoints like `/predict` or `/validate`.
- Test predictions without needing a frontend.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Client

(TODO)

---

## Project Report

A detailed summary of the entire project — from EDA to deployment — is available in the report file:

📄 [`nyc-taxi-trip-summary-report.pdf`](summary/nyc-taxi-trip-summary-report.pdf)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Lessons Learned

- **EDA is critical**: Every useful feature came from clear, testable questions about the data.
- **Assumptions must be validated**: Some features that seemed weak became essential post-encoding.
- **Data leakage can be subtle**: The use of `store_and_fwd_flag` in inference highlighted this challenge.
- **Pipelines are powerful**: They ensured consistency from training to deployment.
- **Experimentation pays off**: Testing multiple pipeline variants revealed performance improvements not obvious during EDA.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

To run this project locally, clone the repository and install the required dependencies.

### Prerequisites

- Python 3.8 or later
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**

  ```bash
  git clone https://github.com/b-4her/nyc-taxi-trip-duration-api.git
  cd nyc-taxi-trip-duration-api
  ```

2.	**Install dependencies**

   ```bash
   pip install -r requirements.txt
  ```

> If you’d prefer not to install the dependencies locally, you can skip this step and use the project via [Docker](#docker-usage) as explained in the next step.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### Docker Usage

(TODO)

- To interact with the api directly see [CURL-Commands](#interacting-with-the-api-using-curl)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## API Usage

The project includes a FastAPI-based service to serve model predictions via HTTP requests.

**Base URL**

  ```bash
  http://localhost:8000
  ```

**Running the API**

> **Important:** You must first change directory into the `api` folder; otherwise, the API will not run correctly.

  ```bash
  cd api
  ```

  ```bash
  uvicorn app:app --reload
  ```

**Available Endpoints**

A full list of endpoints, expected inputs, and response formats can be found in:

📄 [`endpoints.md`](api/endpoints.md)

Basic endpoints include:
-	POST /predict – Make a prediction
-	GET /features – List required features
-	GET /features/sample – Show example input
-	GET /about – About the model
-	POST /validate – Validate input schema
-	GET /help – List all available endpoints

### Interacting with the API Using cURL

(TODO)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## CLI Usage

A command-line interface (CLI) script is provided for interacting with the API directly from your terminal. If you prefer not to use the CLI, you can alternatively interact with the API using standard [curl commands](#interacting-with-the-api-using-curl), as explained in the API Usage section.

To use the CLI, make sure the **API is already running locally**.

### Option 1: Run API and CLI in Split Terminals

You can split the terminal in **VS Code** to run both the API and CLI simultaneously:

1. **Split the Terminal:**
   - Hover over the terminal panel in VS Code.
   - Press `Ctrl` + `\` (Windows/Linux) or `Cmd` + `\` (macOS).  
     *(This is the general shortcut for splitting terminal windows.)*

2. **Start the API:**  
   - In one terminal, `cd` into the `api` directory and run the `uvicorn` command as shown in the [API Usage](#api-usage) section.

3. **Run the CLI:**
   - Use the second terminal pane to run your CLI commands.

### Option 2: Host the API Remotely

If you're hosting the API on a remote server:

1. **Update the CLI:**
   - Change the `BASE_URL` variable at the top of the CLI script to point to your server’s URL.

2. **Run the CLI Directly:**
   - You can now run the CLI from any terminal without needing to start the API locally.


### How It Works

- **Endpoint Selection**: Use the `--endpoint` flag to choose one of the API endpoints:
  - `predict`, `validate`, `features`, `features/sample`, `about`, `version`, or `help`
  - If no endpoint is specified, it defaults to `help`.
  - *Check the [API section](#api-usage) for what each endpoint does.*

**Example Usage**

> Make sure you are using **Python 3** to avoid syntax errors.

  ```bash
  python3 api/api_cli.py --endpoint predict \
  1 \
  1 \
  -73.988 \
  40.748 \
  -73.992 \
  40.763 \
  2016-03-23 \
  02:24 \
  N 
  ```

| Position | Parameter Name        | Description                                                 | Example Value   |
|----------|------------------------|-------------------------------------------------------------|-----------------|
| 1        | `vendor_id`            | Vendor ID of the trip (1 or 2)                              | `1`             |
| 2        | `passenger_count`      | Number of passengers in the trip                            | `1`             |
| 3        | `pickup_longitude`     | Longitude where the trip started                            | `-73.988609`    |
| 4        | `pickup_latitude`      | Latitude where the trip started                             | `40.748977`     |
| 5        | `dropoff_longitude`    | Longitude where the trip ended                              | `-73.992797`    |
| 6        | `dropoff_latitude`     | Latitude where the trip ended                               | `40.763408`     |
| 7        | `pickup_date`          | Date of the pickup in `YYYY-MM-DD` format                   | `2016-03-23`    |
| 8        | `pickup_time`          | Time of the pickup in `HH:MM` format                        | `02:24`         |
| 9        | `store_and_fwd_flag`   | Whether the trip was stored before forwarding (`Y` or `N`)  | `N`             |

### Notes

- All inputs (e.g., vendor ID, pickup coordinates, etc.) are passed as **positional arguments**.
- **Positional arguments are only required when the endpoint is `predict` or `validate`.**
- These arguments must appear **in exact order** that is shown in the example table.
- If any of the required arguments are missing, the script will raise an error and list what's missing.
- For other endpoints like `features`, `about`, or `help`, no additional input is needed.

For usage instructions:

  ```bash
    python3 api/api_cli.py --help
  ```

> Instead of running the script as `api/api_cli.py`, you can also change directory into the `api` folder first and then run the command without the `api/` prefix.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### Contact Information
For any questions or feedback, reach out via:
- LinkedIn: [b-4her](https://www.linkedin.com/in/b-4her)
- GitHub: [b-4her](https://github.com/b-4her)
- YouTube: [b-4her](https://www.youtube.com/@b-4her)
- Email: baher.alabbar@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Resources and Credits

* The dataset used in this project is private and cannot be publicly shared.
* Some code snippets and implementation ideas were refined with the help of ChatGPT. Feedback is welcome on any part of the codebase.
* All analysis, feature engineering, and insights presented in this project are entirely my own.

### Libraries and Frameworks:

* **scikit-learn** – Machine learning models and preprocessing.
* **pandas** – Data manipulation and cleaning.
* **numpy** – Numerical computations.
* **seaborn**, **matplotlib** – Visualization.
* **FastAPI**, **argparse** – Deployment of the API and CLI interface.

### Development Tools:

* **Anaconda** – Environment and package management.
* **Jupyter Notebooks** – Interactive development and testing.
* **VS Code** – Code editing and debugging.
* **GitHub** – Version control and collaboration.
* **Postman** – API endpoint testing.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---
