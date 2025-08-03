<a id="readme-top"></a>

<!-- PROJECT TITLE -->
<br />
<div align="center">
  <h1 align="center"><b>NYC Taxi Trip Duration Prediction</b></h1>

  <p align="center">
    <i>An end-to-end machine learning project that explores, builds, and deploys a predictive model for NYC taxi trip durations — from raw data analysis to API integration, packaged in a Docker container for reproducibility and production readiness.</i>
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
  - [API Client](#api-client)
- [Project Report](#project-report)
- [Lessons Learned](#lessons-learned)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Docker Usage](#docker-usage)
- [API Usage](#api-usage)
  - [Interacting with the API Using cURL](#interacting-with-the-api-using-curl)
- [CLI Usage](#cli-usage)
- [API Client Usage](#api-client-usage)
- [Contact Information](#contact-information)
- [Resources and Credits](#resources-and-credits)
  - [Libraries and Frameworks](#libraries-and-frameworks)
  - [Development Tools](#development-tools)


</details>

---

## Overview

The main idea of this project is to understand the end-to-end deployment process of a machine learning model, focusing on each step in the pipeline and gaining hands-on experience throughout.

The project starts with a comprehensive exploratory data analysis (EDA) and emphasizes feature engineering and data preprocessing. Multiple models were trained and evaluated to predict taxi trip durations based on trip-related features.

The final model is deployed as a FastAPI service. To simulate real-world usage, a command-line interface (CLI) tool was developed to interact with the API. The project is containerized using Docker, making it production-ready and reproducible across different systems.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Repo Structure and File Descriptions

```
project-root/
├── README.md                      # Project overview and usage instructions
├── LICENSE
├── requirements.txt              # Python dependencies
├── .gitignore
├── .dockerignore                 # Docker ignore rules
├── Dockerfile                    # Docker build configuration
│
├── models/                       # Saved Ridge models
│   ├── final_ridge_pipeline.pkl
│   └── ridge_pipeline_5.pkl
│
├── notebooks/                    # Notebooks for EDA and input prep
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
└── api/                          # API and CLI tools
    ├── app.py                    # FastAPI application
    ├── endpoints.md              # API endpoint documentation
    ├── api_cli.py                # CLI tool to interact with API
    ├── api_client.py             # Python-based client interface
    └── api_client_demo.py        # Demo script for client usage
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

### Client

(TODO)

---

## Project Report

A detailed summary of the entire project — from EDA to deployment — is available in the report file:

📄 [`nyc-taxi-trip-summary-report.pdf`](summary/nyc-taxi-trip-summary-report.pdf)

> Note: The Client section, Docker integration, and cURL-based API usage were added after the report was written and are not included in it.

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
  ```
2. **Navigate to the project directory**

  ```bash
  cd nyc-taxi-trip-duration-api
  ```

3.	**Install dependencies**

   ```bash
   pip install -r requirements.txt
  ```

> If you’d prefer not to install the dependencies locally, you can skip this step and use the project via [Docker](#docker-usage) as explained in the next step.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### Docker Usage

To run the API using Docker (no need to install dependencies manually):

1. **Make sure Docker is installed**   
   → [Install Docker](https://docs.docker.com/get-docker/)

2. **Build the Docker image**  
   Run this from the root directory of the project:
   ```bash
   docker build -t nyc-taxi-api .
   ```
3. **Run the API container**  
    This will start the FastAPI app on port 8000:
    ```bash
    docker run --name taxi-api -p 8000:8000 nyc-taxi-api
    ```
    > Tip: If port 8000 is already in use, either stop the conflicting container or run on a different port using -p 8080:8000.  
    
    To interact with the API directly after it is running, see [CURL-Commands](#interacting-with-the-api-using-curl).
4. **See running containers**  
    To list all currently running containers, use:
    ```bash
    docker ps
    ```
    To list all containers, including stopped ones, use:
    ```bash
    docker ps -a
    ```
    This will show container IDs, names, status, ports, and more.
5. **Stop the API container**  
    To stop the running container named taxi-api, run:
    ```bash
    docker stop taxi-api
    ```
    > Tip: Use the container name (like taxi-api) or container ID at the end of the command to specify which container to stop.
6. **Remove the API container**  
    If you want to remove the container after stopping it (to free up resources), run:
    ```docker
    docker rm taxi-api
    ```
    Remove all stopped containers:
    ```bash
    docker container prune
    ```
    > This will prompt for confirmation before deleting all stopped containers.
    Remove all containers (both running and stopped):
    ```bash
    docker rm -f $(docker ps -a -q)
    ```
    > **Warning:** This force-stops and deletes all containers permanently. Use with caution.

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

You can interact with the API directly using `curl`, a command-line tool for making HTTP requests. This is especially useful for testing or automation. The examples below show how to interact with each available endpoint.

> *Make sure the API is running locally** before executing any `curl` commands.  
> You can start the server with:

```bash
uvicorn api.main:app --reload
```

> It's recommended to **split your terminal**—use one pane for hosting the API and the other for running `curl` commands.

> If you're unfamiliar with `curl`, you may prefer the [CLI tool](#cli-usage) described in the next section for easier usage.

#### General Format

- **GET requests** retrieve data and do not require input.
- **POST requests** send data in JSON format to the server.
- Use `| jq` at the end to **prettify** the JSON output (requires [jq](https://stedolan.github.io/jq/)).

#### `GET` Endpoints (No Input Required)

```bash
# Get API information
curl http://127.0.0.1:8000/about | jq

# List of available endpoints
curl http://127.0.0.1:8000/help | jq

# Get list of required features
curl http://127.0.0.1:8000/features | jq

# Get a sample input dictionary
curl http://127.0.0.1:8000/features/sample | jq

# Show version information
curl http://127.0.0.1:8000/version | jq
```

#### `POST` Endpoint Usage (cURL)

The following examples demonstrate how to use `curl` to interact with the API's `POST` endpoints. These endpoints require sending JSON input data in the request body.

Use the `/predict` endpoint to get a trip duration estimate (minutes) based on trip details:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vendor_id": 1,
    "passenger_count": 1,
    "pickup_longitude": -73.988609,
    "pickup_latitude": 40.748977,
    "dropoff_longitude": -73.992797,
    "dropoff_latitude": 40.763408,
    "pickup_date": "2016-03-23",
    "pickup_time": "02:24",
    "store_and_fwd_flag": "N"
  }' | jq
```

Other POST endpoints such as `/validate` follow a similar format — they accept JSON in the request body and return JSON responses.

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

## Client Usage 

(TODO)

[api-client-demo](api/api_client_demo.py)  

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
* **Docker** – Containerization for reproducible deployment

### Development Tools:

* **Anaconda** – Environment and package management.
* **Jupyter Notebooks** – Interactive development and testing.
* **VS Code** – Code editing and debugging.
* **GitHub** – Version control and collaboration.
* **Postman** – API endpoint testing.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---
