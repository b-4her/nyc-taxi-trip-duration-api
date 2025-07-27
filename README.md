# NYC Taxi Trip Duration Prediction

This project uses machine learning to predict trip duration based on extracted patterns from trip-related data.  
[Watch Demo Video](#) <!-- Replace # with the actual video link -->

<details>
<summary><strong>Table of Contents</strong></summary>

- [Project Overview](#project-overview)
- [Repo Structure and File Descriptions](#repo-structure-and-file-descriptions)
- [Development Process](#development-process)
  - [EDA](#eda)
  - [Feature Selection](#feature-selection)
  - [Modeling and Results](#modeling-and-results)
  - [API](#api)
  - [CLI](#cli)
- [Project Report](#project-report)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [API Usage](#api-usage)
- [CLI Usage](#cli-usage)
- [Lessons Learned](#lessons-learned)
- [Contact Information](#contact-information)
- [Acknowledgments](#acknowledgments)

</details>

---

## Project Overview

The main idea of this project is to understand the end-to-end deployment process of a machine learning model, focusing on each step in the pipeline and gaining hands-on experience throughout.

The project starts with a comprehensive exploratory data analysis (EDA) and emphasizes feature engineering and data preprocessing. Multiple models were trained and evaluated to predict taxi trip durations based on trip-related features.

The final model is deployed using a FastAPI service, and a command-line interface (CLI) tool was built to interact with the API — simulating how such a system could be used in production.




---


... TODO

get insights form Ehab's


Update Later:
📦 Project Root
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── final_ridge_pipeline.pkl
│   └── ridge_pipeline_5.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   └── inference-input-prep.ipynb
│
├── preprocessing/
│   ├── final_pipeline.py
│   └── pipeline_5.py
│
├── scripts/
│   ├── helper.py
│   ├── model_trainer.py
│   └── saved_models_evaluator.py
│
├── summary/
│   ├── feature_selection(remove).md
│   ├── model_results.md
│   ├── report.md
│   └── TODO.md
│
└── api/
    ├── app.py
    ├── endpoints.md
    └── predict_cli.py


one readme is enough, though start it with detailed toggels

mention directories navigation when necessary

Mention main goal to understand end to end production
