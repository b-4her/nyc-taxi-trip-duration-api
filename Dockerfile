# Base image with Python
FROM python:3.11-slim

# Avoid writing .pyc files, force logs to stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy only the necessary directories and files
COPY api/ ./api/
COPY models/ ./models/
COPY models/final_ridge_pipeline.pkl ./models/final_ridge_pipeline.pkl
COPY preprocessing/final_pipeline.py ./preprocessing/final_pipeline.py
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /app/api

# Expose the FastAPI default port
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]