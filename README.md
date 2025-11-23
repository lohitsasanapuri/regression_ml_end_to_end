End-to-End Mlops Production ready: Housing Price Prediction Pipeline 🏠

📋 Project Overview

This project implements a complete, production-ready Machine Learning pipeline for predicting housing prices. It goes beyond simple model training by implementing MLOps best practices, ensuring the model is reproducible, trackable, and deployable.

The system is built using a modular architecture, containerized with Docker, and deployed to AWS ECS (Elastic Container Service) using Fargate.

Key Features

Modular Pipeline: Separation of concerns with dedicated pipelines for feature_engineering, training, and inference.

Experiment Tracking: Full integration with MLflow for tracking hyperparameters, metrics, and model versioning.

Advanced Data Processing: Includes dedicated notebooks for data cleaning, frequency encoding, and target encoding.

Microservices Architecture:

API Service: A FastAPI backend serving predictions.

UI Service: A Streamlit frontend for user interaction.

Infrastructure as Code: Includes AWS Task Definitions (housing-api-task-def.json) for ECS deployment.

CI/CD: Automated testing (pytest) and deployment workflows via GitHub Actions.

📸 Project Screenshots

<!--
TIPS FOR ADDING IMAGES:

Create a folder named assets in your repo.

Upload your images to that folder.

Update the paths below (e.g., ./assets/my-ui-screenshot.png).
-->

Streamlit UI

The user interface for making predictions.

<!--  -->

MLflow Tracking

Experiment tracking and model registry dashboard.

<!--  -->

🛠️ Tech Stack & Tools

Category

Technologies

Language & Env

Python 3.11, uv (Astral)

ML Libraries

XGBoost, Scikit-Learn, Pandas, NumPy

Tracking

MLflow

Web Frameworks

FastAPI (Backend), Streamlit (Frontend)

Containerization

Docker, Docker Compose

Cloud (AWS)

ECR (Registry), ECS (Orchestration), S3 (Storage)

Testing

Pytest, Smoke Tests (Jupyter)

☁️ Cloud Architecture

The application is deployed on AWS using a serverless container architecture. The diagram below illustrates the flow from code commit to production deployment.

graph TD
    subgraph "Local Development"
        Dev[User / Developer]
        Code[VS Code / Git]
    end

    subgraph "CI/CD Pipeline (GitHub Actions)"
        Push[Git Push]
        Test[Pytest & Lint]
        Build[Docker Build]
    end

    subgraph "AWS Production Environment"
        ECR[AWS ECR Registry]
        
        subgraph "ECS Cluster (Fargate)"
            API[FastAPI Container]
            UI[Streamlit Container]
        end
        
        S3[(AWS S3 Bucket)]
    end

    Dev -->|Commit Code| Code
    Code -->|Trigger| Push
    Push --> Test
    Test -->|Success| Build
    Build -->|Push Image| ECR
    ECR -->|Pull Image| API
    ECR -->|Pull Image| UI
    API -.->|Load Model| S3
    UI -->|Request| API


🔄 Deployment Workflow

Push to GitHub: Code is pushed to the repository, triggering the ci.yaml workflow.

Continuous Integration: GitHub Actions runs the unit tests (pytest).

Continuous Delivery:

If tests pass, the Docker images are built.

Images are tagged and pushed to Amazon Elastic Container Registry (ECR).

Deployment: Amazon ECS pulls the new images and updates the running Fargate tasks with zero downtime.

🏗️ Design Decisions

Why AWS Fargate? We chose Fargate (Serverless ECS) to avoid managing EC2 instances. It scales automatically and reduces operational overhead.

Why Docker? Ensures the model runs exactly the same in the cloud as it does on the local machine.

Why Separation of Concerns? The UI (Streamlit) interacts with the Model only via the API (FastAPI), decoupling the frontend from the ML logic.

📂 Project Structure

REGRESSION_ML_END_TO_END
├── .github/workflows         # CI/CD pipelines (ci.yml)
├── data                      # Raw and processed datasets
├── models                    # Serialized models (xgb_model.pkl, encoders)
├── notebooks                 # Experimentation & EDA
│   ├── 00_data_processing.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_eng_and_encoding.ipynb
│   ├── 03_basline_model.ipynb
│   ├── 05_xgboost.ipynb
│   ├── 06_Hyperparameter_tuning_mlflow.ipynb
│   └── 07_Push_dataset_aws.ipynb
├── src                       # Production Source Code
│   ├── api                   # FastAPI application (main.py)
│   ├── feature_pipeline      # Data transformation logic
│   ├── training_pipeline     # Model training logic
│   └── inference_pipeline    # Prediction logic
├── tests                     # Unit & Integration tests
├── housing-api-task-def.json # AWS ECS Task Definition (API)
├── streamlit-task-def.json   # AWS ECS Task Definition (UI)
├── Dockerfile                # API Docker Image
├── Dockerfile.streamlit      # UI Docker Image
├── docker-compose.yml        # Local orchestration
└── pyproject.toml            # Dependencies


🚀 Local Installation & Setup

This project uses uv for high-performance dependency management.

1. Initialize Environment

# 1. Initialize uv and python version
uv init
uv python install 3.11
uv python pin 3.11

# 2. Install Dependencies
uv sync

# 3. Activate the virtual environment (Powershell)
.\.venv\Scripts\Activate


2. VS Code Configuration (Manual Step)

To ensure VS Code uses the correct environment:

Open the Command Palette (Ctrl+Shift+P).

Search for Python: Select Interpreter.

Select the path manually:
.\.venv\Scripts\python.exe

3. Setup Jupyter Kernel

Required for running the notebooks locally:

# Install kernel support
uv add ipykernel

# Register the kernel
python -m ipykernel install --user --name=regression-ml-end-to-end --display-name "regression-ml-end-to-end"


🏃 Running the Application

Option A: Docker Compose (Recommended)

Build and run both the API and the UI simultaneously.

# 1. Start application
docker compose up -d

# 2. Rebuild and start (use this if you modify code or Dockerfiles)
docker compose up -d --build

# 3. Stop application
docker compose down


API Docs: http://localhost:8000/docs

Streamlit UI: http://localhost:8501

MLflow UI: http://localhost:5000

Option B: Running Services Manually

1. Start MLflow UI
View experiment logs and model registry.

mlflow ui
# Accessible at [http://127.0.0.1:5000](http://127.0.0.1:5000)
# Press CTRL + C to Quit


2. Run FastAPI Backend

uvicorn src.api.main:app --reload


3. Run Streamlit Frontend
(If running outside Docker)

streamlit run src/ui/app.py 
# (Adjust path based on your specific UI file location)


🐳 Docker Commands (Manual)

If you need to build specific images individually using docker run with environment variables:

1. Build & Run API

docker build -t regression-ml-api . 

# Run container with AWS Credentials (replace values as needed)
docker run -d -p 8000:8000 --name regression_api \
    -e AWS_ACCESS_KEY_ID="" \
    -e AWS_SECRET_ACCESS_KEY="" \
    -e AWS_DEFAULT_REGION="us-east-1" \
    regression-ml-api 


2. Build & Run UI

docker build -t regression-ml-ui . -f Dockerfile.streamlit

docker run -d -p 8501:8501 --name regression-ml-ui regression-ml-ui


🧪 Testing

The project includes a robust testing suite using pytest.

# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_train.py
pytest tests/test_inference.py


👏 Acknowledgements

This project was developed as a hands-on implementation of MLOps principles, inspired by the comprehensive guide from Anas Riad. It serves as a practical demonstration of taking a model from a Jupyter Notebook to a deployed cloud application.