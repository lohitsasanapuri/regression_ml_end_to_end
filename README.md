
# initiate ENve
uv init
uv python install 3.11
uv python pin 3.11
# Install Dependencies
uv sync
# Activate Environment
.\.venv\Scripts\Activate --powershell

uv add ipykernal
python -m ipykernel install --user --name=regression-ml-end-to-end --display-name "regression-ml-end-to-end"

# manual Add the environment in the vs code 
<Project_Path>\.venv\Scripts\python.exe

## View Local MLFlow Logs
mlflow ui
http://127.0.0.1:5000
CTRL + C to Quit

## Test the  Fast Api App 
uvicorn src.api.main:app --reload

## Dcoker Command for Local Running application 
docker build -t regression-ml-api . 
docker run -d -p 8000:8000  --name regression_api -e AWS_ACCESS_KEY_ID=""  -e AWS_SECRET_ACCESS_KEY=""-e AWS_DEFAULT_REGION=  regression-ml-api 

docker build -t regression-ml-ui . -f Dockerfile.streamlit

docker run -d -p 8501:8501 --name regression-ml-ui regression-ml-ui

## Docker compose 
docker compose up -d
-- ReBuild
docker compose up -d --build
-- down 
docker compose down


