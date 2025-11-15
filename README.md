
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