# -- exposing the functions as API endpoints --
from fastapi import FastAPI
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import os, boto3

from src.Inference_pipeline.inference import predict

# configuing AWS S3 client
S3_BUCKET = os.getenv("S3_BUCKET", "ml-prj-data-repo")
REGION = os.getenv("AWS_REGION", "us-east-1")
#access_key = os.getenv("AWS_ACCESS_KEY_ID", "#your-access-key")
#secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "#your-secret-key")
s3 = boto3.client("s3", region_name=REGION)
#, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

def download_file_from_s3(s3_key: str, local_path: Path) -> None:
    """Download a file from S3 to a local path."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
    s3.download_file(S3_BUCKET, s3_key, str(local_path))
    print(f"Downloaded {s3_key} to {local_path}")
    return str(local_path)

#  Path configurations
MODEL_PATH = Path(download_file_from_s3("models/xgb_model.pkl", "models/xgb_model.pkl"))
TRAIN_FE_PATH = Path(download_file_from_s3("processed/train_fe.csv", "data/processed/train_fe.csv"))

# Load expected training features for alignment
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]
else:
    TRAIN_FEATURE_COLUMNS = None

# fast api app
app = FastAPI(title="Housing Price Prediction API", version="1.0")

@app.get('/')
def home():
    return {"message": "Welcome to the Housing Price Prediction API"}

@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not found"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
    return status

@app.post("/predict")
def predict_batch(data: List[dict]):
    #print(f' Data : {data}')
    if not MODEL_PATH.exists():
        return {"error": f"Model not found at {str(MODEL_PATH)}"}

    df = pd.DataFrame(data)
    #df = pd.read_csv(TRAIN_FE_PATH).head()
    if df.empty:
        return {"error": "No data provided"}
    
    #print(df)

    preds_df = predict(df, model_path=MODEL_PATH)

    resp = {"predictions": preds_df["predicted_price"].astype(float).tolist()}
    if "actual_price" in preds_df.columns:
        resp["actuals"] = preds_df["actual_price"].astype(float).tolist()

    return resp

@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        return {"error": "No predictions found"}

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }

# uvicorn src.api.main:app --reload