# test / Inference Pipeline
import sys
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.Inference_pipeline.inference import predict

@pytest.fixture(scope="session")
def sample_df():
    sample_path = ROOT / "data/processed/val_fe.csv"
    print(f'Data : {pd.read_csv(sample_path).sample(n=5, random_state=42).reset_index(drop=True).columns} ')
    return pd.read_csv(sample_path).sample(n=5, random_state=42).reset_index(drop=True)

def test_infrerence_runs_and_returns_dataframe(sample_df):

    preds_df = predict(sample_df)

    assert not preds_df.empty, "The predictions DataFrame is empty."

    assert "predicted_price" in preds_df.columns, "The predictions DataFrame does not contain 'predicted_price' column."

    assert pd.api.types.is_numeric_dtype(preds_df["predicted_price"]), "'predicted_price' column is not numeric."

    print("Inference test passed successfully.")  

    print(preds_df["predicted_price"].head())

