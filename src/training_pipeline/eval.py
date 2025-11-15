from __future__ import annotations
from pathlib import Path
from typing import Optional,Dict

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

VAL_PATH = Path(r"data/processed/val_fe.csv")
MODEL_PATH = Path(r"models/xgb_model.pkl")

def _get_sample(df : pd.DataFrame, sample_frac: Optional[float], random_state: int)-> pd.DataFrame :
    if sample_frac is None :
        return df
    if sample_frac <= 0 or sample_frac >= 1 :
        return df
    return df.sample(frac = sample_frac, random_state= random_state).reset_index(drop=True) 

def evaluate_model(
        val_path : Path|str = VAL_PATH,
        model_path : Path|str = MODEL_PATH,
        model_param : Optional[dict] = None,
        sample_frac: Optional[float] = None,
        random_state : int = 22)-> Dict[str, float]:
    """
    This Function Loads the model from the memory and evaluate on the validation dataset.
    
    Returns: 
    metrics : dict[str, float]
    """
    
    val_df = pd.read_csv(val_path)

    val_df = _get_sample(val_df, sample_frac= None, random_state= 22)

    target_column = "price"
    X_val, y_val = val_df.drop(columns=[target_column]),val_df[target_column]

    model = load(MODEL_PATH)
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val,y_pred)
    rmse = np.sqrt(mean_squared_error(y_val,y_pred))
    r2 = r2_score(y_val,y_pred)

    metrics = {"mae":mae, "rmse": rmse, "r2": r2}

    print(" Model Evalution Completed ")
    print(f" mae : {mae}, rmse : {rmse} , r2 : {r2} ")

    return metrics

if __name__ == "__main__" :
    evaluate_model()


