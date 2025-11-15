from  __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DEFAULT_TRAIN = Path("data/processed/train_fe.csv")
DEFAULT_VAL = Path("data/processed/val_fe.csv")
MODEL_PATH = Path("models/xgb_model.pkl")

def _get_sample(df:pd.DataFrame, sample_frac:Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac = sample_frac, random_state=random_state).reset_index(drop=True)

def train_model(
        train_path : Path|str = DEFAULT_TRAIN,
        val_path : Path|str = DEFAULT_VAL,
        model_path : Path|str = MODEL_PATH,
        model_params : Optional[Dict] = None,
        sample_frac : Optional[float] = None,
        random_state : int = 22):
    """
    Train Basline Model XGB and Save to the file
    Returns:
    ----------
    model : XGB Regressor
    metrics : dict[str, float]
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_df = _get_sample(train_df, sample_frac, random_state)
    val_df = -_get_sample(val_df, sample_frac, random_state)

    target_column = "price"
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val =  val_df.drop(columns = [target_column]), val_df[target_column]

    params = {
        "n_estimators" : 500,
        "learning_rate" : 0.005,
        "max_depth" : 6,
        "subsample": 0.8,
        "colsample_bytree" : 0.8,
        "random_state" : random_state,
        "n_jobs" : -1,
        "tree_method":"hist"
    } 
    if model_params :
        model_params.update(params)

    model = XGBRegressor(**params)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_val)
    mae = float(mean_absolute_error(y_val,y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val,y_pred)))
    r2 = float(r2_score(y_val,y_pred))

    metrics = {"mae": mae, "rmse" : rmse, "r2":r2}

    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model,out)
    print(f" Model trained and Saved to {out} ")
    print(f" MAE  : {mae}, RMSE: {rmse}, r2 : {r2}")

    return model , metrics

if __name__ == "__main__" :
    train_model()