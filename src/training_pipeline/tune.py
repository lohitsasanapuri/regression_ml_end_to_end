"""
Hyperparamater Tuning Optune + Ml flow
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.xgboost

TRAIN_PATH = Path(r"data/processed/train_fe.csv")
VAL_PATH = Path(r"data/processed/val_fe.csv")
MODEL_PATH = Path(r"model/xgboost_best_model.pkl")

def _get_sample(df : pd.DataFrame, sample_frac : Optional[float], random_state : int = 22)-> pd.Dataframe:
    if sample_frac is None:
        return df
    if sample_frac <= 0 or sample_frac >= 0 :
        return df
    return df.sample(frac= sample_frac, random_state=  random_state).reset_index(drop=True)

def _load_data(
        train_path: Path | str = TRAIN_PATH,
        val_path: Path | str = VAL_PATH,
        sample_frac: Optional[float] = None,
        random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) 

    train_df = _get_sample(train_df, sample_frac=sample_frac, random_state=random_state)
    val_df = _get_sample(val_df, sample_frac=sample_frac, random_state=random_state)

    target_column = "price"
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]

    return X_train, y_train, X_val, y_val


def tune_model(
        train_path : Path|str = TRAIN_PATH,
        val_path : Path|str = VAL_PATH,
        model_path : Path|str =MODEL_PATH,
        random_state : int = 22,
        sample_frac : Optional[float] = None,
        n_trails : int = 15,
        tracking_url : str = None,
        experiment_name :str = "xgboost_optuna_housing"
) -> Tuple[Dict, Dict] :
    
    """ Running Hyperparameter Tuning with Optuna 
        Returns :
        best_params :[str parms]
        best_metrics:[str float] """
    
    if tracking_url:
        mlflow.set_registry_uri(tracking_url)

    mlflow.set_experiment(experiment_name)

    X_train , y_train, X_test, y_test = _load_data(train_path,val_path)

    def objective(trail : optuna.Trial):
        params = {
            "n_estimators": trail.suggest_int("n_estimators",200, 800),
            "max_depth": trail.suggest_int("max_depth", 3, 10),
            "learning_rate": trail.suggest_float("learning_rate",0.01,0.3,log=True),
            "subsample": trail.suggest_float("subsample",0.5,1.0),
            "colsample_bytree": trail.suggest_float("colsample_bytree",0.5,1.0),
            "min_child_weight": trail.suggest_int("min_child_weight",1,10),
            "gamma": trail.suggest_float("gamma",0,5),
            "reg_alpha": trail.suggest_float("reg_alpha",0,5),
            "reg_lambda": trail.suggest_float("reg_lambda",0,5),
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        with mlflow.start_run(nested= True):
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            mlflow.log_params(params)
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

        return rmse
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials= n_trails)

    best_params = study.best_params
    print(" Best Hyperparameters : ", best_params)

    best_model = XGBRegressor(**best_params, random_state= random_state, n_jobs= -1, tree_method= "hist")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    best_metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }
    print(" Best Model Metrics : ", best_metrics)

    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out)
    print(f" Best Model saved at {out} ")

    with mlflow.start_run(run_name= "base_xgboost_model"):
        mlflow.xgboost.log_model(best_model,"model")
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)

    return best_params, best_metrics

if __name__ == "__main__" :
    tune_model()