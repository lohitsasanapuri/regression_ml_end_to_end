import math
from pathlib import Path
from joblib import load

from  src.training_pipeline.train import train_model
from  src.training_pipeline.eval import evaluate_model
from  src.training_pipeline.tune import tune_model

TRAIN_PATH = Path(r"data/processed/train_fe.csv")
VAL_PATH = Path(r"data/processed/val_fe.csv")

def _assert_metrics(m):
    assert set(m.keys()) == {"mae", "rmse", "r2"}
    assert all([isinstance(v, float) and math.isfinite(v)  for v in m.values()])

def test_train_evaluate_pipeline(tmp_path):
    out_path = tmp_path / "model_test.pkl"

    _, metrics = train_model(
        train_path= TRAIN_PATH,
        val_path= VAL_PATH,
        model_path= out_path,
        model_params= {"n_estimators": 100, "max_depth": 5, "random_state": 22, "n_jobs": -1, "tree_method": "hist"},
        sample_frac= 0.02,
    )

    assert out_path.exists()
    _assert_metrics(metrics)
    model = load(out_path)
    assert model is not None
    print(" Train_model test passed ")

def test_eval_works_with_saved_model(tmp_path):

    model_path = tmp_path / "xgb_model.pkl"
    train_model(
        train_path= TRAIN_PATH,
        val_path= VAL_PATH,
        model_path= model_path,
        model_params= {"n_estimators": 100, "max_depth": 5, "random_state": 22, "n_jobs": -1, "tree_method": "hist"},
        sample_frac= 0.02,
    )
    metrics = evaluate_model(
        val_path= VAL_PATH,
        model_path= model_path,
        sample_frac= 0.02,
        random_state= 22
    )
    _assert_metrics(metrics)
    print(" Evaluate_model test passed ")

def test_tune_model(tmp_path):
    model_path = tmp_path / "xgb_best_model.pkl"
    tracking_dir = tmp_path / "mlruns"
    best_params, best_metrics = tune_model(
        train_path= TRAIN_PATH,
        val_path= VAL_PATH,
        model_path= model_path,
        random_state= 22,
        sample_frac= 0.02,
        n_trails= 5,
        tracking_url= tracking_dir.as_uri(),
        experiment_name= "test_xgboost_optuna_housing"
    )
    assert model_path.exists()
    assert isinstance(best_params, dict) and len(best_params) > 0
    _assert_metrics(best_metrics)
    print(" Tune_model test passed ")

