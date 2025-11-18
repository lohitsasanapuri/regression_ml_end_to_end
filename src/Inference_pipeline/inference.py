# Raw Input -> Preprocessing  -> Feature Engineering -> Model prediction -> Output
from __future__ import annotations
import argparse
import pathlib
from pathlib import Path
import pandas as pd
from joblib import load

from src.feature_pipeline.process import  clean_and_merge, drop_duplicates, remove_outliers
from src.feature_pipeline.feature_engineering import add_data_features, drop_unused_columns

# Deafult Paths

PROJECT_Path = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL = PROJECT_Path / "models/xgb_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_Path / "models/freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_Path / "models/target_encoder.pkl"
TRAIN_FE_PATH  = PROJECT_Path / "data/processed/train_fe.csv"
DEFAULT_OUTPUT = PROJECT_Path / "predictions.csv"

print(" Inference using project path :", PROJECT_Path)

# Load Feature Columns

if TRAIN_FE_PATH.exists():
    train_fe_df = pd.read_csv(TRAIN_FE_PATH)
    TRAIN_FEATURE_COLUMNS = [col for col in train_fe_df.columns if col != "price"]
else:
    TRAIN_FEATURE_COLUMNS = None


def predict(
        input_df :pd.DataFrame,
        model_path : Path|str = DEFAULT_MODEL,
        freq_encoder_path : Path|str = DEFAULT_FREQ_ENCODER,
        target_encoder_path : Path|str = DEFAULT_TARGET_ENCODER     
) -> pd.DataFrame:
    
    # Process the RAW Input Data
    df = clean_and_merge(input_df)
    #df = drop_duplicates(df)
    #df = remove_outliers(df)

    # Feature Engineering
    if "date" in df.columns:
        df = add_data_features(df)

    # Frequency Encoding
    if Path(freq_encoder_path).exists() and "zipcode" in df.columns:
        freq_map = load(freq_encoder_path)
        df['zipcode_freq'] = df["zipcode"].map(freq_map).fillna(0)
        df = df.drop(columns=["zipcode"], errors='ignore')
    
    # Target Encoding
    if Path(target_encoder_path).exists() and "city_full" in df.columns:
        target_encoder = load(target_encoder_path)
        df['city_full_enc'] = target_encoder.transform(df["city_full"])
        df = df.drop(columns=["city_full"], errors='ignore')

    # Drop Unused Columns
    df, _ = drop_unused_columns(df.copy(), df.copy())

    # Separatre prediction features
    y_true = None
    if "price" in df.columns:
        y_true = df["price"]
        df = df.drop(columns=["price"], errors='ignore')
    
    # Aligns with training features
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # Load Model and Predict
    model = load(model_path)
    #print(f"feature names: {model.get_booster().feature_names}")
    y_pred = model.predict(df)

    # Build Output DataFrame
    out = df.copy()
    out['predicted_price'] = y_pred
    if y_true is not None:
        out['actual_price'] = y_true.values
    return out

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Pipeline for House Price Prediction")
    parser.add_argument("--input",type=str, required=True, help="Path to input CSV file containing raw data")  
    parser.add_argument("--output",type=str, default=DEFAULT_OUTPUT, help="Path to output CSV file for predictions")
    parser.add_argument("--model",type=str, default=DEFAULT_MODEL, help="Path to trained model file")
    parser.add_argument("--freq_encoder",type=str, default=DEFAULT_FREQ_ENCODER, help="Path to frequency encoder file")
    parser.add_argument("--target_encoder",type=str, default=DEFAULT_TARGET_ENCODER, help="Path to target encoder file")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        input_df= raw_df,
        model_path= args.model,
        freq_encoder_path= args.freq_encoder,
        target_encoder_path= args.target_encoder
    )
    preds_df.to_csv(args.output, index=False)
    print(f" Predictions saved to {args.output} ")