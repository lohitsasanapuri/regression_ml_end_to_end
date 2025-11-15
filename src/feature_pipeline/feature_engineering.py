from pathlib import Path
import pandas as pd 
from category_encoders import TargetEncoder
from joblib import dump

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def add_data_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional time-based features to the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame with a 'date' column.    
    Returns:
        pd.DataFrame: The DataFrame with added 'year', 'quarter', and 'month' columns.           
    """

    df['date'] = pd.to_datetime(df['date'])

    # Extract year, quarter, and month from the 'date' column
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month

    # Reorder columns to have 'date', 'year', 'quarter', 'month' at the front
    df.insert(1,"year",df.pop("year"))
    df.insert(2,"quarter",df.pop("quarter"))
    df.insert(3,"month",df.pop("month"))
    
    return df

def frequency_encode(train : pd.DataFrame, val: pd.DataFrame, column: str):
    """
    Apply frequency encoding to a specified categorical column in both training and validation DataFrames.
    Args:
        train (pd.DataFrame): The training DataFrame.
        val (pd.DataFrame): The validation DataFrame.
        column (str): The name of the categorical column to be frequency encoded.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, dict]: The modified training and validation DataFrames with a new frequency encoded column,
        and the frequency mapping dictionary.
    """
    freq_map = train[column].value_counts()
    train[f"{column}_freq"] = train[column].map(freq_map)
    val[f"{column}_freq"] = val[column].map(freq_map).fillna(0)
    return train, val, freq_map

def target_encode(train: pd.DataFrame, val: pd.DataFrame, col: str , target: str ) :
    te = TargetEncoder(train[col])
    column_name = f"{col}_enc" if col != "city_full " else "city_full_enc"
    train[column_name] = te.fit_transform(train[col],train[target])
    val[column_name] = te.transform(val[col])
    return train, val, te

def drop_unused_columns(train: pd.DataFrame, val : pd.DataFrame):
    drop_columns = ["date", "city_full", "city", "zipcode", "median_sale_price"]
    train = train.drop(columns=[c for c in drop_columns if c in train.columns], errors='ignore')
    val = val.drop(columns=[c for c in drop_columns if c in val.columns], errors='ignore')
    return train, val

# ---------------- Pipeline ----------------------

def run_feature_engineering(
        in_train_path : Path|str|None = None,
        in_val_path : Path|str|None = None,
        in_test_path : Path|str|None = None,
        output_dir : Path|str = PROCESSED_DIR) :
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    
    # Defaults for Inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DIR/"train_cleaned.csv"
    if in_val_path is None:
        in_val_path = PROCESSED_DIR/"val_cleaned.csv"
    if in_test_path is None :
        in_test_path = PROCESSED_DIR/"test_cleaned.csv"

    train_df = pd.read_csv(in_train_path)
    val_df = pd.read_csv(in_val_path)
    test_df = pd.read_csv(in_test_path)

    if "date" in train_df.columns:
        train_df = add_data_features(train_df)
        val_df = add_data_features(val_df)
        test_df = add_data_features(test_df)

    freq_map = None
    if "zipcode" in train_df.columns :
        train_df, val_df, freq_map = frequency_encode(train_df,val_df, 'zipcode')
        test_df['zipcode_freq'] = test_df["zipcode"].map(freq_map).fillna(0)
        dump(freq_map, MODELS_DIR/"freq_encoder.pkl") # Save Mapping

    target_encoder = None
    if "city_full" in train_df.columns:
        train_df, val_df, target_encoder = target_encode(train_df, val_df, 'city_full','price')
        test_df['city_full_enc'] = target_encoder.transform(test_df["city_full"])
        dump(target_encoder, MODELS_DIR/"target_encoder.pkl")

    # Drop unused columns 
    train_df, val_df = drop_unused_columns(train_df,val_df)
    test_df, _ = drop_unused_columns(test_df.copy(), test_df.copy())

    out_train_path = output_dir/'train_fe.csv'
    out_val_path =  output_dir/'val_fe.csv'
    out_test_path = output_dir/'test_fe.csv'

    train_df.to_csv(out_train_path,index=False)
    val_df.to_csv(out_val_path, index=False)
    test_df.to_csv(out_test_path,index=False)

    print(" Feature Engineering Completed ")
    print(f" Train Shape : {train_df.shape}")
    print(f" Test  Shape : {test_df.shape}")
    print(f" Val   Shape : {val_df.shape}")

    return train_df, val_df, test_df, freq_map, target_encoder

if __name__ == "__main__" :
    run_feature_engineering()

