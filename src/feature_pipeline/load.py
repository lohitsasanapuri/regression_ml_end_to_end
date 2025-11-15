import pandas as pd
from pathlib import Path

"""
This module provides functions to load and split the dataset into training, validation, and test sets.
it reads raw data from a csv file, splits it into train, validation, and test sets, and saves these sets as separate csv files.

"""
DATA_DIR = Path("data/raw")

def load_and_split(
        raw_data_path: str = r'data/raw/HouseTS.csv',
        output_dir: Path = DATA_DIR): 
    # Load the raw data
    df = pd.read_csv(raw_data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Cutoff dates for splitting
    train_cutoff = "2020-01-01"
    val_cutoff = "2022-01-01"

    # Split the data
    train_df = df[df['date'] < train_cutoff]
    val_df = df[(df['date'] >= train_cutoff) & (df['date'] < val_cutoff)]
    test_df = df[df['date'] >= val_cutoff]

    # Save the splits
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / 'train.csv', index=False)
    val_df.to_csv(outdir / 'val.csv', index=False)
    test_df.to_csv(outdir / 'test.csv', index=False)
    print(f"Data split into train, val, and test sets and saved to {outdir}")
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    return train_df, val_df, test_df

if __name__ == "__main__":
    load_and_split()


