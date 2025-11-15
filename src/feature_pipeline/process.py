import pandas as pd
from pathlib import Path
import re
"""
This module provides functions to process the dataset by handling missing values,

- cleans and normalizes City Name
- Map Cities to Metro Areas and City to Latitude and Longitude
- Drops duplicate rows
- Saves the processed data to a new CSV file

"""


# Data directory paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Manual fixes for known mismatches (normalized form)
CITY_MAPPING = {
    "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
    "denver-aurora-lakewood": "denver-aurora-centennial",
    "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
    "austin-round rock-georgetown": "austin-round rock-san marcos",
    "miami-fort lauderdale-pompano beach": "miami-fort lauderdale-west palm beach",
    "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
    "dc_metro": "washington-arlington-alexandria",
    "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
}

def normalize_city(city: str) -> str:
    """
    Normalize city names by converting to lowercase, stripping whitespace,
    and replacing hyphens with spaces.

    Args:
        city (str): The city name to normalize.
    Returns:
        str: The normalized city name.
    """
    if pd.isna(city):
        return city
    city = city.strip().lower()
    city = re.sub(r"-", " ", city) # Replace hyphens with spaces
    city = re.sub(r"\s+", " ", city)  # Replace multiple spaces with a single space
    return city

def clean_and_merge(df : pd.DataFrame, metros_path : str|None = r'data/raw/usmetros.csv') -> pd.DataFrame:

    if "city_full" not in df.columns:
        print("Column 'city_full' not found in DataFrame.")
        return df
    
    # Normalize city names
    df["city_full"] = df["city_full"].apply(normalize_city)
    norm_mapping = {k: normalize_city(v) for k, v in CITY_MAPPING.items()}

    # Apply Map
    df["city_full"] = df["city_full"].replace(norm_mapping)

    # check if lat and lng columns already exist
    if {"lat", "lng"}.issubset(df.columns):
        print("Latitude and Longitude columns already exist. Skipping merge.")
        return df
    
    # Load metro area data
    if not metros_path or not Path(metros_path).exista():
        print(f"Metro area file not found at {metros_path}. Skipping merge.")
        return df
    
    # check if required columns exist in metros data
    metros = pd.read_csv(metros_path)
    if "metro_full" not in metros.columns or {"lag", "lng"}.issubset(metros.columns) == False:
        print("Required columns not found in metro area data. Skipping merge.")
        return df
    
    metros["metro_full"] = metros["metro_full"].apply(normalize_city)
    df = df.merge(metros[["metro_full", "lat", "lng"]], how="left", left_on="city_full", right_on="metro_full")
    df.drop(columns=["metro_full"], inplace=True)

    missing  = df[df["lat"].isna() | df["lng"].isna()]["city_full"].unique()
    if len(missing) > 0:
        print(f"Warning: Missing lat/lng for cities: {missing}")
    else:
        print("All cities successfully mapped to lat/lng.")

    return df

def drop_duplicates(df:pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows from the DataFrame, except for 'date' and 'year' columns.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    before = df.shape[0]
    df = df.drop_duplicates(subset= df.columns.difference(['date','year']), keep='first')
    after = df.shape[0] 
    print(f"Dropped {before - after} duplicate rows except 'date' and 'year' columns.")
    return df

def remove_outliers(df:pd.DataFrame) -> pd.DataFrame:

    if "median_list_price" not in df.columns:
        print("Column 'median_list_price' not found in DataFrame.")
        return df
    before = df.shape[0]
    df = df[df["median_list_price"] <= 19_000_000].copy()
    after = df.shape[0]
    print(f"Removed {before - after} outlier rows based on 'median_list_price'.")
    return df

def process_split(
        split: str,
        raw_dir: Path | str = RAW_DIR,
        processed_dir: Path | str = PROCESSED_DIR,
        metros_path: str|None = r'data/raw/usmetros.csv'
        ) -> pd.DataFrame:
     
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir) 
    processed_dir.mkdir(parents=True, exist_ok=True)
    input_path = raw_dir / f"{split}.csv"

    df = pd.read_csv(input_path)
    df = clean_and_merge(df, metros_path)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    output_path = processed_dir / f"{split}_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed {split} data saved to {output_path} (shape: {df.shape})")

    return df

def run_preprocess(
        splits: tuple[str] = ("train", "val", "test"),
        raw_dir: Path | str = RAW_DIR,
        processed_dir: Path | str = PROCESSED_DIR,
        metros_path: str|None = r'data/raw/usmetros.csv'
        ):
    for split in splits:
        process_split(split, raw_dir, processed_dir, metros_path)

if __name__ == "__main__":
    run_preprocess()