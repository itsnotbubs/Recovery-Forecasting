import pandas as pd
import zipfile

def load_raw_options_data(path):    
    # Open the zip file and extract all parquet files
    dfs = []
    with zipfile.ZipFile(path, 'r') as zip_ref:
        # Get all parquet files in the zip
        parquet_files = [f for f in zip_ref.namelist() if f.endswith('.parquet')]
        
        # Read each parquet file and create a dataframe
        for parquet_file in parquet_files:
            with zip_ref.open(parquet_file) as f:
                df = pd.read_parquet(f)
                dfs.append(df)
                # print(f"Loaded {parquet_file}: {df.shape}")
    
    # Concatenate all dataframes
    concatenated_df = pd.concat(dfs, ignore_index=True)
    # print(f"\nConcatenated dataframe shape: {concatenated_df.shape}")
    # print(concatenated_df.head())
    
    return concatenated_df


def load_rf_data(path):
    # Load DGS3MO.csv and forward fill missing values
    dgs_df = pd.read_csv(path)
    # print("DGS3MO shape before forward fill:", dgs_df.shape)
    # print(dgs_df.head())

    # Assuming the csv has a date column and a value column
    # Adjust column names as needed
    dgs_df.columns = ['date', 'rate']
    dgs_df['date'] = pd.to_datetime(dgs_df['date'])

    # Track which values are original vs forward filled
    dgs_df['is_forward_filled'] = dgs_df['rate'].isna()
    dgs_df['rate'] = dgs_df['rate'].ffill()
    return dgs_df


def filter_rf_data(df, dgs_df):
    # Get unique dates from raw options data
    df['date'] = pd.to_datetime(df['date'])  # Adjust if date column name is different
    options_dates = df['date'].unique()

    # Keep only DGS3MO values that have corresponding dates in raw_options_data
    dgs_filtered = dgs_df[dgs_df['date'].isin(options_dates)].copy()
    dgs_filtered = dgs_filtered.sort_values('date').reset_index(drop=True)

    # print(f"\nFiltered DGS3MO (dates matching raw_options_data): {dgs_filtered.shape}")
    print(dgs_filtered[dgs_filtered['is_forward_filled']])
    print(f"\nForward filled values: {dgs_filtered['is_forward_filled'].sum()}")
    return dgs_filtered


def load_data(data_path="data/raw_options_data.zip", rf_path="data/DGS3MO.csv"):
    # Use relative path from project root (go up 2 levels from notebook location)
    df = load_raw_options_data(data_path)
    rf_raw = load_rf_data(rf_path)
    rf = filter_rf_data(df, rf_raw)
    # Ensure dates are datetime objects
    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    rf['date'] = pd.to_datetime(rf['date'])

    # Merge options data with risk-free rates
    df = df.merge(rf[['date', 'rate']], on='date', how='left')
    # print("Data shape after merging with risk-free rates:", df.shape)
    df['rate'] = df['rate'] / 100  # Convert percentage to decimal
    # print(df[['date', 'rate']].head())

    df['is_call'] = df['Call/Put'].apply(lambda x: True if x == 'C' else False)
    df['T'] = (df['expiration'] - df['date']).dt.days / 365.0

    return df