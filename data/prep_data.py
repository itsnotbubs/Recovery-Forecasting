import numpy as np
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
    df['T'] = (df['expiration'] - df['date']).dt.days/ 365.25

    return df


def market_returns(time_horizons=[7, 7 * 2, 7 * 4, 7 * 4 * 2]):
    df = load_data()
    price_by_date = (
        df[['date', 'Adjusted close']]
        .drop_duplicates()
        .assign(date=lambda d: pd.to_datetime(d['date']))
        .sort_values('date')
        .rename(columns={'Adjusted close': 'adj_close', 'date': 'trading_date'})
    )

    # Prepare observations (unique date, adjusted close, rate pairs)
    df_prices = (
        df[['date', 'Adjusted close', 'rate']]
        .drop_duplicates()
        .copy()
    )
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    df_prices = df_prices.rename(columns={'Adjusted close': 'adj_close_start'})

    # Build results for each time horizon
    results_by_horizon = {}

    for horizon_days in time_horizons:
        horizon_results = []
        
        for idx, row in df_prices.iterrows():
            start_date = row['date']
            start_price = row['adj_close_start']
            risk_free_rate = row['rate']  # annualized rate in decimal form
            
            # Target date is start_date + horizon_days
            target_date = start_date + pd.Timedelta(days=horizon_days)
            
            # Find the closest trading date to the target date using merge_asof
            # Create a temporary df for this search
            temp_search = price_by_date[['trading_date', 'adj_close']].copy()
            temp_search = temp_search.sort_values('trading_date')
            
            # merge_asof with direction='nearest' to find closest date
            match = pd.merge_asof(
                pd.DataFrame({'target': [target_date]}),
                temp_search.rename(columns={'trading_date': 'target'}),
                on='target',
                direction='backward'
            )
            
            if not match.empty and pd.notna(match['adj_close'].iloc[0]):
                horizon_date = match['target'].iloc[0]
                horizon_price = match['adj_close'].iloc[0]
                days_diff = (horizon_date - start_date).days
                
                # Compute log return
                log_return = np.log(horizon_price / start_price)
                
                # Compute accrued risk-free return over the time horizon
                # rf_return = rate * (days_diff / 365.25)
                rf_return = risk_free_rate * (days_diff / 365.25)
                
                # Excess return: subtract risk-free return from asset return
                excess_return = log_return - rf_return
                
                horizon_results.append({
                    'date': start_date,
                    'adj_close': start_price,
                    'rate': risk_free_rate,
                    'horizon_days': horizon_days,
                    'horizon_date': horizon_date,
                    'adj_close_horizon': horizon_price,
                    'days_diff': days_diff / 365.25,
                    'log_return': log_return,
                    'rf_return': rf_return,
                    'excess_return': excess_return,
                })
        
        results_by_horizon[horizon_days] = pd.DataFrame(horizon_results)

    # Concatenate all horizons into one dataframe
    returns_by_horizon_df = pd.concat(results_by_horizon.values(), ignore_index=True)

    # Summary and quick checks
    print(f"Total observations across all horizons: {len(returns_by_horizon_df)}")
    print(f"\nBreakdown by time horizon:")
    print(returns_by_horizon_df.groupby('horizon_days').size())
    print(f"\nSample data:")
    print(returns_by_horizon_df.head(5))
    print(f"\nLog return statistics by horizon:")
    print(returns_by_horizon_df.groupby('horizon_days')['log_return'].describe())

    return returns_by_horizon_df