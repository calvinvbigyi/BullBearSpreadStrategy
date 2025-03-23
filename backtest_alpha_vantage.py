import requests
import pandas as pd
import time
from datetime import datetime
import os

def get_intraday_data(symbol, interval='1min', api_key=None, output_size='full'):
    """
    Fetch intraday data from Alpha Vantage API
    
    Parameters:
    symbol (str): Stock symbol (e.g., 'QQQ')
    interval (str): Time interval between data points (1min, 5min, 15min, 30min, 60min)
    api_key (str): Your Alpha Vantage API key
    output_size (str): 'compact' (latest 100 data points) or 'full' (up to 30 days of minute data)
    
    Returns:
    pandas.DataFrame: DataFrame containing the intraday data
    """
    if api_key is None:
        raise ValueError("API key is required")
    
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'apikey': api_key,
        'outputsize': output_size,
        'datatype': 'json'
    }
    
    print(f"Fetching {interval} data for {symbol}...")
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
    
    data = response.json()
    
    # Check for error messages
    if 'Error Message' in data:
        raise Exception(f"API returned an error: {data['Error Message']}")
    
    # Extract time series data
    time_series_key = f"Time Series ({interval})"
    if time_series_key not in data:
        print("Full response:", data)
        raise Exception(f"Expected key '{time_series_key}' not found in response")
    
    time_series = data[time_series_key]
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')
    
    # Rename columns
    df.columns = [col.split('. ')[1] for col in df.columns]
    
    # Convert values to float
    for col in df.columns:
        df[col] = df[col].astype(float)
    
    # Add symbol column
    df['symbol'] = symbol
    
    # Reset index and rename to 'timestamp'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df.sort_values('timestamp', inplace=True)
    
    return df

def save_to_csv(df, symbol, interval, directory='data'):
    """Save DataFrame to CSV file"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    today = datetime.now().strftime('%Y%m%d')
    filename = f"{directory}/{symbol}_{interval}_{today}.csv"
    
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    return filename

def fetch_multiple_months(symbol, interval='1min', api_key=None, months=12):
    """
    Fetch multiple months of data by using the slice parameter
    Note: Alpha Vantage limits this to a maximum of 24 months
    """
    all_data = []
    
    # Free tier has limitations, so we'll use 'compact' for testing
    first_batch = get_intraday_data(symbol, interval, api_key, 'full')
    all_data.append(first_batch)
    
    # If you need more historical data and have a premium API key:
    # Alpha Vantage offers intraday extended history with slices like 'year1month1'
    if months > 1 and interval == '1min':
        # Premium API functionality
        base_url = 'https://www.alphavantage.co/query'
        
        for year in range(1, (months // 12) + 1):
            for month in range(1, 13):
                if (year-1)*12 + month > months:
                    break
                    
                slice_param = f"year{year}month{month}"
                
                params = {
                    'function': 'TIME_SERIES_INTRADAY_EXTENDED',
                    'symbol': symbol,
                    'interval': interval,
                    'slice': slice_param,
                    'apikey': api_key
                }
                
                print(f"Fetching slice: {slice_param}")
                response = requests.get(base_url, params=params)
                
                if response.status_code == 200:
                    # This endpoint returns CSV directly
                    data = pd.read_csv(pd.StringIO(response.text))
                    all_data.append(data)
                    
                    # Respect API rate limits
                    time.sleep(15)  # Alpha Vantage has a rate limit of 5 calls per minute for free tier
                else:
                    print(f"Failed to fetch slice {slice_param}: {response.status_code}")
    
    # Combine all data
    if len(all_data) > 1:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Remove duplicates
        combined_data.drop_duplicates(subset=['timestamp'], inplace=True)
        # Sort by timestamp
        combined_data.sort_values('timestamp', inplace=True)
        return combined_data
    else:
        return all_data[0]

def analyze_data(df):
    """Perform basic analysis on the data"""
    # Daily statistics
    df['date'] = df['timestamp'].dt.date
    daily_stats = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Calculate daily returns
    daily_stats['daily_return'] = daily_stats['close'].pct_change() * 100
    
    # Calculate volatility (standard deviation of returns)
    volatility = daily_stats['daily_return'].std()
    
    # Print summary
    print("\nData Summary:")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of trading days: {len(daily_stats)}")
    print(f"Number of data points: {len(df)}")
    print(f"Average daily volume: {daily_stats['volume'].mean():.2f}")
    print(f"Daily volatility: {volatility:.2f}%")
    
    return daily_stats

# Main execution
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = "YOUR_API_KEY_HERE"
    
    try:
        # Get the most recent ~30 days of minute data (free tier limit)
        qqq_data = get_intraday_data('QQQ', '1min', API_KEY, 'full')
        
        # Save data to CSV
        csv_file = save_to_csv(qqq_data, 'QQQ', '1min')
        
        # Analyze the data
        daily_stats = analyze_data(qqq_data)
        
        # For premium API users who need more historical data:
        # Uncomment the following line to get multiple months of data
        # extended_data = fetch_multiple_months('QQQ', '1min', API_KEY, months=6)
        # save_to_csv(extended_data, 'QQQ', '1min_extended')
        
        print(f"\nSuccessfully retrieved {len(qqq_data)} minute-level data points for QQQ")
        print(f"Data saved to {csv_file}")
        
    except Exception as e:
        print(f"Error: {e}")