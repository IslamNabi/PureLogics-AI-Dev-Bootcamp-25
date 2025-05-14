import pandas as pd
import json
import os

def load_data(filepath):
    """Load weather data from CSV file."""
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        print("Data loaded successfully")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def clean_data(df):
    """Clean the weather data by handling missing values."""
    try:
        # Fill numeric columns with their means
        numeric_cols = ['temperature_C', 'humidity_%', 'wind_speed_kmph', 
                       'pressure_hPa', 'visibility_km']
        df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))
        
        print("Data cleaned successfully")
        return df
    except Exception as e:
        raise Exception(f"Error cleaning data: {str(e)}")

def compute_summary(df):
    """Compute summary statistics for the weather data."""
    try:
        numeric_cols = ['temperature_C', 'humidity_%', 'wind_speed_kmph', 
                       'pressure_hPa', 'visibility_km']
        
        summary = {
            'period': {
                'start': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'),
                'days': (df['timestamp'].max() - df['timestamp'].min()).days
            },
            'statistics': {}
        }
        
        for col in numeric_cols:
            summary['statistics'][col] = {
                'mean': round(df[col].mean(), 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2)
            }
        
        print("Summary computed successfully")
        return summary
    except Exception as e:
        raise Exception(f"Error computing summary: {str(e)}")

def save_summary(summary, output_path):
    """Write summary dictionary to a text file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Summary saved to {output_path}")
    except Exception as e:
        raise Exception(f"Error saving summary: {str(e)}")