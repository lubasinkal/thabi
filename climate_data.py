"""
Climate Data Module - Fetches rainfall, soil moisture, and flood indices for Molapo region
"""
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


def get_climate_data(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch climate data for flood risk modeling
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with daily climate variables
    """
    cache_session = requests_cache.CachedSession('.cache.sqlite', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "precipitation_sum",
            "rain_sum",
            "temperature_2m_max",
            "temperature_2m_min"
        ]
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    daily = response.Daily()
    
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "precipitation_mm": daily.Variables(0).ValuesAsNumpy(),
        "rainfall_mm": daily.Variables(1).ValuesAsNumpy(),
        "temp_max": daily.Variables(2).ValuesAsNumpy(),
        "temp_min": daily.Variables(3).ValuesAsNumpy()
    }
    
    df = pd.DataFrame(data=daily_data)
    
    # Calculate 7-day rolling rainfall for flood indicator
    df['rainfall_7day'] = df['rainfall_mm'].rolling(window=7, min_periods=1).sum()
    
    # Calculate Soil Saturation Index (SSI) proxy from cumulative precipitation
    # Higher recent rainfall = higher soil saturation
    df['rainfall_30day'] = df['rainfall_mm'].rolling(window=30, min_periods=1).sum()
    # Normalize SSI to roughly 0-1 range (assuming max 30-day rainfall ~300mm)
    df['SSI'] = (df['rainfall_30day'] / 300).clip(0, 1)
    
    return df
