"""
Data loading and preprocessing module.
Handles reading claims data and climate data from files.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def load_claims_data(file_path: Path) -> pd.DataFrame:
    """
    Load claims data from Excel file and perform basic cleaning.
    
    Args:
        file_path: Path to the claims Excel file
        
    Returns:
        Cleaned DataFrame with claims data
        
    Raises:
        FileNotFoundError: If the claims file doesn't exist
        ValueError: If required columns are missing
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Claims file not found: {file_path}")
    
    print(f"Loading claims data from: {file_path}")
    
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Rename columns to standardized names
    if 'Total Claim' in df.columns:
        df = df.rename(columns={'Total Claim': 'claim_amount'})
    elif 'Claim Cost' in df.columns:
        df = df.rename(columns={'Claim Cost': 'claim_amount'})
    else:
        raise ValueError("No claim amount column found (expected 'Total Claim' or 'Claim Cost')")
    
    # Convert claim amounts to numeric (handle any formatting issues)
    df['claim_amount'] = pd.to_numeric(df['claim_amount'], errors='coerce')
    
    # Parse dates if available
    if 'LossDate' in df.columns:
        df['LossDate'] = pd.to_datetime(df['LossDate'], errors='coerce')
        df['year'] = df['LossDate'].dt.year
    else:
        # If no date, assume all claims from 2025 (Feb flooding event)
        df['year'] = 2025
    
    # Remove rows with missing claim amounts
    df = df.dropna(subset=['claim_amount'])
    
    # Filter out invalid claims (negative or zero)
    df = df[df['claim_amount'] > 0]
    
    print(f"   Loaded {len(df)} valid claims")
    print(f"   Claim range: ${df['claim_amount'].min():,.2f} to ${df['claim_amount'].max():,.2f}")
    print(f"   Mean claim: ${df['claim_amount'].mean():,.2f}")
    print(f"   Median claim: ${df['claim_amount'].median():,.2f}")
    
    return df


def load_climate_data(file_path: Path) -> pd.DataFrame:
    """
    Load historical climate data from CSV file.
    
    Args:
        file_path: Path to the climate CSV file
        
    Returns:
        DataFrame with climate data (temperature and rainfall)
        
    Raises:
        FileNotFoundError: If the climate file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Climate file not found: {file_path}")
    
    print("Loading climate data...")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Extract required columns
    required_cols = ['Year', 'Annual_mean_max_C', 'Annual_total_mm']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with missing data
    df = df.dropna(subset=required_cols)
    
    # Handle year format (e.g., "2024/25" -> 2024)
    if df['Year'].dtype == 'object':
        df['Year'] = df['Year'].str.split('/').str[0].astype(int)
    
    print(f"   Loaded {len(df)} years of climate data")
    print(f"   Temperature range: {df['Annual_mean_max_C'].min():.1f} - {df['Annual_mean_max_C'].max():.1f} C")
    print(f"   Rainfall range: {df['Annual_total_mm'].min():.1f} - {df['Annual_total_mm'].max():.1f} mm")
    
    return df


def add_climate_to_claims(
    claims_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Add climate variables to claims data using bootstrap sampling.
    
    Since all claims are from a single event (Feb 2025 flooding), we use
    historical climate data via bootstrap sampling to add temporal variation
    for climate-adjusted modeling.
    
    Args:
        claims_df: DataFrame with claims data
        climate_df: DataFrame with historical climate data
        random_seed: Random seed for reproducibility
        
    Returns:
        Claims DataFrame with added climate variables
    """
    print("Adding climate variables to claims (bootstrap sampling)...")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Sample climate data (with replacement) to match number of claims
    n_claims = len(claims_df)
    climate_sample = climate_df.sample(n=n_claims, replace=True, random_state=random_seed)
    
    # Add climate variables to claims
    result = claims_df.copy()
    result['temperature'] = climate_sample['Annual_mean_max_C'].values
    result['rainfall'] = climate_sample['Annual_total_mm'].values
    
    # Calculate standardized climate variables (z-scores)
    # Standardization: (x - mean) / std_dev
    temp_mean = climate_df['Annual_mean_max_C'].mean()
    temp_std = climate_df['Annual_mean_max_C'].std()
    rain_mean = climate_df['Annual_total_mm'].mean()
    rain_std = climate_df['Annual_total_mm'].std()
    
    result['temp_std'] = (result['temperature'] - temp_mean) / temp_std
    result['rain_std'] = (result['rainfall'] - rain_mean) / rain_std
    
    print(f"   Temperature range: {result['temperature'].min():.1f} - {result['temperature'].max():.1f} C")
    print(f"   Rainfall range: {result['rainfall'].min():.1f} - {result['rainfall'].max():.1f} mm")
    
    return result


def get_pareto_scale_parameter(claims_df: pd.DataFrame) -> float:
    """
    Determine K (scale parameter) for Type I Pareto distribution.
    
    For Type I Pareto, K is the minimum possible value the distribution can take.
    We set K = minimum observed claim value.
    
    Args:
        claims_df: DataFrame with 'claim_amount' column
        
    Returns:
        K value (scale parameter)
    """
    k_value = claims_df['claim_amount'].min()
    
    print("\n=== Type I Pareto Distribution: Scale Parameter K ===")
    print(f"K (minimum observed claim): ${k_value:,.2f}")
    print("Note: For Type I Pareto, K is the lower bound of the distribution")
    print("      All claims must satisfy: claim >= K")
    
    return k_value


def calculate_reinsurance_layer(claims_df: pd.DataFrame, percentile: float = 75) -> Tuple[float, float]:
    """
    Calculate reinsurance layer parameters (separate from Pareto K).
    
    The reinsurance layer deductible is a business decision, typically set at
    a high percentile of claims to protect against large losses.
    
    Args:
        claims_df: DataFrame with 'claim_amount' column
        percentile: Percentile to use for layer deductible (default: 75th)
        
    Returns:
        Tuple of (layer_deductible, layer_cover)
    """
    layer_deductible = np.percentile(claims_df['claim_amount'], percentile)
    
    print(f"\n=== Reinsurance Layer (Business Parameters) ===")
    print(f"Layer deductible: ${layer_deductible:,.2f} ({percentile}th percentile)")
    print("Note: This is separate from Pareto parameter K")
    
    return layer_deductible
