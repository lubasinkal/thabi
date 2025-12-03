"""
Synthetic Claims Data Generator for Flood Insurance
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_claims(
    n_years: int = 5,
    base_frequency: float = 50,
    threshold: float = 25000,
    pareto_scale: float = 25000,
    pareto_shape: float = 1.8,
    climate_severity_factor: float = 0.3,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic flood claims data with climate correlation
    
    Args:
        n_years: Number of years of data to generate
        base_frequency: Average number of claims per year
        threshold: Minimum claim size to include (deductible)
        pareto_scale: Scale parameter K for Pareto distribution
        pareto_shape: Shape parameter q for Pareto distribution
        climate_severity_factor: How much climate affects claim severity (0-1)
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: date, loss_amount, SSI, rainfall_7day
    """
    np.random.seed(random_seed)
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    n_claims = int(base_frequency * n_years)
    
    # Generate random claim dates across the period
    days_range = n_years * 365
    random_days = np.random.randint(0, days_range, n_claims)
    claim_dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    # Generate synthetic climate variables (SSI and rainfall)
    # Higher values = more severe flood conditions
    base_ssi = np.random.normal(0.25, 0.05, n_claims)
    base_ssi = np.clip(base_ssi, 0.1, 0.5)
    
    # Add seasonal pattern (more floods in rainy season)
    day_of_year = np.array([d.timetuple().tm_yday for d in claim_dates])
    seasonal_factor = 0.1 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
    ssi = base_ssi + seasonal_factor
    ssi = np.clip(ssi, 0.1, 0.6)
    
    # Generate rainfall data correlated with SSI
    rainfall_7day = 20 + 100 * ssi + np.random.normal(0, 10, n_claims)
    rainfall_7day = np.clip(rainfall_7day, 0, 200)
    
    # Generate loss amounts with climate adjustment
    # Higher SSI → lower q → heavier tail → larger claims
    climate_adjusted_shape = pareto_shape * (1 - climate_severity_factor * (ssi - ssi.mean()) / ssi.std())
    climate_adjusted_shape = np.clip(climate_adjusted_shape, 0.8, 2.5)
    
    # Generate Pareto-distributed losses
    losses = []
    for q in climate_adjusted_shape:
        # Pareto Type II: loss = K * (u^(-1/q) - 1)
        u = np.random.uniform(0, 1)
        loss = pareto_scale * (u ** (-1/q))
        losses.append(loss)
    
    losses = np.array(losses)
    
    # Filter to only include claims above threshold
    mask = losses > threshold
    
    df = pd.DataFrame({
        'date': np.array(claim_dates)[mask],
        'loss_amount': losses[mask],
        'SSI': ssi[mask],
        'rainfall_7day': rainfall_7day[mask]
    })
    
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    # Generate and display sample data
    claims = generate_synthetic_claims(n_years=5, random_seed=42)
    print(f"\nGenerated {len(claims)} synthetic flood claims")
    print(f"\nSummary Statistics:")
    print(claims.describe())
    print(f"\nTotal Losses: ${claims['loss_amount'].sum():,.2f}")
    print(f"Average Claim: ${claims['loss_amount'].mean():,.2f}")
    print(f"Max Claim: ${claims['loss_amount'].max():,.2f}")
