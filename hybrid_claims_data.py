"""
Hybrid Claims Data: Synthetic claims matched with REAL climate data
"""
import pandas as pd
import numpy as np
from climate_data import get_climate_data
from datetime import datetime, timedelta


def generate_claims_with_real_climate(
    latitude: float,
    longitude: float,
    n_years: int = 5,
    base_frequency: float = 50,
    threshold: float = 25000,
    pareto_scale: float = 25000,
    pareto_shape: float = 1.8,
    climate_severity_factor: float = 0.3,
    random_seed: int = 42
):
    """
    Generate synthetic flood claims correlated with REAL climate data
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        n_years: Number of years of data
        base_frequency: Average claims per year
        threshold: Minimum claim size
        pareto_scale: Pareto scale parameter K
        pareto_shape: Baseline Pareto shape q
        climate_severity_factor: How much climate affects severity
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic claims + real climate data
    """
    np.random.seed(random_seed)
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_years * 365)
    
    print(f"\n{'='*70}")
    print(f"Fetching REAL climate data for Molapo")
    print(f"{'='*70}")
    print(f"Coordinates: {latitude}°, {longitude}°")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Fetch real climate data
    climate_df = get_climate_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    print(f"✓ Fetched {len(climate_df)} days of real climate data")
    print(f"  Mean SSI: {climate_df['SSI'].mean():.3f}")
    print(f"  Max SSI: {climate_df['SSI'].max():.3f}")
    print(f"  Mean 7-day rainfall: {climate_df['rainfall_7day'].mean():.1f}mm")
    
    # Generate claim dates weighted by high SSI periods (more claims during wet periods)
    climate_df['date_only'] = pd.to_datetime(climate_df['date']).dt.date
    
    # Calculate probability weights: higher SSI = higher probability of claim
    # Normalize SSI to create probability distribution
    weights = (climate_df['SSI'] + 0.1) ** 2  # Quadratic to emphasize high SSI days
    weights = weights / weights.sum()
    
    # Sample claim dates based on climate conditions
    n_claims = int(base_frequency * n_years)
    claim_indices = np.random.choice(
        len(climate_df), 
        size=n_claims, 
        replace=True,
        p=weights
    )
    
    # Get climate data for sampled dates
    claim_climate = climate_df.iloc[claim_indices].copy()
    claim_climate = claim_climate.reset_index(drop=True)
    
    # Generate loss amounts correlated with real SSI values
    ssi_values = claim_climate['SSI'].values
    
    # Climate-adjusted Pareto shape
    # Higher SSI → Lower q → Heavier tail → Larger losses
    climate_adjusted_shape = pareto_shape * (1 - climate_severity_factor * (ssi_values - ssi_values.mean()) / ssi_values.std())
    climate_adjusted_shape = np.clip(climate_adjusted_shape, 0.8, 2.5)
    
    # Generate Pareto-distributed losses
    losses = []
    for q in climate_adjusted_shape:
        u = np.random.uniform(0, 1)
        loss = pareto_scale * (u ** (-1/q))
        losses.append(loss)
    
    losses = np.array(losses)
    
    # Filter to only include claims above threshold
    mask = losses > threshold
    
    # Create final dataframe
    claims_df = pd.DataFrame({
        'date': pd.to_datetime(claim_climate.loc[mask, 'date']).dt.tz_localize(None),
        'loss_amount': losses[mask],
        'SSI': claim_climate.loc[mask, 'SSI'].values,
        'rainfall_7day': claim_climate.loc[mask, 'rainfall_7day'].values,
        'precipitation_mm': claim_climate.loc[mask, 'precipitation_mm'].values,
        'temp_max': claim_climate.loc[mask, 'temp_max'].values
    })
    
    claims_df = claims_df.sort_values('date').reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print(f"Generated {len(claims_df)} synthetic claims with REAL climate data")
    print(f"{'='*70}")
    print(f"Total losses: ${claims_df['loss_amount'].sum():,.2f}")
    print(f"Average claim: ${claims_df['loss_amount'].mean():,.2f}")
    print(f"Max claim: ${claims_df['loss_amount'].max():,.2f}")
    print(f"\nClaims by year:")
    claims_by_year = claims_df.groupby(claims_df['date'].dt.year).size()
    for year, count in claims_by_year.items():
        print(f"  {year}: {count} claims")
    
    print(f"\nClimate statistics for claim dates:")
    print(f"  Mean SSI: {claims_df['SSI'].mean():.3f} (higher than overall due to weighting)")
    print(f"  Mean rainfall (7-day): {claims_df['rainfall_7day'].mean():.1f}mm")
    
    return claims_df


if __name__ == "__main__":
    # Test with Molapo coordinates
    claims = generate_claims_with_real_climate(
        latitude=-24.6545,
        longitude=25.9086,
        n_years=5,
        random_seed=42
    )
    
    print(f"\nSample claims:")
    print(claims.head(10))
    
    # Save to CSV
    claims.to_csv('hybrid_claims_real_climate.csv', index=False)
    print(f"\n✓ Saved to: hybrid_claims_real_climate.csv")
