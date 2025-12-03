"""
Test fetching REAL climate data from Open-Meteo
"""
from climate_data import get_climate_data
import pandas as pd

# Molapo region coordinates (example - replace with actual)
MOLAPO_LAT = -25.7479  # Example: Botswana region
MOLAPO_LON = 25.9269

print("Fetching REAL climate data for Molapo region...")
print(f"Coordinates: {MOLAPO_LAT}°, {MOLAPO_LON}°")
print(f"Period: 2020-01-01 to 2024-12-31\n")

climate_df = get_climate_data(
    latitude=MOLAPO_LAT,
    longitude=MOLAPO_LON,
    start_date="2020-01-01",
    end_date="2024-12-31"
)

print(f"✓ Fetched {len(climate_df)} days of climate data\n")
print("Sample data:")
print(climate_df.head(10))

print("\n" + "="*70)
print("Climate Statistics")
print("="*70)
print(climate_df[['precipitation_mm', 'rainfall_mm', 'SSI', 'rainfall_7day']].describe())

print("\n" + "="*70)
print("Extreme Events (High SSI)")
print("="*70)
extreme_days = climate_df.nlargest(10, 'SSI')
print(extreme_days[['date', 'SSI', 'rainfall_7day', 'precipitation_mm']])

# Save to CSV
climate_df.to_csv('real_molapo_climate_data.csv', index=False)
print(f"\n✓ Saved to: real_molapo_climate_data.csv")
