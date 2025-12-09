"""
Configuration file for flood risk analysis.
Contains all constants and parameters used throughout the project.
"""
from pathlib import Path

# Data file paths
DATA_DIR = Path("data")
CLAIMS_FILE = DATA_DIR / "CAT Claims.csv"
CLIMATE_FILE = DATA_DIR / "temp_hazard.csv"

# Output file names
OUTPUT_PRICING = "molapo_reinsurance_pricing.csv"
OUTPUT_CLAIMS = "molapo_claims_processed.csv"
OUTPUT_SCENARIOS = "climate_scenarios.csv"
OUTPUT_PLOT = "mean_excess_plot.png"

# Reinsurance layer parameters
LAYER_COVER = 475_000  # Maximum coverage amount in PULA

# Random seed for reproducibility
RANDOM_SEED = 42

# Climate variables for modeling
CLIMATE_VARS = ['temp_std', 'rain_std']

# Risk confidence levels for VaR and TVaR calculations
CONFIDENCE_LEVELS = [0.95, 0.99, 0.995]

# Model parameters
FFT_NODES = 2**17  # Number of nodes for FFT in aggregate loss distribution
SIMULATION_SIZE = 1_000_000  # Number of simulations for TVaR calculation
