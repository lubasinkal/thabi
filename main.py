"""
Flood Risk Model for Molapo Insurance
Main entry point for climate-adjusted reinsurance pricing
Uses actual claims data and historical climate data
"""
from flood_risk_model import FloodRiskModel
from reinsurance_pricing import ReinsurancePricer
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    """
    Run complete flood risk analysis pipeline:
    1. Load actual claims and climate data from files
    2. Estimate baseline and climate-adjusted Pareto parameters
    3. Price reinsurance contracts under different scenarios
    4. Generate reports and visualizations
    """
    print("="*70)
    print("MOLAPO FLOOD RISK MODEL - INSURANCE PRICING")
    print("Using Actual Claims and Historical Climate Data")
    print("="*70)

    # Configuration
    DATA_DIR = Path("data")
    
    # Step 1: Load actual claims data from Excel file
    print("\n[1/5] Loading actual claims data from file...")
    claims_raw = pd.read_excel(DATA_DIR / "CAT Claims.csv")

    # Clean and prepare claims data
    claims_raw.columns = claims_raw.columns.str.strip()
    
    # Map to expected column names
    # The file has: Total Claim, Excess (deductible), Amount Paid, LossDate
    if 'Total Claim' in claims_raw.columns:
        claims_raw = claims_raw.rename(columns={'Total Claim': 'Claim Cost'})
    
    # Clean claim amounts
    claims_raw['Claim Cost'] = pd.to_numeric(claims_raw['Claim Cost'], errors='coerce')
    
    # Clean date column
    if 'LossDate' in claims_raw.columns:
        claims_raw['LossDate'] = pd.to_datetime(claims_raw['LossDate'], errors='coerce')
    
    # Remove rows with missing critical data
    claims_raw = claims_raw.dropna(subset=['Claim Cost'])
    
    print(f"   Loaded {len(claims_raw)} claims")
    print(f"   Claim range: ${claims_raw['Claim Cost'].min():,.2f} to ${claims_raw['Claim Cost'].max():,.2f}")
    print(f"   Mean claim: ${claims_raw['Claim Cost'].mean():,.2f}")
    print(f"   Median claim: ${claims_raw['Claim Cost'].median():,.2f}")
    
    # For Type I Pareto distribution: K is the minimum possible value (scale parameter)
    # Not a threshold we choose, but the minimum of the distribution
    print("\n   === Type I Pareto Distribution Parameter ===")
    print("   K (scale parameter) = minimum possible claim value")
    
    # Option 1: Use minimum observed value
    K_min_observed = claims_raw['Claim Cost'].min()
    
    # Option 2: Set slightly below minimum to allow for potential smaller claims
    # Common practice: Use 95-99% of minimum
    K_conservative = K_min_observed * 0.95
    
    print(f"   Minimum observed claim: ${K_min_observed:,.2f}")
    print(f"   Conservative K (95% of min): ${K_conservative:,.2f}")
    
    # For research: Use minimum observed as K (most defensible)
    THRESHOLD = K_min_observed
    n_claims_for_fit = len(claims_raw)
    
    print(f"\n   Selected K = ${THRESHOLD:,.2f} (minimum observed claim)")
    print(f"   Number of claims for Pareto fitting: {n_claims_for_fit} (all claims)")
    print(f"   Justification: Type I Pareto definition (x >= K)")
    
    # For reinsurance layer, use a different deductible based on business needs
    # This is separate from the Pareto parameter K
    LAYER_DEDUCTIBLE = np.percentile(claims_raw['Claim Cost'], 75)  # Business decision
    LAYER_COVER = 475000
    
    print(f"\n   Reinsurance layer (separate from Pareto K):")
    print(f"   Layer deductible: ${LAYER_DEDUCTIBLE:,.2f}")
    print(f"   Layer cover: ${LAYER_COVER:,.0f}")

    # Extract year from LossDate (if available)
    if 'LossDate' in claims_raw.columns and claims_raw['LossDate'].notna().any():
        claims_raw['Year'] = claims_raw['LossDate'].dt.year
    else:
        # All claims are from Feb 2025 flooding
        claims_raw['Year'] = 2025
    
    # Check if climate data file exists for climate-adjusted modeling
    climate_file = DATA_DIR / "temp_hazard.csv"
    use_climate_modeling = climate_file.exists()
    
    if use_climate_modeling:
        print("   Loading historical climate data for climate-adjusted modeling...")
        climate_data = pd.read_csv(climate_file)
        climate_data.columns = climate_data.columns.str.strip()
        climate_data = climate_data.dropna(subset=['Year', 'Annual_mean_max_C', 'Annual_total_mm'])

        # Extract year number from Year column (format: "2024/25" -> 2024)
        if climate_data['Year'].dtype == 'object':
            climate_data['Year'] = climate_data['Year'].str.split('/').str[0].astype(int)

        # Since all claims are from a single event (Feb 2025 flooding),
        # use bootstrap sampling from historical climate to add temporal variation
        np.random.seed(42)
        n_claims = len(claims_raw)
        climate_sample = climate_data.sample(n=n_claims, replace=True, random_state=42)

        claims = claims_raw.copy()
        claims['Annual_mean_max_C'] = climate_sample['Annual_mean_max_C'].values
        claims['Annual_total_mm'] = climate_sample['Annual_total_mm'].values

        print(f"   Assigned historical climate data (bootstrap sampling)")
        print(f"   Climate range: Temp {claims['Annual_mean_max_C'].min():.1f}-{claims['Annual_mean_max_C'].max():.1f}°C, Rain {claims['Annual_total_mm'].min():.1f}-{claims['Annual_total_mm'].max():.1f}mm")

        # Create standardized climate variables
        temp_mean = climate_data['Annual_mean_max_C'].mean()
        temp_std = climate_data['Annual_mean_max_C'].std()
        rain_mean = climate_data['Annual_total_mm'].mean()
        rain_std = climate_data['Annual_total_mm'].std()

        claims['temp_std'] = (claims['Annual_mean_max_C'] - temp_mean) / temp_std
        claims['rain_std'] = (claims['Annual_total_mm'] - rain_mean) / rain_std
    else:
        print("   No climate data file found - skipping climate-adjusted modeling")
        claims = claims_raw.copy()

    # Rename for compatibility with existing model
    if 'Claim Cost' in claims.columns:
        claims = claims.rename(columns={'Claim Cost': 'loss_amount'})

    # Calculate N_YEARS from data
    N_YEARS = claims['Year'].nunique()

    print(f"   Total claims in dataset: {len(claims)}")
    print(f"   Claims range: {claims['loss_amount'].min():,.2f} to {claims['loss_amount'].max():,.2f} PULA")

    # Step 2: Estimate Pareto parameters
    print("\n[2/5] Estimating Pareto parameters...")
    model = FloodRiskModel(threshold=THRESHOLD)

    # Generate mean excess plot for threshold validation
    print("   Generating Mean Excess Plot for threshold validation...")
    try:
        model.mean_excess_plot(claims)
        print("   [OK] Mean Excess Plot saved to: mean_excess_plot.png")
        print("   Review this plot to validate threshold selection - look for linearity")
    except Exception as e:
        print(f"   Warning: Could not generate mean excess plot: {e}")

    # Baseline q using MLE
    q_baseline = model.estimate_baseline_q(claims)

    # Climate-adjusted q (if climate data available)
    if use_climate_modeling and 'temp_std' in claims.columns:
        # Climate-adjusted q using GLM with standardized temperature and rainfall
        model_results = model.estimate_climate_adjusted_q(claims, climate_vars=['temp_std', 'rain_std'])

        print("\n--- GLM Model Coefficients (ANOVA-like Table) ---")
        print(model_results['model'].summary().tables[1].as_text())
        print("-------------------------------------------------")

        # Scenario analysis
        scenarios = model.scenario_analysis(claims, climate_vars=['temp_std', 'rain_std'])
    else:
        print("\n--- Skipping climate-adjusted modeling (no climate data) ---")
        model_results = None
        scenarios = pd.DataFrame({
            'Scenario': ['Baseline (Static)'],
            'q_estimate': [q_baseline]
        })

    # Step 3: Get q values for pricing
    print("\n[3/5] Calculating q parameters for pricing scenarios...")
    # Retrieve q values from the scenario analysis results.
    q_baseline = model.baseline_q
    
    if use_climate_modeling and model_results is not None:
        q_mean = scenarios[scenarios['Scenario'] == 'Average Climate']['q_estimate'].iloc[0]
        q_p95 = scenarios[scenarios['Scenario'] == '95th Percentile (Adverse)']['q_estimate'].iloc[0]
        q_p99 = scenarios[scenarios['Scenario'] == '99th Percentile (Extreme)']['q_estimate'].iloc[0]

        print("\n--- Pareto q Values for Reinsurance Pricing ---")
        print(f"  Baseline (static):         q = {q_baseline:.4f}")
        print(f"  Average Climate:           q = {q_mean:.4f}")
        print(f"  95th Percentile (Adverse): q = {q_p95:.4f}")
        print(f"  99th Percentile (Extreme): q = {q_p99:.4f}")
        print("-------------------------------------------------")
    else:
        # Use baseline only
        q_mean = q_baseline
        q_p95 = q_baseline
        q_p99 = q_baseline
        
        print("\n--- Pareto q Value for Reinsurance Pricing ---")
        print(f"  Baseline (static):         q = {q_baseline:.4f}")
        print("  (No climate-adjusted scenarios available)")
        print("-------------------------------------------------")

    # Step 4: Price reinsurance contracts
    print("\n[4/5] Pricing reinsurance contracts...")
    pricer = ReinsurancePricer(
        threshold=THRESHOLD,
        frequency_mu=len(claims) / N_YEARS,
        layer_deductible=LAYER_DEDUCTIBLE,
        layer_cover=LAYER_COVER
    )

    print(f"\n  Layer structure: ${LAYER_COVER:,.0f} xs ${LAYER_DEDUCTIBLE:,.0f}")

    # Price under baseline vs adverse scenarios
    pricing_results = pricer.price_contract(
        q_baseline=q_baseline,
        q_adverse=q_p95,
        confidence_levels=[0.95, 0.99, 0.995]
    )

    # Step 5: Export results
    print("\n[5/5] Exporting results...")
    pricer.export_pricing_summary(pricing_results, "molapo_reinsurance_pricing.csv")

    # Save processed claims data with climate
    claims.to_csv("molapo_claims_processed.csv", index=False)
    print(f"[OK] Processed claims data exported to: molapo_claims_processed.csv")

    # Save scenario analysis
    scenarios.to_csv("climate_scenarios.csv", index=False)
    print(f"[OK] Scenarios exported to: climate_scenarios.csv")

    print("\n" + "="*70)
    print("                 MOLAPO FLOOD RISK ANALYSIS COMPLETE                 ")
    print("="*70)

    baseline_tvar = pricing_results[pricing_results['Scenario'] == 'Baseline']['TVaR_99'].values[0]
    adverse_tvar = pricing_results[pricing_results['Scenario'] == 'Adverse Climate']['TVaR_99'].values[0]

    # Handle cases where adverse_tvar is zero or negative due to model behavior
    if baseline_tvar > 0:
        risk_multiplier = adverse_tvar / baseline_tvar
        climate_premium = adverse_tvar - baseline_tvar
        risk_increase_percent = (risk_multiplier - 1) * 100
        recommended_loading_percent = (climate_premium / baseline_tvar) * 100
    else:
        # If baseline TVaR is zero, special handling for division by zero
        risk_multiplier = np.nan
        climate_premium = adverse_tvar # Additional premium is just the adverse TVaR
        risk_increase_percent = np.nan
        recommended_loading_percent = np.nan

    print("\n\n--- Key Findings & Reinsurance Recommendations ---")
    print(f"  Baseline TVaR(99%):              ${baseline_tvar:,.2f}")
    print(f"  Adverse Climate TVaR(99%):       ${adverse_tvar:,.2f}")

    if not np.isnan(risk_multiplier):
        print(f"  Climate Risk Multiplier:         {risk_multiplier:.2f}x (Adverse / Baseline)")
        print(f"  Climate Risk Impact on TVaR(99%): {risk_increase_percent:+.1f}%")
        print(f"  Additional Climate Risk Premium: ${climate_premium:,.2f}")
        print(f"  Recommended Loading (above baseline): {recommended_loading_percent:+.1f}%")
    else:
        print("  Could not calculate risk multiplier/premium due to zero baseline TVaR.")
        print(f"  Adverse Climate Premium (if baseline was 0): ${climate_premium:,.2f}")
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    main()
