"""
Flood Risk Analysis and Reinsurance Pricing
Main script for analyzing catastrophe insurance claims using Pareto distribution

This script:
1. Loads claims data from CAT Claims.csv
2. Fits Type I Pareto distribution to estimate tail risk
3. Incorporates climate adjustments (if data available)
4. Prices reinsurance contracts under different scenarios
5. Exports results for further analysis

Author: Molapo Insurance Research Team
Date: December 2025
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Import our custom modules
import config
from data_loader import (
    add_climate_to_claims,
    calculate_reinsurance_layer,
    get_pareto_scale_parameter,
    load_claims_data,
    load_climate_data,
)
from pareto_model import ParetoModel
from reinsurance_pricer import ReinsurancePricer


def print_header():
    """Print a nice header for the console output."""
    print("=" * 70)
    print("MOLAPO INSURANCE - FLOOD RISK ANALYSIS")
    print("Type I Pareto Distribution & Climate-Adjusted Reinsurance Pricing")
    print("=" * 70)


def print_step(step_num: int, total_steps: int, description: str):
    """Print a step header."""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 70)


def main():
    """
    Main function that orchestrates the entire analysis workflow.

    Steps:
    1. Load and prepare claims data
    2. Determine Pareto parameters (K and q)
    3. Fit climate-adjusted model (if data available)
    4. Generate risk scenarios
    5. Price reinsurance contracts
    6. Export results
    """
    print_header()

    # ========================================================================
    # STEP 1: Load Claims Data
    # ========================================================================
    print_step(1, 6, "Loading Claims Data")

    try:
        claims_df = load_claims_data(config.CLAIMS_FILE)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure the claims file exists in the data directory.")
        return
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # ========================================================================
    # STEP 2: Determine Pareto Scale Parameter (K)
    # ========================================================================
    print_step(2, 6, "Determining Type I Pareto Parameters")

    # For Type I Pareto, K is the minimum observed claim
    scale_K = get_pareto_scale_parameter(claims_df)

    # Calculate reinsurance layer (business decision, separate from Pareto K)
    layer_deductible = calculate_reinsurance_layer(claims_df, percentile=75)

    # ========================================================================
    # STEP 3: Add Climate Data (if available)``
    # ========================================================================
    print_step(3, 6, "Incorporating Climate Data")

    use_climate = config.CLIMATE_FILE.exists()

    if use_climate:
        try:
            climate_df = load_climate_data(config.CLIMATE_FILE)
            claims_df = add_climate_to_claims(
                claims_df, climate_df, random_seed=config.RANDOM_SEED
            )
            print("Climate variables added successfully.")
        except Exception as e:
            print(f"WARNING: Could not load climate data: {e}")
            print("Proceeding without climate-adjusted modeling.")
            use_climate = False
    else:
        print("Climate data file not found.")
        print("Proceeding with baseline analysis only (no climate adjustment).")

    # ========================================================================
    # STEP 4: Fit Pareto Distribution
    # ========================================================================
    print_step(4, 6, "Fitting Pareto Distribution")

    # Initialize Pareto model with scale parameter K
    pareto = ParetoModel(scale_K=scale_K)

    # Estimate baseline shape parameter q using MLE
    q_baseline = pareto.estimate_baseline_shape_parameter(claims_df)

    # Fit climate-adjusted model (if climate data available)
    if use_climate:
        print("\nFitting climate-adjusted model...")
        pareto.estimate_climate_adjusted_parameters(
            claims_df, climate_vars=config.CLIMATE_VARS
        )

        # Print GLM summary table
        print("\n--- GLM Coefficients (Summary Table) ---")
        print(pareto.glm_model.summary().tables[1].as_text())
        print("-" * 40)

    # ========================================================================
    # STEP 5: Generate Climate Scenarios
    # ========================================================================
    print_step(5, 6, "Generating Risk Scenarios")

    if use_climate:
        # Generate scenarios with climate variation
        scenarios_df = pareto.generate_scenarios(claims_df, config.CLIMATE_VARS)

        # Extract q values for pricing
        q_mean = scenarios_df[scenarios_df["Scenario"] == "Average Climate"][
            "q_estimate"
        ].values[0]
        q_adverse = scenarios_df[
            scenarios_df["Scenario"] == "95th Percentile (Adverse)"
        ]["q_estimate"].values[0]
        q_extreme = scenarios_df[
            scenarios_df["Scenario"] == "99th Percentile (Extreme)"
        ]["q_estimate"].values[0]

        print("\nShape parameter q for pricing scenarios:")
        print(f"  Baseline (static):     q = {q_baseline:.4f}")
        print(f"  Average climate:       q = {q_mean:.4f}")
        print(f"  Adverse (95th %ile):   q = {q_adverse:.4f}")
        print(f"  Extreme (99th %ile):   q = {q_extreme:.4f}")
    else:
        # Use baseline only
        scenarios_df = pd.DataFrame(
            {"Scenario": ["Baseline (Static)"], "q_estimate": [q_baseline]}
        )
        q_adverse = q_baseline
        print(f"Using baseline q = {q_baseline:.4f} (no climate scenarios)")

    # ========================================================================
    # STEP 6: Price Reinsurance Contract
    # ========================================================================
    print_step(6, 6, "Pricing Reinsurance Contract")

    # Calculate expected frequency (claims per year)
    n_years = claims_df["year"].nunique()
    frequency_mu = len(claims_df) / n_years

    print(f"Frequency parameter: {frequency_mu:.2f} claims per year")
    print(f"(Based on {len(claims_df)} claims over {n_years} year(s))")

    # Initialize reinsurance pricer
    pricer = ReinsurancePricer(
        scale_K=scale_K,
        frequency_mu=frequency_mu,
        layer_deductible=layer_deductible,
        layer_cover=config.LAYER_COVER,
    )

    # Price contract under baseline vs adverse scenarios
    pricing_df = pricer.price_reinsurance_contract(
        q_baseline=q_baseline,
        q_adverse=q_adverse,
        confidence_levels=config.CONFIDENCE_LEVELS,
        n_fft_nodes=config.FFT_NODES,
        n_simulations=config.SIMULATION_SIZE,
    )

    # ========================================================================
    # Export Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)

    # Export pricing results
    pricer.export_results(pricing_df, config.OUTPUT_PRICING)

    # Export processed claims data
    claims_df.to_csv(config.OUTPUT_CLAIMS, index=False)
    print(f"[OK] Processed claims exported to: {config.OUTPUT_CLAIMS}")

    # Export scenarios
    scenarios_df.to_csv(config.OUTPUT_SCENARIOS, index=False)
    print(f"[OK] Scenarios exported to: {config.OUTPUT_SCENARIOS}")

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nKey Findings:")
    print(f"  • Total claims analyzed: {len(claims_df)}")
    print(f"  • Pareto scale parameter K: ${scale_K:,.2f}")
    print(f"  • Baseline shape parameter q: {q_baseline:.4f}")
    if q_baseline < 1:
        print(f"    (q < 1: Infinite mean - extremely heavy tail)")
    elif q_baseline < 2:
        print(f"    (q < 2: Infinite variance - heavy tail)")

    if use_climate:
        baseline_tvar = pricing_df[pricing_df["Scenario"] == "Baseline"][
            "TVaR_99"
        ].values[0]
        adverse_tvar = pricing_df[pricing_df["Scenario"] == "Adverse Climate"][
            "TVaR_99"
        ].values[0]

        if baseline_tvar > 0:
            climate_impact = ((adverse_tvar - baseline_tvar) / baseline_tvar) * 100
            print(f"\n  • Climate risk impact on TVaR(99%): {climate_impact:+.1f}%")

    print("\nNext steps:")
    print("  1. Review exported CSV files for detailed results")
    print("  2. Examine mean_excess_plot.png to validate Pareto fit")
    print("  3. Consider sensitivity analysis with different K values")
    print("  4. Review GLM coefficients for climate variable significance")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    """
    Entry point when script is run directly.
    Wraps main() in try-except to handle unexpected errors gracefully.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: An unexpected error occurred:")
        print(f"{type(e).__name__}: {e}")
        print("\nPlease check your data files and try again.")
        import traceback

        traceback.print_exc()
