"""
Flood Risk Model for Molapo Insurance
Main entry point for climate-adjusted reinsurance pricing
"""
from synthetic_claims import generate_synthetic_claims
from flood_risk_model import FloodRiskModel
from reinsurance_pricing import ReinsurancePricer
import numpy as np


def main():
    """
    Run complete flood risk analysis pipeline:
    1. Generate/load claims data
    2. Estimate baseline and climate-adjusted Pareto parameters
    3. Price reinsurance contracts under different scenarios
    4. Generate reports and visualizations
    """
    print("="*70)
    print("MOLAPO FLOOD RISK MODEL - INSURANCE PRICING")
    print("="*70)
    
    # Configuration
    THRESHOLD = 25000  # Minimum claim size for Pareto tail
    N_YEARS = 5
    LAYER_DEDUCTIBLE = 25000
    LAYER_COVER = 475000
    
    # Step 1: Generate synthetic claims data
    print("\n[1/5] Generating synthetic claims data...")
    claims = generate_synthetic_claims(
        n_years=N_YEARS,
        base_frequency=50,
        threshold=THRESHOLD,
        random_seed=42
    )
    print(f"✓ Generated {len(claims)} claims over {N_YEARS} years")
    print(f"  Total losses: ${claims['loss_amount'].sum():,.2f}")
    print(f"  Average claim: ${claims['loss_amount'].mean():,.2f}")
    
    # Step 2: Estimate Pareto parameters
    print("\n[2/5] Estimating Pareto parameters...")
    model = FloodRiskModel(threshold=THRESHOLD)
    
    # Baseline q using MLE
    q_baseline = model.estimate_baseline_q(claims)
    
    # Climate-adjusted q using GLM
    model.estimate_climate_adjusted_q(claims, climate_var='SSI')
    
    # Scenario analysis
    scenarios = model.scenario_analysis(claims, climate_var='SSI')
    
    # Step 3: Get q values for pricing
    print("\n[3/5] Calculating scenario q parameters...")
    ssi_mean = claims['SSI'].mean()
    ssi_p95 = np.percentile(claims['SSI'].values, 95)
    ssi_p99 = np.percentile(claims['SSI'].values, 99)
    
    q_mean = model.predict_q(ssi_mean)
    q_p95 = model.predict_q(ssi_p95)
    q_p99 = model.predict_q(ssi_p99)
    
    print(f"  Baseline (static): q = {q_baseline:.4f}")
    print(f"  Mean climate: q = {q_mean:.4f}")
    print(f"  P95 adverse: q = {q_p95:.4f}")
    print(f"  P99 extreme: q = {q_p99:.4f}")
    
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
    
    # Save claims data
    claims.to_csv("synthetic_claims_data.csv", index=False)
    print(f"✓ Claims data exported to: synthetic_claims_data.csv")
    
    # Save scenario analysis
    scenarios.to_csv("climate_scenarios.csv", index=False)
    print(f"✓ Scenarios exported to: climate_scenarios.csv")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    baseline_tvar = pricing_results[pricing_results['Scenario'] == 'Baseline']['TVaR_99'].values[0]
    adverse_tvar = pricing_results[pricing_results['Scenario'] == 'Adverse Climate']['TVaR_99'].values[0]
    risk_multiplier = adverse_tvar / baseline_tvar
    climate_premium = adverse_tvar - baseline_tvar
    
    print(f"• Climate risk increases TVaR(99%) by {(risk_multiplier-1)*100:.1f}%")
    print(f"• Additional climate risk premium: ${climate_premium:,.2f}")
    print(f"• Recommended loading: {climate_premium/baseline_tvar*100:.1f}% above baseline")


if __name__ == "__main__":
    main()
