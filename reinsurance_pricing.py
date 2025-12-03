"""
Reinsurance Pricing Module using GEMAct
Integrates climate-adjusted q parameters with aggregate loss modeling
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from gemact import Frequency, Severity, PolicyStructure, Layer, LossModel


class ReinsurancePricer:
    """
    Price reinsurance contracts using climate-adjusted aggregate loss models
    """
    
    def __init__(
        self,
        threshold: float = 25000,
        frequency_mu: float = 50,
        layer_deductible: float = 25000,
        layer_cover: float = 475000
    ):
        """
        Args:
            threshold: Claims threshold (K)
            frequency_mu: Expected number of claims per year
            layer_deductible: Deductible for reinsurance layer
            layer_cover: Coverage limit for reinsurance layer
        """
        self.threshold = threshold
        self.frequency_mu = frequency_mu
        self.layer_deductible = layer_deductible
        self.layer_cover = layer_cover
        
    def build_loss_model(
        self,
        q: float,
        freq_mu: float = None,
        n_nodes: int = 2**15
    ) -> LossModel:
        """
        Build GEMAct aggregate loss model with specified parameters
        
        Args:
            q: Pareto shape parameter
            freq_mu: Frequency parameter (defaults to self.frequency_mu)
            n_nodes: Number of FFT nodes for aggregate distribution
            
        Returns:
            Configured LossModel
        """
        if freq_mu is None:
            freq_mu = self.frequency_mu
        
        # 1. Frequency: Poisson distribution
        frequency = Frequency(dist='poisson', par={'mu': freq_mu})
        
        # 2. Severity: Pareto2 distribution
        severity = Severity(
            dist='pareto2',
            par={
                'scale': self.threshold,
                'shape': q
            }
        )
        
        # 3. Policy Structure: Excess of loss layer
        policy = PolicyStructure(
            layers=Layer(
                deductible=self.layer_deductible,
                cover=self.layer_cover
            )
        )
        
        # 4. Aggregate Loss Model
        lm = LossModel(
            frequency=frequency,
            severity=severity,
            policystructure=policy,
            aggr_loss_dist_method='fft',
            sev_discr_method='massdispersal',
            n_aggr_dist_nodes=n_nodes
        )
        
        return lm
    
    def price_contract(
        self,
        q_baseline: float,
        q_adverse: float,
        confidence_levels: list = [0.95, 0.99, 0.995]
    ) -> pd.DataFrame:
        """
        Price reinsurance contract under different climate scenarios
        
        Args:
            q_baseline: Baseline Pareto parameter
            q_adverse: Adverse climate Pareto parameter
            confidence_levels: List of VaR/TVaR confidence levels
            
        Returns:
            DataFrame with pricing metrics for each scenario
        """
        scenarios = {
            'Baseline': q_baseline,
            'Adverse Climate': q_adverse
        }
        
        results = []
        
        for scenario_name, q_value in scenarios.items():
            print(f"\n{'='*50}")
            print(f"Computing: {scenario_name} (q = {q_value:.4f})")
            print(f"{'='*50}")
            
            # Build loss model
            lm = self.build_loss_model(q=q_value)
            
            # Calculate key metrics
            # pure_premium is a list with one element per layer
            pure_premium = lm.pure_premium[0] if isinstance(lm.pure_premium, list) else lm.pure_premium
            mean_loss = lm.mean()
            std_loss = lm.std()
            skewness = lm.skewness()
            
            # Calculate VaR and TVaR at different confidence levels
            # TVaR is computed by simulating from the distribution
            n_sim = 100000
            simulated_losses = lm.rvs(n_sim)
            
            var_values = {}
            tvar_values = {}
            
            for cl in confidence_levels:
                # VaR: quantile of the distribution
                var = lm.ppf(cl)
                
                # TVaR: expected value of losses exceeding VaR
                losses_above_var = simulated_losses[simulated_losses > var]
                if len(losses_above_var) > 0:
                    tvar = losses_above_var.mean()
                else:
                    tvar = var  # fallback if no exceedances
                
                var_values[f'VaR_{int(cl*100)}'] = var
                tvar_values[f'TVaR_{int(cl*100)}'] = tvar
            
            result = {
                'Scenario': scenario_name,
                'q_parameter': q_value,
                'Pure_Premium': pure_premium,
                'Mean_Loss': mean_loss,
                'Std_Loss': std_loss,
                'Skewness': skewness,
                **var_values,
                **tvar_values
            }
            
            results.append(result)
            
            print(f"Pure Premium: ${pure_premium:,.2f}")
            print(f"Mean Loss: ${mean_loss:,.2f}")
            print(f"Std Loss: ${std_loss:,.2f}")
            print(f"Skewness: {skewness:.4f}")
            for cl in confidence_levels:
                var = var_values[f'VaR_{int(cl*100)}']
                tvar = tvar_values[f'TVaR_{int(cl*100)}']
                print(f"VaR({cl:.1%}): ${var:,.2f} | TVaR({cl:.1%}): ${tvar:,.2f}")
        
        df = pd.DataFrame(results)
        
        # Calculate risk adjustments
        baseline_tvar_99 = df[df['Scenario'] == 'Baseline']['TVaR_99'].values[0]
        adverse_tvar_99 = df[df['Scenario'] == 'Adverse Climate']['TVaR_99'].values[0]
        climate_risk_premium = adverse_tvar_99 - baseline_tvar_99
        
        print(f"\n{'='*50}")
        print(f"CLIMATE RISK ADJUSTMENT")
        print(f"{'='*50}")
        print(f"Baseline TVaR(99%): ${baseline_tvar_99:,.2f}")
        print(f"Adverse TVaR(99%): ${adverse_tvar_99:,.2f}")
        print(f"Climate Risk Premium: ${climate_risk_premium:,.2f}")
        print(f"Risk Multiplier: {adverse_tvar_99/baseline_tvar_99:.2f}x")
        
        return df
    
    def sensitivity_analysis(
        self,
        q_baseline: float,
        q_range: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on q parameter
        
        Args:
            q_baseline: Baseline q value
            q_range: Array of q values to test (defaults to ±30% of baseline)
            
        Returns:
            DataFrame with sensitivity results
        """
        if q_range is None:
            q_range = np.linspace(q_baseline * 0.7, q_baseline * 1.3, 13)
        
        results = []
        
        for q in q_range:
            lm = self.build_loss_model(q=q)
            
            # Calculate TVaR by simulation
            simulated_losses = lm.rvs(50000)
            var_99 = lm.ppf(0.99)
            losses_above_var = simulated_losses[simulated_losses > var_99]
            tvar_99 = losses_above_var.mean() if len(losses_above_var) > 0 else var_99
            
            results.append({
                'q': q,
                'q_pct_change': (q - q_baseline) / q_baseline * 100,
                'Mean_Loss': lm.mean(),
                'VaR_95': lm.ppf(0.95),
                'VaR_99': var_99,
                'TVaR_99': tvar_99
            })
        
        df = pd.DataFrame(results)
        
        print(f"\n=== Sensitivity Analysis: q Parameter ===")
        print(df.to_string(index=False))
        
        return df
    
    def export_pricing_summary(
        self,
        pricing_df: pd.DataFrame,
        filename: str = "reinsurance_pricing_summary.csv"
    ):
        """Export pricing results to CSV"""
        pricing_df.to_csv(filename, index=False)
        print(f"\n✓ Pricing summary exported to: {filename}")


if __name__ == "__main__":
    from synthetic_claims import generate_synthetic_claims
    from flood_risk_model import FloodRiskModel
    
    print("="*70)
    print("FLOOD RISK MODEL - REINSURANCE PRICING")
    print("="*70)
    
    # Step 1: Generate synthetic claims
    print("\n[1/4] Generating synthetic claims data...")
    claims = generate_synthetic_claims(n_years=5, random_seed=42)
    
    # Step 2: Estimate q parameters
    print("\n[2/4] Estimating Pareto parameters...")
    model = FloodRiskModel(threshold=25000)
    q_baseline = model.estimate_baseline_q(claims)
    model.estimate_climate_adjusted_q(claims, climate_var='SSI')
    
    # Get adverse scenario q
    ssi_p95 = np.percentile(claims['SSI'].values, 95)
    q_adverse = model.predict_q(ssi_p95)
    
    print(f"\nq_baseline: {q_baseline:.4f}")
    print(f"q_adverse (95th percentile SSI): {q_adverse:.4f}")
    
    # Step 3: Price reinsurance contract
    print("\n[3/4] Pricing reinsurance contract...")
    pricer = ReinsurancePricer(
        threshold=25000,
        frequency_mu=len(claims) / 5,  # Claims per year
        layer_deductible=25000,
        layer_cover=475000
    )
    
    pricing_results = pricer.price_contract(
        q_baseline=q_baseline,
        q_adverse=q_adverse,
        confidence_levels=[0.95, 0.99, 0.995]
    )
    
    # Step 4: Export results
    print("\n[4/4] Exporting results...")
    pricer.export_pricing_summary(pricing_results)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
