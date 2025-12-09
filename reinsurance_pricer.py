"""
Reinsurance pricing module using aggregate loss modeling.

This module calculates reinsurance contract prices using the GEMAct library,
which models aggregate losses as a combination of frequency and severity distributions.
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from gemact import Frequency, Severity, PolicyStructure, Layer, LossModel


class ReinsurancePricer:
    """
    Calculate reinsurance contract prices using aggregate loss modeling.
    
    The model combines:
    1. Frequency distribution (how many claims per year)
    2. Severity distribution (how large each claim is)
    3. Policy structure (deductible and coverage limits)
    
    Attributes:
        scale_K: Pareto scale parameter (minimum claim value)
        frequency_mu: Expected number of claims per year
        layer_deductible: Deductible amount for reinsurance layer
        layer_cover: Maximum coverage amount for reinsurance layer
    """
    
    def __init__(
        self,
        scale_K: float,
        frequency_mu: float,
        layer_deductible: float,
        layer_cover: float
    ):
        """
        Initialize reinsurance pricer with model parameters.
        
        Args:
            scale_K: Pareto scale parameter (minimum value)
            frequency_mu: Expected number of claims per year (lambda in Poisson)
            layer_deductible: Deductible for the reinsurance layer
            layer_cover: Maximum coverage limit for the layer
        """
        self.scale_K = scale_K
        self.frequency_mu = frequency_mu
        self.layer_deductible = layer_deductible
        self.layer_cover = layer_cover
    
    def build_aggregate_loss_model(
        self,
        shape_q: float,
        n_fft_nodes: int = 2**17
    ) -> LossModel:
        """
        Build aggregate loss model combining frequency and severity distributions.
        
        The aggregate loss distribution represents total losses over a year,
        combining:
        - Frequency: Poisson distribution (number of claims)
        - Severity: Pareto distribution (size of each claim)
        - Policy: Excess of loss layer (deductible and cover)
        
        Args:
            shape_q: Pareto shape parameter
            n_fft_nodes: Number of FFT nodes for numerical computation (higher = more accurate)
            
        Returns:
            Configured LossModel object from GEMAct
        """
        # 1. Frequency distribution: Poisson (models number of claims per year)
        # mu = expected number of claims
        frequency_dist = Frequency(
            dist='poisson',
            par={'mu': self.frequency_mu}
        )
        
        # 2. Severity distribution: Pareto Type 2 (models size of each claim)
        # scale = K (minimum value)
        # shape = q (tail heaviness)
        severity_dist = Severity(
            dist='pareto2',
            par={
                'scale': self.scale_K,
                'shape': shape_q
            }
        )
        
        # 3. Policy structure: Excess of loss layer
        # Reinsurer pays: min(claim - deductible, cover) for claims > deductible
        policy = PolicyStructure(
            layers=Layer(
                deductible=self.layer_deductible,
                cover=self.layer_cover
            )
        )
        
        # 4. Build aggregate loss model
        # This combines frequency + severity + policy using FFT
        loss_model = LossModel(
            frequency=frequency_dist,
            severity=severity_dist,
            policystructure=policy,
            aggr_loss_dist_method='fft',  # Use Fast Fourier Transform
            sev_discr_method='massdispersal',  # Discretization method
            n_aggr_dist_nodes=n_fft_nodes
        )
        
        return loss_model
    
    def calculate_risk_metrics(
        self,
        loss_model: LossModel,
        confidence_levels: List[float],
        n_simulations: int = 1_000_000
    ) -> Dict:
        """
        Calculate risk metrics for a loss model.
        
        Metrics calculated:
        - Pure Premium: Expected loss
        - Mean, Std Dev, Skewness: Distribution moments
        - VaR (Value at Risk): Quantile of loss distribution
        - TVaR (Tail Value at Risk): Expected loss beyond VaR
        
        Args:
            loss_model: Fitted aggregate loss model
            confidence_levels: List of confidence levels for VaR/TVaR (e.g., [0.95, 0.99])
            n_simulations: Number of Monte Carlo simulations for TVaR
            
        Returns:
            Dictionary with all calculated metrics
        """
        # Basic statistics
        pure_premium = loss_model.pure_premium
        if isinstance(pure_premium, list):
            pure_premium = pure_premium[0]
        
        mean_loss = loss_model.mean()
        std_loss = loss_model.std()
        skewness = loss_model.skewness()
        
        # Simulate losses for TVaR calculation
        simulated_losses = loss_model.rvs(n_simulations)
        
        # Calculate VaR and TVaR at each confidence level
        metrics = {
            'Pure_Premium': pure_premium,
            'Mean_Loss': mean_loss,
            'Std_Loss': std_loss,
            'Skewness': skewness
        }
        
        for cl in confidence_levels:
            # VaR: The loss level that will not be exceeded with probability cl
            var = loss_model.ppf(cl)
            metrics[f'VaR_{int(cl*100)}'] = var
            
            # TVaR: Expected loss given that loss exceeds VaR
            losses_beyond_var = simulated_losses[simulated_losses > var]
            if len(losses_beyond_var) > 0:
                tvar = losses_beyond_var.mean()
            else:
                tvar = var  # Fallback if no exceedances
            metrics[f'TVaR_{int(cl*100)}'] = tvar
        
        return metrics
    
    def price_reinsurance_contract(
        self,
        q_baseline: float,
        q_adverse: float,
        confidence_levels: List[float],
        n_fft_nodes: int = 2**17,
        n_simulations: int = 1_000_000
    ) -> pd.DataFrame:
        """
        Price reinsurance contract under baseline and adverse climate scenarios.
        
        This compares the price under:
        1. Baseline scenario (historical q parameter)
        2. Adverse climate scenario (higher risk q parameter)
        
        Args:
            q_baseline: Baseline Pareto shape parameter
            q_adverse: Adverse climate Pareto shape parameter
            confidence_levels: Confidence levels for risk metrics
            n_fft_nodes: FFT nodes for computation
            n_simulations: Monte Carlo simulations for TVaR
            
        Returns:
            DataFrame with pricing results for both scenarios
        """
        print("\n=== Pricing Reinsurance Contract ===")
        print(f"Layer structure: ${self.layer_cover:,.0f} excess of ${self.layer_deductible:,.0f}")
        print(f"(Reinsurer pays claims between ${self.layer_deductible:,.0f} and ${self.layer_deductible + self.layer_cover:,.0f})")
        
        results = []
        
        # Price under baseline scenario
        print(f"\n--- Baseline Scenario (q = {q_baseline:.4f}) ---")
        baseline_model = self.build_aggregate_loss_model(q_baseline, n_fft_nodes)
        baseline_metrics = self.calculate_risk_metrics(baseline_model, confidence_levels, n_simulations)
        
        self._print_metrics(baseline_metrics, confidence_levels)
        
        results.append({
            'Scenario': 'Baseline',
            'q_parameter': q_baseline,
            **baseline_metrics
        })
        
        # Price under adverse climate scenario
        print(f"\n--- Adverse Climate Scenario (q = {q_adverse:.4f}) ---")
        adverse_model = self.build_aggregate_loss_model(q_adverse, n_fft_nodes)
        adverse_metrics = self.calculate_risk_metrics(adverse_model, confidence_levels, n_simulations)
        
        self._print_metrics(adverse_metrics, confidence_levels)
        
        results.append({
            'Scenario': 'Adverse Climate',
            'q_parameter': q_adverse,
            **adverse_metrics
        })
        
        # Calculate climate risk adjustment
        self._print_climate_adjustment(baseline_metrics, adverse_metrics)
        
        return pd.DataFrame(results)
    
    def _print_metrics(self, metrics: Dict, confidence_levels: List[float]) -> None:
        """Helper function to print risk metrics in a readable format."""
        print(f"Pure Premium: ${metrics['Pure_Premium']:,.2f}")
        print(f"Mean Loss: ${metrics['Mean_Loss']:,.2f}")
        print(f"Std Loss: ${metrics['Std_Loss']:,.2f}")
        print(f"Skewness: {metrics['Skewness']:.4f}")
        
        for cl in confidence_levels:
            var_key = f'VaR_{int(cl*100)}'
            tvar_key = f'TVaR_{int(cl*100)}'
            print(f"VaR({cl:.1%}): ${metrics[var_key]:,.2f} | TVaR({cl:.1%}): ${metrics[tvar_key]:,.2f}")
    
    def _print_climate_adjustment(self, baseline: Dict, adverse: Dict) -> None:
        """Helper function to print climate risk adjustment summary."""
        print("\n=== Climate Risk Adjustment ===")
        
        baseline_tvar99 = baseline['TVaR_99']
        adverse_tvar99 = adverse['TVaR_99']
        
        climate_premium = adverse_tvar99 - baseline_tvar99
        
        if baseline_tvar99 > 0:
            risk_multiplier = adverse_tvar99 / baseline_tvar99
            pct_change = (risk_multiplier - 1) * 100
            loading_pct = (climate_premium / baseline_tvar99) * 100
            
            print(f"Baseline TVaR(99%): ${baseline_tvar99:,.2f}")
            print(f"Adverse TVaR(99%): ${adverse_tvar99:,.2f}")
            print(f"Climate Risk Multiplier: {risk_multiplier:.2f}x")
            print(f"Risk Impact: {pct_change:+.1f}%")
            print(f"Additional Premium: ${climate_premium:,.2f}")
            print(f"Recommended Loading: {loading_pct:+.1f}%")
        else:
            print("Unable to calculate risk multiplier (baseline TVaR is zero)")
    
    def export_results(self, results_df: pd.DataFrame, filename: str) -> None:
        """
        Export pricing results to CSV file.
        
        Args:
            results_df: DataFrame with pricing results
            filename: Output CSV filename
        """
        results_df.to_csv(filename, index=False)
        print(f"\n[OK] Pricing results exported to: {filename}")
