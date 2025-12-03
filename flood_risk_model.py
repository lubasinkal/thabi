"""
Flood Risk Model for Insurance - Climate-Adjusted Pareto Parameter Estimation
Following the methodology from docs/steps.md
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Tuple, Dict
import warnings


class FloodRiskModel:
    """
    Climate-adjusted flood risk model using Pareto distribution
    with dynamic q parameter based on climate variables
    """
    
    def __init__(self, threshold: float = 25000):
        """
        Args:
            threshold: Threshold K above which Pareto distribution applies
        """
        self.threshold = threshold
        self.baseline_q = None
        self.glm_model = None
        self.beta_0 = None
        self.beta_1 = None
        
    def estimate_baseline_q(self, claims: pd.DataFrame) -> float:
        """
        Step 2: Estimate baseline Pareto parameter q using MLE
        Formula: q_hat = n / sum(ln(x_i / K))
        
        Args:
            claims: DataFrame with 'loss_amount' column
            
        Returns:
            Estimated baseline q parameter
        """
        # Filter claims above threshold
        filtered_losses = claims[claims['loss_amount'] > self.threshold]['loss_amount'].values
        n = len(filtered_losses)
        
        if n == 0:
            raise ValueError(f"No claims above threshold {self.threshold}")
        
        # MLE formula for Pareto q
        log_ratios = np.log(filtered_losses / self.threshold)
        q_hat = n / np.sum(log_ratios)
        
        self.baseline_q = q_hat
        
        print(f"\n=== Baseline Pareto Estimation ===")
        print(f"Threshold (K): ${self.threshold:,.0f}")
        print(f"Number of claims above threshold: {n}")
        print(f"Baseline q estimate: {q_hat:.4f}")
        print(f"Interpretation: Higher q = lighter tail, Lower q = heavier tail")
        
        return q_hat
    
    def estimate_climate_adjusted_q(self, claims: pd.DataFrame, climate_var: str = 'SSI') -> Dict:
        """
        Step 3: Estimate climate-adjusted q using GLM
        Model: ln(q_t) = β₀ + β₁ × SSI_t
        
        Args:
            claims: DataFrame with 'loss_amount' and climate variable columns
            climate_var: Name of climate variable column (default: 'SSI')
            
        Returns:
            Dictionary with model results
        """
        # Filter claims above threshold
        filtered = claims[claims['loss_amount'] > self.threshold].copy()
        
        # For each claim, compute individual q contribution (weights for GLM)
        # Using the relationship: q is related to individual exceedances
        filtered['log_ratio'] = np.log(filtered['loss_amount'] / self.threshold)
        
        # Prepare data for GLM
        # We model: ln(q) = β₀ + β₁ × climate_var
        # Using reciprocal of log_ratio as proxy for local q
        y = 1 / filtered['log_ratio']  # Proxy for q at each claim
        X = filtered[climate_var].values
        X = sm.add_constant(X)
        
        # Fit GLM with Gamma family (appropriate for positive continuous response)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log()))
            self.glm_model = glm.fit()
        
        self.beta_0 = self.glm_model.params.iloc[0]
        self.beta_1 = self.glm_model.params.iloc[1]
        
        print(f"\n=== Climate-Adjusted q Model (GLM) ===")
        print(f"Model: ln(q_t) = β₀ + β₁ × {climate_var}_t")
        print(f"β₀ (intercept): {self.beta_0:.4f}")
        print(f"β₁ (climate effect): {self.beta_1:.4f}")
        print(f"p-value for β₁: {self.glm_model.pvalues.iloc[1]:.4f}")
        
        if self.beta_1 < 0:
            print(f"✓ Negative β₁: Higher {climate_var} → Lower q → Heavier tail (as expected)")
        else:
            print(f"⚠ Positive β₁: Higher {climate_var} → Higher q → Lighter tail (unexpected)")
        
        return {
            'beta_0': self.beta_0,
            'beta_1': self.beta_1,
            'model': self.glm_model,
            'aic': self.glm_model.aic,
            'bic': self.glm_model.bic
        }
    
    def predict_q(self, climate_value: float) -> float:
        """
        Step 4: Calculate q for a specific climate scenario
        
        Args:
            climate_value: Value of climate variable (e.g., SSI)
            
        Returns:
            Predicted q for this climate condition
        """
        if self.beta_0 is None or self.beta_1 is None:
            raise ValueError("Must fit GLM model first using estimate_climate_adjusted_q()")
        
        # ln(q) = β₀ + β₁ × climate_value
        ln_q = self.beta_0 + self.beta_1 * climate_value
        q_pred = np.exp(ln_q)
        
        return q_pred
    
    def scenario_analysis(self, claims: pd.DataFrame, climate_var: str = 'SSI') -> pd.DataFrame:
        """
        Perform scenario analysis: baseline vs adverse climate conditions
        
        Args:
            claims: DataFrame with claims and climate data
            climate_var: Climate variable to use
            
        Returns:
            DataFrame with scenario results
        """
        if self.baseline_q is None:
            self.estimate_baseline_q(claims)
        
        if self.beta_0 is None:
            self.estimate_climate_adjusted_q(claims, climate_var)
        
        # Calculate scenarios
        climate_values = claims[climate_var].values
        q_mean = climate_values.mean()
        q_p50 = np.percentile(climate_values, 50)
        q_p95 = np.percentile(climate_values, 95)
        q_p99 = np.percentile(climate_values, 99)
        
        scenarios = pd.DataFrame({
            'Scenario': ['Baseline (Static)', 'Average Climate', '50th Percentile', 
                        '95th Percentile (Adverse)', '99th Percentile (Extreme)'],
            'Climate_Value': [q_mean, q_mean, q_p50, q_p95, q_p99],
            'q_estimate': [
                self.baseline_q,
                self.predict_q(q_mean),
                self.predict_q(q_p50),
                self.predict_q(q_p95),
                self.predict_q(q_p99)
            ]
        })
        
        # Calculate expected loss multiplier (approximation)
        # Lower q means heavier tail and higher expected losses
        scenarios['Relative_Tail_Risk'] = scenarios['q_estimate'].iloc[0] / scenarios['q_estimate']
        
        print(f"\n=== Scenario Analysis ===")
        print(scenarios.to_string(index=False))
        
        return scenarios
    
    def mean_excess_plot(self, claims: pd.DataFrame, thresholds: np.ndarray = None):
        """
        Generate Mean Excess Plot for threshold selection
        
        Args:
            claims: DataFrame with 'loss_amount' column
            thresholds: Array of thresholds to test
        """
        import matplotlib.pyplot as plt
        
        losses = claims['loss_amount'].values
        
        if thresholds is None:
            # Use quantiles from 10th to 95th percentile
            thresholds = np.percentile(losses, np.linspace(10, 95, 50))
        
        mean_excesses = []
        counts = []
        
        for k in thresholds:
            exceedances = losses[losses > k] - k
            if len(exceedances) > 0:
                mean_excesses.append(exceedances.mean())
                counts.append(len(exceedances))
            else:
                mean_excesses.append(np.nan)
                counts.append(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Mean Excess Plot
        ax1.plot(thresholds, mean_excesses, 'b-', linewidth=2)
        ax1.axvline(self.threshold, color='r', linestyle='--', label=f'Selected K = ${self.threshold:,.0f}')
        ax1.set_xlabel('Threshold (K)')
        ax1.set_ylabel('Mean Excess')
        ax1.set_title('Mean Excess Plot\n(Linear region indicates Pareto tail)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sample size at each threshold
        ax2.plot(thresholds, counts, 'g-', linewidth=2)
        ax2.axvline(self.threshold, color='r', linestyle='--', label=f'Selected K = ${self.threshold:,.0f}')
        ax2.set_xlabel('Threshold (K)')
        ax2.set_ylabel('Number of Exceedances')
        ax2.set_title('Sample Size Above Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mean_excess_plot.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Mean Excess Plot saved to: mean_excess_plot.png")
        
        return fig


if __name__ == "__main__":
    from synthetic_claims import generate_synthetic_claims
    
    # Generate synthetic data
    print("Generating synthetic flood claims data...")
    claims = generate_synthetic_claims(n_years=5, random_seed=42)
    print(f"Generated {len(claims)} claims")
    
    # Initialize model
    model = FloodRiskModel(threshold=25000)
    
    # Step 2: Baseline estimation
    baseline_q = model.estimate_baseline_q(claims)
    
    # Step 3: Climate-adjusted estimation
    glm_results = model.estimate_climate_adjusted_q(claims, climate_var='SSI')
    
    # Step 4: Scenario analysis
    scenarios = model.scenario_analysis(claims, climate_var='SSI')
    
    print(f"\n=== Model Summary ===")
    print(f"Baseline q: {baseline_q:.4f}")
    print(f"Climate effect (β₁): {glm_results['beta_1']:.4f}")
    print(f"Risk multiplier (99th vs baseline): {scenarios.iloc[-1]['Relative_Tail_Risk']:.2f}x")
