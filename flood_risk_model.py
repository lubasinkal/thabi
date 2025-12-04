"""
Flood Risk Model for Insurance - Climate-Adjusted Pareto Parameter Estimation
Following the methodology from docs/steps.md
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Tuple, Dict, List
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
        self.beta_coeffs = None # To store coefficients for climate variables

    def estimate_baseline_q(self, claims: pd.DataFrame) -> float:
        """
        Step 2: Estimate baseline Pareto parameter q using MLE
        Formula: q_hat = n / sum(ln(x_i / K))\n
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

    def estimate_climate_adjusted_q(self, claims: pd.DataFrame, climate_vars: List[str] = ['SSI']) -> Dict:
        """
        Step 3: Estimate climate-adjusted q using GLM
        Model: ln(q_t) = β₀ + β₁ × ClimateVar1_t + β₂ × ClimateVar2_t + ...

        Args:
            claims: DataFrame with 'loss_amount' and climate variable columns
            climate_vars: List of climate variable column names (default: ['SSI'])

        Returns:
            Dictionary with model results containing beta coefficients and model summary
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
        X = filtered[climate_vars]
        X = sm.add_constant(X) # X now is a DataFrame with 'const' and climate_vars

        # Fit GLM with Gamma family (appropriate for positive continuous response)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log()))
            self.glm_model = glm.fit()

        self.beta_0 = self.glm_model.params.iloc[0]
        self.beta_coeffs = self.glm_model.params.iloc[1:] # Store all climate variable coefficients

        print(f"\n=== Climate-Adjusted q Model (GLM) ===")
        # Dynamically build the model equation string
        model_equation_parts = [f"β₀"] + [f"β{i+1} × {var}_t" for i, var in enumerate(climate_vars)]
        print(f"Model: ln(q_t) = {' + '.join(model_equation_parts)}")

        print(f"β₀ (intercept): {self.beta_0:.4f} (p-value: {self.glm_model.pvalues.iloc[0]:.4f})")
        for i, var in enumerate(climate_vars):
            coeff = self.beta_coeffs.iloc[i]
            p_value = self.glm_model.pvalues.iloc[i+1] # +1 because intercept is at index 0
            print(f"β{i+1} ({var} effect): {coeff:.4f} (p-value: {p_value:.4f})")

            if coeff < 0:
                print(f"✓ Negative β{i+1}: Higher {var} → Lower q → Heavier tail (as expected for severity increasing with variable)")
            else:
                print(f"⚠ Positive β{i+1}: Higher {var} → Higher q → Lighter tail (unexpected for severity increasing with variable)")

        return {
            'beta_0': self.beta_0,
            'beta_coeffs': self.beta_coeffs.to_dict(), # Store as dict for easier access
            'model': self.glm_model,
            'aic': self.glm_model.aic,
            'bic': self.glm_model.bic,
            'pvalues': self.glm_model.pvalues.iloc[1:].to_dict() # Store climate var p-values
        }

    def predict_q(self, climate_values: Dict[str, float]) -> float:
        """
        Step 4: Calculate q for a specific climate scenario using multiple climate variables.

        Args:
            climate_values: Dictionary of climate variable names and their values.
                            Example: {'SSI': 0.5} or {'temp_std': 1.2, 'rain_std': -0.3}

        Returns:
            Predicted q for this climate condition
        """
        if self.beta_0 is None or self.beta_coeffs is None:
            raise ValueError("Must fit GLM model first using estimate_climate_adjusted_q()")

        # ln(q) = β₀ + Σ (βᵢ × climate_valueᵢ)
        ln_q = self.beta_0
        for var_name, coeff in self.beta_coeffs.items():
            if var_name in climate_values:
                ln_q += coeff * climate_values[var_name]
            else:
                # If a climate variable used in training is not provided,
                # assume its mean value (or 0 if standardized) or raise an error.
                # For simplicity, we'll raise an error here.
                raise ValueError(f"Climate variable '{var_name}' is missing from input `climate_values`.")

        q_pred = np.exp(ln_q)

        return q_pred

    def calculate_climate_var_for_target_q(self, target_q: float, climate_var_name: str) -> float:
        """
        Calculates the value of a climate variable required to achieve a target q.
        Model: ln(q_t) = β₀ + β₁ × ClimateVar_t
        So, ClimateVar_t = (ln(target_q) - β₀) / β₁
        """
        if self.beta_0 is None or self.beta_coeffs is None:
            raise ValueError("GLM model must be fitted first.")
        if climate_var_name not in self.beta_coeffs:
            raise ValueError(f"Climate variable '{climate_var_name}' not found in fitted GLM.")

        beta_0 = self.beta_0
        beta_1 = self.beta_coeffs[climate_var_name]

        if beta_1 == 0:
            raise ValueError("Coefficient for climate variable is zero, cannot calculate target value.")

        ln_target_q = np.log(target_q)
        climate_var_value = (ln_target_q - beta_0) / beta_1
        return climate_var_value

    def scenario_analysis(self, claims: pd.DataFrame, climate_vars: List[str] = ['SSI']) -> pd.DataFrame:
        """
        Perform scenario analysis: baseline vs adverse climate conditions

        Args:
            claims: DataFrame with claims and climate data
            climate_vars: List of climate variables to use for scenarios

        Returns:
            DataFrame with scenario results
        """
        if self.baseline_q is None:
            self.estimate_baseline_q(claims)

        if self.beta_0 is None:
            self.estimate_climate_adjusted_q(claims, climate_vars) # Pass climate_vars here

        # Ensure we are working with a single climate variable for this targeted scenario logic
        if len(climate_vars) != 1:
            warnings.warn("Scenario analysis for targeted q values is best suited for a single climate variable. Using the first one.")
            primary_climate_var = climate_vars[0]
        else:
            primary_climate_var = climate_vars[0]

        # Calculate scenario values for the primary climate variable
        climate_values_for_var = claims[primary_climate_var].values

        avg_climate_val = climate_values_for_var.mean()
        p50_climate_val = np.percentile(climate_values_for_var, 50)

        # Define target q values for adverse scenarios to ensure they are lower than baseline
        # Let's aim for 10% and 20% heavier tail than baseline (lower q)
        # Ensure target_q is positive and smaller than baseline to reflect heavier tail
        q_adverse_target = self.baseline_q * 0.9 if self.baseline_q * 0.9 > 0 else self.baseline_q * 0.1
        q_extreme_target = self.baseline_q * 0.8 if self.baseline_q * 0.8 > 0 else self.baseline_q * 0.05

        # Calculate the climate variable values that would achieve these target q values
        adverse_climate_val = self.calculate_climate_var_for_target_q(q_adverse_target, primary_climate_var)
        extreme_climate_val = self.calculate_climate_var_for_target_q(q_extreme_target, primary_climate_var)

        # Prepare dictionaries for predict_q based on scenarios
        avg_climate_dict = {primary_climate_var: avg_climate_val}
        p50_climate_dict = {primary_climate_var: p50_climate_val}
        adverse_climate_dict = {primary_climate_var: adverse_climate_val}
        extreme_climate_dict = {primary_climate_var: extreme_climate_val}

        scenarios = pd.DataFrame({
            'Scenario': ['Baseline (Static)', 'Average Climate', '50th Percentile',
                        '95th Percentile (Adverse)', '99th Percentile (Extreme)'],
            'q_estimate': [
                self.baseline_q,
                self.predict_q(avg_climate_dict),
                self.predict_q(p50_climate_dict),
                self.predict_q(adverse_climate_dict),
                self.predict_q(extreme_climate_dict)
            ]
        })

        # Add climate values to the DataFrame for inspection
        scenarios[f'{primary_climate_var}_Scenario_Value'] = [
            avg_climate_val, # For Baseline (Static), use mean as a placeholder for context
            avg_climate_val,
            p50_climate_val,
            adverse_climate_val,
            extreme_climate_val
        ]

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
```

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
    glm_results = model.estimate_climate_adjusted_q(claims, climate_vars=['SSI'])

    # Step 4: Scenario analysis
    scenarios = model.scenario_analysis(claims, climate_vars=['SSI'])

    print(f"\n=== Model Summary ===")
    print(f"Baseline q: {baseline_q:.4f}")
    print(f"Climate effects (βs): {glm_results['beta_coeffs']}")
    print(f"P-values for climate effects: {glm_results['pvalues']}")
    print(f"Risk multiplier (99th vs baseline): {scenarios.iloc[0]['q_estimate'] / scenarios.iloc[-1]['q_estimate']:.2f}x (Lower q -> Higher Risk)")
