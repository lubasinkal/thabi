"""
Type I Pareto distribution parameter estimation module.

This module fits a Type I Pareto distribution to insurance claims data
and provides climate-adjusted parameter estimates.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List
import warnings


class ParetoModel:
    """
    Type I Pareto distribution model for insurance claims.
    
    The Type I Pareto distribution is defined by:
        f(x) = (q * K^q) / x^(q+1)  for x >= K
    
    Where:
        K = scale parameter (minimum value)
        q = shape parameter (controls tail heaviness)
    
    Attributes:
        scale_K: Scale parameter (minimum claim value)
        shape_q_baseline: Baseline shape parameter estimated from data
        glm_model: GLM model for climate-adjusted q estimation (if fitted)
    """
    
    def __init__(self, scale_K: float):
        """
        Initialize Pareto model with scale parameter K.
        
        Args:
            scale_K: Scale parameter (minimum claim value in distribution)
        """
        self.scale_K = scale_K
        self.shape_q_baseline = None
        self.glm_model = None
        self.climate_coefficients = None
    
    def estimate_baseline_shape_parameter(self, claims_df: pd.DataFrame) -> float:
        """
        Estimate baseline Pareto shape parameter q using Maximum Likelihood Estimation.
        
        For Type I Pareto with known K, the MLE for q is:
            q_hat = n / sum(ln(x_i / K))
        
        where all x_i >= K (all claims are used).
        
        Args:
            claims_df: DataFrame with 'claim_amount' column
            
        Returns:
            Estimated shape parameter q
            
        Raises:
            ValueError: If no valid claims found or if claims exist below K
        """
        print("\n=== Estimating Baseline Pareto Shape Parameter (q) ===")
        
        # Get all claims (Type I Pareto uses all data, not just tail)
        claims = claims_df['claim_amount'].values
        
        # Verify all claims are >= K (data quality check)
        min_claim = claims.min()
        if min_claim < self.scale_K:
            print(f"WARNING: Minimum claim (${min_claim:.2f}) < K (${self.scale_K:.2f})")
            print("Some claims violate Type I Pareto assumption (x >= K)")
        
        # Filter to ensure all claims >= K
        valid_claims = claims[claims >= self.scale_K]
        n = len(valid_claims)
        
        if n == 0:
            raise ValueError(f"No claims >= K = ${self.scale_K:.2f}")
        
        # Maximum Likelihood Estimation formula
        # q_hat = n / sum(ln(x_i / K))
        log_ratios = np.log(valid_claims / self.scale_K)
        q_hat = n / np.sum(log_ratios)
        
        # Store the baseline estimate
        self.shape_q_baseline = q_hat
        
        # Print results with interpretation
        print(f"K (scale parameter): ${self.scale_K:,.2f}")
        print(f"Number of claims used: {n}")
        print(f"Estimated q (shape parameter): {q_hat:.4f}")
        if q_hat < 1:
            print("Note: q < 1 indicates infinite mean (extremely heavy tail)")
        
        return q_hat
    
    def estimate_climate_adjusted_parameters(
        self,
        claims_df: pd.DataFrame,
        climate_vars: List[str]
    ) -> Dict:
        """
        Estimate climate-adjusted shape parameter q using Generalized Linear Model (GLM).
        
        We model the shape parameter as a function of climate variables:
            ln(q_t) = beta_0 + beta_1 * climate_var_1 + beta_2 * climate_var_2 + ...
        
        This allows q to vary with climate conditions (e.g., temperature, rainfall).
        
        Args:
            claims_df: DataFrame with 'claim_amount' and climate variable columns
            climate_vars: List of climate variable column names to use as predictors
            
        Returns:
            Dictionary containing:
                - 'intercept': beta_0 coefficient
                - 'coefficients': Dictionary of climate variable coefficients
                - 'model': Fitted GLM model object
                - 'pvalues': P-values for each coefficient
        """
        print("\n=== Climate-Adjusted Pareto Model (GLM) ===")
        
        # Filter claims >= K
        valid_claims_df = claims_df[claims_df['claim_amount'] >= self.scale_K].copy()
        
        # For GLM, we model individual q contributions
        # Using reciprocal of log-ratio as proxy for local q
        valid_claims_df['log_ratio'] = np.log(valid_claims_df['claim_amount'] / self.scale_K)
        
        # Remove any infinite or very small values that could cause numerical issues
        valid_claims_df = valid_claims_df[valid_claims_df['log_ratio'] > 0.01]  # Filter out very small ratios
        
        if len(valid_claims_df) < 30:
            raise ValueError("Not enough valid claims for climate-adjusted model after filtering")
        
        y = 1 / valid_claims_df['log_ratio']  # Response variable (proxy for q)
        
        # Remove any extreme outliers in y (potential numerical issues)
        y_median = np.median(y)
        y_mad = np.median(np.abs(y - y_median))  # Median Absolute Deviation
        if y_mad > 0:
            # Keep values within reasonable range (±5 MAD from median)
            mask = np.abs(y - y_median) < 5 * y_mad
            y = y[mask]
            valid_claims_df = valid_claims_df[mask]
        
        # Predictor variables: climate variables
        X = valid_claims_df[climate_vars]
        X = sm.add_constant(X)  # Add intercept term
        
        # Fit Generalized Linear Model with Gamma family
        # (appropriate for positive continuous response variables)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log()))
            self.glm_model = glm.fit()
        
        # Extract coefficients
        beta_0 = self.glm_model.params.iloc[0]  # Intercept
        betas = self.glm_model.params.iloc[1:]  # Climate variable coefficients
        
        # Store coefficients
        self.climate_coefficients = {
            'intercept': beta_0,
            'climate_effects': betas.to_dict()
        }
        
        # Print model equation
        equation_parts = ['beta_0']
        for i, var in enumerate(climate_vars, 1):
            equation_parts.append(f'beta_{i} * {var}')
        print(f"Model: ln(q_t) = {' + '.join(equation_parts)}")
        
        # Print coefficients with interpretations
        print(f"\nbeta_0 (intercept): {beta_0:.4f} (p-value: {self.glm_model.pvalues.iloc[0]:.4f})")
        
        for i, var in enumerate(climate_vars):
            coef = betas.iloc[i]
            pval = self.glm_model.pvalues.iloc[i + 1]
            print(f"beta_{i+1} ({var}): {coef:.4f} (p-value: {pval:.4f})")
            
            # Interpret coefficient sign
            if coef < 0:
                print(f"  -> Negative: Higher {var} -> Lower q -> Heavier tail")
            else:
                print(f"  -> Positive: Higher {var} -> Higher q -> Lighter tail")
        
        return {
            'intercept': beta_0,
            'coefficients': betas.to_dict(),
            'model': self.glm_model,
            'pvalues': self.glm_model.pvalues.iloc[1:].to_dict()
        }
    
    def predict_q_for_climate(self, climate_values: Dict[str, float]) -> float:
        """
        Predict shape parameter q for specific climate conditions.
        
        Uses the fitted GLM to predict q based on climate variable values.
        
        Args:
            climate_values: Dictionary mapping climate variable names to their values
                           Example: {'temp_std': 1.5, 'rain_std': -0.5}
        
        Returns:
            Predicted q for the given climate conditions
            
        Raises:
            ValueError: If GLM model not yet fitted or if climate variables missing
        """
        if self.climate_coefficients is None:
            raise ValueError("Must fit climate-adjusted model first")
        
        # Calculate ln(q) using model equation
        ln_q = self.climate_coefficients['intercept']
        
        for var_name, coef in self.climate_coefficients['climate_effects'].items():
            if var_name not in climate_values:
                raise ValueError(f"Missing climate variable: {var_name}")
            ln_q += coef * climate_values[var_name]
        
        # Return q = exp(ln(q))
        q_pred = np.exp(ln_q)
        return q_pred
    
    def generate_scenarios(
        self,
        claims_df: pd.DataFrame,
        climate_vars: List[str]
    ) -> pd.DataFrame:
        """
        Generate q estimates for different climate scenarios.
        
        Scenarios include:
        - Baseline (static): Original MLE estimate
        - Average climate: Mean climate conditions
        - 50th percentile: Median climate conditions
        - 95th percentile: Adverse climate conditions
        - 99th percentile: Extreme climate conditions
        
        Args:
            claims_df: DataFrame with claims and climate data
            climate_vars: List of climate variables
            
        Returns:
            DataFrame with scenarios and corresponding q estimates
        """
        print("\n=== Climate Scenario Analysis ===")
        
        scenarios = []
        
        # Scenario 1: Baseline (no climate adjustment)
        scenarios.append({
            'Scenario': 'Baseline (Static)',
            'q_estimate': self.shape_q_baseline
        })
        
        # Calculate climate percentiles for each variable
        climate_stats = {}
        for var in climate_vars:
            values = claims_df[var].values
            climate_stats[var] = {
                'mean': values.mean(),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
        
        # Scenario 2: Average climate
        avg_climate = {var: climate_stats[var]['mean'] for var in climate_vars}
        scenarios.append({
            'Scenario': 'Average Climate',
            'q_estimate': self.predict_q_for_climate(avg_climate),
            **{f'{var}_value': val for var, val in avg_climate.items()}
        })
        
        # Scenario 3: 50th percentile (median)
        p50_climate = {var: climate_stats[var]['p50'] for var in climate_vars}
        scenarios.append({
            'Scenario': '50th Percentile',
            'q_estimate': self.predict_q_for_climate(p50_climate),
            **{f'{var}_value': val for var, val in p50_climate.items()}
        })
        
        # Scenario 4: 95th percentile (adverse)
        p95_climate = {var: climate_stats[var]['p95'] for var in climate_vars}
        scenarios.append({
            'Scenario': '95th Percentile (Adverse)',
            'q_estimate': self.predict_q_for_climate(p95_climate),
            **{f'{var}_value': val for var, val in p95_climate.items()}
        })
        
        # Scenario 5: 99th percentile (extreme)
        p99_climate = {var: climate_stats[var]['p99'] for var in climate_vars}
        scenarios.append({
            'Scenario': '99th Percentile (Extreme)',
            'q_estimate': self.predict_q_for_climate(p99_climate),
            **{f'{var}_value': val for var, val in p99_climate.items()}
        })
        
        result_df = pd.DataFrame(scenarios)
        print(result_df.to_string(index=False))
        
        return result_df
