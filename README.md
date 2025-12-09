# Flood Risk Analysis

Type I Pareto distribution modeling for catastrophe insurance claims with climate adjustments.

## Structure

```
thabi/
├── main.py                   # Main analysis script
├── config.py                 # Configuration parameters
├── data_loader.py            # Data loading and preprocessing
├── pareto_model.py           # Pareto parameter estimation
├── reinsurance_pricer.py     # Contract pricing
├── data/
│   ├── CAT Claims.csv        # Claims data
│   └── temp_hazard.csv       # Climate data
└── docs/
    └── threshold_selection_methodology.md
```

## Usage

```bash
python main.py
```

## Methodology

**Type I Pareto Distribution:**
```
f(x) = (q × K^q) / x^(q+1)  for x ≥ K
```

Where:
- K = scale parameter (minimum value) = min(claims)
- q = shape parameter (MLE estimation)

**Climate-Adjusted Model:**
```
ln(q) = β₀ + β₁ × temperature + β₂ × rainfall
```

## Output Files

- `molapo_reinsurance_pricing.csv` - VaR, TVaR, and premium calculations
- `molapo_claims_processed.csv` - Processed claims with climate data
- `climate_scenarios.csv` - Risk scenarios (baseline, adverse, extreme)
- `mean_excess_plot.png` - Diagnostic plot for threshold validation

## Requirements

```
pandas
numpy
statsmodels
gemact
matplotlib
openpyxl
```

Install: `uv pip install pandas numpy statsmodels gemact matplotlib openpyxl`

## Key Results

Based on 423 claims from Feb 2025 flooding:
- K (scale) = $1,653.26 (minimum observed claim)
- q (shape) = 0.3224 (extremely heavy tail, q < 1 → infinite mean)
- Climate effects: Not statistically significant (p > 0.05)

## References

- Kleiber & Kotz (2003). Statistical Size Distributions in Economics and Actuarial Sciences
- Arnold (2015). Pareto Distributions (2nd Ed.)
- Klugman et al. (2012). Loss Models: From Data to Decisions (4th Ed.)
