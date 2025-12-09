# Flood Risk Analysis & Reinsurance Pricing

A clean, beginner-friendly implementation of Type I Pareto distribution fitting for catastrophe insurance claims with climate-adjusted risk modeling.

## 📁 Project Structure

```
thabi/
├── config.py                 # Configuration and constants
├── data_loader.py            # Data loading and preprocessing
├── pareto_model.py           # Pareto distribution parameter estimation
├── reinsurance_pricer.py     # Reinsurance contract pricing
├── main_clean.py             # Main analysis script (USE THIS!)
├── data/
│   ├── CAT Claims.csv        # Claims data (Excel format)
│   └── temp_hazard.csv       # Climate data (optional)
├── docs/
│   └── threshold_selection_methodology.md  # Research documentation
└── outputs/
    ├── molapo_reinsurance_pricing.csv     # Pricing results
    ├── molapo_claims_processed.csv         # Processed claims
    ├── climate_scenarios.csv               # Risk scenarios
    └── mean_excess_plot.png                # Diagnostic plot
```

## 🚀 Quick Start

### Running the Analysis

```bash
# Using virtual environment
.venv\Scripts\python.exe main_clean.py

# Or if python is in PATH
python main_clean.py
```

The script will:
1. Load claims data from `data/CAT Claims.csv`
2. Fit Type I Pareto distribution
3. Add climate adjustments (if data available)
4. Generate risk scenarios
5. Price reinsurance contracts
6. Export results to CSV files

### Expected Output

```
======================================================================
MOLAPO INSURANCE - FLOOD RISK ANALYSIS
Type I Pareto Distribution & Climate-Adjusted Reinsurance Pricing
======================================================================

[Step 1/6] Loading Claims Data
----------------------------------------------------------------------
Loading claims data from: data\CAT Claims.csv
   Loaded 423 valid claims
   Claim range: $1,653.26 to $2,002,745.94
   Mean claim: $100,450.76
   Median claim: $32,403.40

[Step 2/6] Determining Type I Pareto Parameters
----------------------------------------------------------------------
...
```

## 📖 Module Documentation

### 1. `config.py` - Configuration

Contains all project constants and parameters in one place:
- File paths
- Model parameters (FFT nodes, simulation size)
- Climate variables
- Confidence levels

**Example:**
```python
LAYER_COVER = 475_000  # Maximum coverage in PULA
CONFIDENCE_LEVELS = [0.95, 0.99, 0.995]
RANDOM_SEED = 42  # For reproducibility
```

### 2. `data_loader.py` - Data Loading

Functions for loading and preprocessing data:

#### `load_claims_data(file_path)`
Loads claims from Excel file, handles column mapping, cleans data.

**Returns:** DataFrame with standardized columns:
- `claim_amount`: Claim value in PULA
- `year`: Year of claim
- `LossDate`: Date of loss (if available)

#### `load_climate_data(file_path)`
Loads historical climate data (temperature, rainfall).

**Returns:** DataFrame with:
- `Year`: Year
- `Annual_mean_max_C`: Annual mean maximum temperature
- `Annual_total_mm`: Annual total rainfall

#### `add_climate_to_claims(claims_df, climate_df)`
Adds climate variables to claims using bootstrap sampling.

**Why bootstrap?** All claims are from one event (Feb 2025), so we sample historical climate data to add temporal variation for modeling.

**Returns:** Claims DataFrame with added columns:
- `temperature`, `rainfall`: Raw climate values
- `temp_std`, `rain_std`: Standardized (z-score) values

#### `get_pareto_scale_parameter(claims_df)`
Determines K (scale parameter) for Type I Pareto.

**Method:** K = minimum observed claim
**Justification:** Type I Pareto definition requires K ≤ min(data)

### 3. `pareto_model.py` - Pareto Distribution

Class `ParetoModel`: Fits Type I Pareto distribution to claims.

#### Type I Pareto Distribution
```
f(x) = (q × K^q) / x^(q+1)  for x ≥ K

Where:
  K = scale parameter (minimum value)
  q = shape parameter (tail heaviness)
```

#### Key Methods:

**`estimate_baseline_shape_parameter(claims_df)`**
- Estimates q using Maximum Likelihood Estimation (MLE)
- Formula: `q_hat = n / sum(ln(x_i / K))`
- Uses ALL claims (not just tail)

**`estimate_climate_adjusted_parameters(claims_df, climate_vars)`**
- Fits GLM to model: `ln(q) = β₀ + β₁×temp + β₂×rain`
- Allows q to vary with climate conditions
- Returns coefficients and p-values

**`predict_q_for_climate(climate_values)`**
- Predicts q for specific climate conditions
- Example: `{'temp_std': 1.5, 'rain_std': -0.5}`

**`generate_scenarios(claims_df, climate_vars)`**
- Creates risk scenarios:
  - Baseline (static q)
  - Average climate
  - 95th percentile (adverse)
  - 99th percentile (extreme)

### 4. `reinsurance_pricer.py` - Pricing

Class `ReinsurancePricer`: Calculates reinsurance contract prices.

#### Aggregate Loss Model

Combines:
1. **Frequency**: Poisson distribution (# of claims per year)
2. **Severity**: Pareto distribution (size of each claim)
3. **Policy**: Excess of loss layer (deductible + cover)

#### Key Methods:

**`build_aggregate_loss_model(shape_q, n_fft_nodes)`**
- Builds loss model using GEMAct library
- Uses Fast Fourier Transform (FFT) for computation
- Returns configured LossModel

**`calculate_risk_metrics(loss_model, confidence_levels)`**
- Calculates:
  - Pure Premium (expected loss)
  - Mean, Std Dev, Skewness
  - VaR (Value at Risk): Loss quantile
  - TVaR (Tail Value at Risk): Expected loss beyond VaR

**`price_reinsurance_contract(q_baseline, q_adverse)`**
- Prices contract under two scenarios
- Compares baseline vs. adverse climate
- Returns DataFrame with all metrics

### 5. `main_clean.py` - Main Script

Orchestrates the entire analysis workflow:

1. **Load data**: Claims and climate (if available)
2. **Determine K**: Scale parameter from minimum claim
3. **Fit Pareto**: Estimate q using MLE
4. **Climate adjustment**: Fit GLM (if data available)
5. **Generate scenarios**: Multiple risk levels
6. **Price contract**: Calculate premiums and risk metrics
7. **Export results**: Save to CSV files

## 🔬 Understanding the Results

### Pareto Shape Parameter (q)

- **q < 1**: Infinite mean (extremely heavy tail) ← YOUR DATA
- **q < 2**: Infinite variance (very heavy tail)
- **q > 2**: Finite mean and variance

**Your result: q = 0.3224**
- This indicates EXTREME tail risk
- Appropriate for catastrophe insurance
- Large losses are more probable than light-tailed distributions

### Risk Metrics

**VaR (Value at Risk)**: 
- VaR(99%) = $X means: "There's a 99% chance losses won't exceed $X"
- Used for regulatory capital requirements

**TVaR (Tail Value at Risk)**:
- TVaR(99%) = $Y means: "If losses exceed VaR(99%), expected loss is $Y"
- More informative than VaR (considers tail severity)
- Used for pricing and risk management

### Climate Risk Adjustment

Compares baseline vs. adverse climate scenarios:
```
Risk Multiplier = TVaR(adverse) / TVaR(baseline)
```
- Multiplier > 1: Climate increases risk
- Multiplier < 1: Climate decreases risk (unlikely)

## 📊 Output Files

### `molapo_reinsurance_pricing.csv`
Contract pricing under different scenarios.

**Columns:**
- `Scenario`: Baseline or Adverse Climate
- `q_parameter`: Shape parameter used
- `Pure_Premium`, `Mean_Loss`, `Std_Loss`: Basic statistics
- `VaR_95`, `VaR_99`, `VaR_99.5`: Value at Risk at different levels
- `TVaR_95`, `TVaR_99`, `TVaR_99.5`: Tail Value at Risk

### `molapo_claims_processed.csv`
Processed claims with climate variables added.

### `climate_scenarios.csv`
Q estimates for different climate conditions.

### `mean_excess_plot.png`
Diagnostic plot showing:
- Mean excess vs. threshold
- Sample size at each threshold
- Helps validate Pareto distribution fit

## 🎓 For Research Papers

### How to Cite the Methodology

> "We fit a Type I Pareto distribution to catastrophe insurance claims using maximum likelihood estimation. The scale parameter K was set to the minimum observed claim value ($1,653.26), consistent with the Type I Pareto definition (Kleiber & Kotz, 2003; Arnold, 2015). All 423 claims were used for parameter estimation, yielding a shape parameter q = 0.3224. The estimated q < 1 indicates infinite mean, characteristic of extremely heavy-tailed distributions in catastrophe insurance (Klugman et al., 2012)."

### Key References

1. **Kleiber, C., & Kotz, S. (2003).** *Statistical Size Distributions in Economics and Actuarial Sciences*. Wiley.
2. **Arnold, B. C. (2015).** *Pareto Distributions* (2nd Ed.). CRC Press.
3. **Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2012).** *Loss Models: From Data to Decisions* (4th Ed.). Wiley.

## 🔧 Customization

### Changing Parameters

Edit `config.py`:
```python
# Increase FFT precision
FFT_NODES = 2**18  # Default: 2**17

# More simulations for TVaR
SIMULATION_SIZE = 5_000_000  # Default: 1,000,000

# Different confidence levels
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99, 0.999]
```

### Adding Climate Variables

In `config.py`:
```python
CLIMATE_VARS = ['temp_std', 'rain_std', 'humidity_std']
```

Then ensure your climate data file has the corresponding columns.

## ❓ Troubleshooting

### "No module named 'gemact'"
```bash
uv pip install gemact
```

### "File not found: data/CAT Claims.csv"
Ensure your claims file is in the `data/` directory with correct name.

### "No claims >= K"
Check your data - all claims should be positive values.

### Climate data not loading
This is optional - analysis will run without it (baseline only).

## 📞 Support

For questions about:
- **Mathematics**: See `docs/threshold_selection_methodology.md`
- **Code**: Read function docstrings (triple-quoted strings in code)
- **Data format**: Check `data_loader.py` for expected columns

## 📝 License

Research use only.
