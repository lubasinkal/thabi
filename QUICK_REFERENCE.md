# Quick Reference Guide

## 📁 File Structure

```
thabi/
├── 🎮 main_clean.py              # START HERE - Run this file
├── ⚙️  config.py                  # All constants and parameters
├── 📥 data_loader.py              # Load and preprocess data
├── 📊 pareto_model.py             # Pareto distribution fitting
├── 💰 reinsurance_pricer.py       # Price reinsurance contracts
├── 📖 README.md                   # Project documentation
├── 🎓 TUTORIAL.md                 # Beginner's tutorial
└── ⚡ QUICK_REFERENCE.md          # This file
```

## 🚀 How to Run

```bash
# Simple way
python main_clean.py

# With virtual environment
.venv\Scripts\python.exe main_clean.py
```

## 📦 Module Cheat Sheet

### `config.py` - Configuration

| Variable | What It Does | Default Value |
|----------|--------------|---------------|
| `CLAIMS_FILE` | Location of claims data | `data/CAT Claims.csv` |
| `CLIMATE_FILE` | Location of climate data | `data/temp_hazard.csv` |
| `LAYER_COVER` | Max reinsurance coverage | 475,000 PULA |
| `CONFIDENCE_LEVELS` | Risk metric confidence levels | [0.95, 0.99, 0.995] |
| `RANDOM_SEED` | For reproducibility | 42 |
| `FFT_NODES` | Computation precision | 131,072 (2^17) |

**When to modify:** When you want different parameters or file locations.

---

### `data_loader.py` - Data Loading

#### Key Functions

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `load_claims_data(file_path)` | Excel file path | Clean claims DataFrame | Loads and validates claims |
| `load_climate_data(file_path)` | CSV file path | Climate DataFrame | Loads temperature/rainfall data |
| `add_climate_to_claims(...)` | Claims + Climate DFs | Enhanced claims DF | Adds climate variables via bootstrap |
| `get_pareto_scale_parameter(...)` | Claims DataFrame | K value (float) | Determines minimum for Pareto |
| `calculate_reinsurance_layer(...)` | Claims DataFrame | Deductible (float) | Sets layer based on percentile |

#### DataFrame Columns

**Claims DataFrame:**
- `claim_amount`: Claim value (PULA)
- `year`: Year of claim
- `temperature`: Temperature (°C) - if climate added
- `rainfall`: Rainfall (mm) - if climate added
- `temp_std`: Standardized temperature (z-score)
- `rain_std`: Standardized rainfall (z-score)

---

### `pareto_model.py` - Pareto Fitting

#### Class: `ParetoModel`

**Attributes:**
- `scale_K`: Minimum claim value (scale parameter)
- `shape_q_baseline`: Baseline shape parameter from MLE
- `glm_model`: Fitted GLM model (if climate-adjusted)
- `climate_coefficients`: β coefficients from GLM

**Methods:**

| Method | What It Does | Returns |
|--------|--------------|---------|
| `estimate_baseline_shape_parameter(claims_df)` | MLE estimation of q | q value (float) |
| `estimate_climate_adjusted_parameters(...)` | Fit GLM: ln(q) ~ climate | Dict with coefficients |
| `predict_q_for_climate(climate_values)` | Predict q for given climate | q value (float) |
| `generate_scenarios(claims_df, climate_vars)` | Create risk scenarios | DataFrame with scenarios |

**Interpretation Guide:**

| q Value | Meaning | Risk Level |
|---------|---------|------------|
| q < 1 | Infinite mean | EXTREME risk ⚠️⚠️⚠️ |
| 1 < q < 2 | Infinite variance | HIGH risk ⚠️⚠️ |
| q > 2 | Finite mean & variance | MODERATE risk ⚠️ |

Your data: **q = 0.3224** → Extremely heavy tail

---

### `reinsurance_pricer.py` - Pricing

#### Class: `ReinsurancePricer`

**Attributes:**
- `scale_K`: Pareto scale parameter
- `frequency_mu`: Expected claims per year
- `layer_deductible`: Reinsurance deductible
- `layer_cover`: Maximum coverage

**Methods:**

| Method | What It Does | Returns |
|--------|--------------|---------|
| `build_aggregate_loss_model(shape_q)` | Create loss model | GEMAct LossModel |
| `calculate_risk_metrics(loss_model, ...)` | Calculate VaR/TVaR | Dict with metrics |
| `price_reinsurance_contract(...)` | Price under scenarios | DataFrame with prices |
| `export_results(results_df, filename)` | Save results to CSV | None (creates file) |

**Risk Metrics:**

| Metric | What It Means | Usage |
|--------|---------------|-------|
| Pure Premium | Expected loss | Base price |
| VaR(99%) | 99th percentile loss | Regulatory capital |
| TVaR(99%) | Expected loss beyond VaR | Risk-based pricing |
| Mean Loss | Average annual loss | Planning |
| Std Loss | Loss volatility | Risk assessment |

---

### `main_clean.py` - Main Script

**Workflow:**

```
1. Load claims data
        ↓
2. Determine K (scale parameter)
        ↓
3. Load climate data (if available)
        ↓
4. Fit Pareto distribution (estimate q)
        ↓
5. Fit climate-adjusted model (GLM)
        ↓
6. Generate risk scenarios
        ↓
7. Price reinsurance contracts
        ↓
8. Export results (CSV files)
        ↓
9. Print summary
```

**Output Files:**

| File | Contains | Use For |
|------|----------|---------|
| `molapo_reinsurance_pricing.csv` | VaR, TVaR, premiums | Pricing decisions |
| `molapo_claims_processed.csv` | Claims + climate | Further analysis |
| `climate_scenarios.csv` | q estimates by scenario | Risk comparison |
| `mean_excess_plot.png` | Diagnostic plot | Model validation |

---

## 🔧 Common Modifications

### Change Reinsurance Layer

```python
# In config.py, change:
LAYER_COVER = 1_000_000  # Increase to 1M coverage
```

Then in `data_loader.py`, modify percentile:
```python
layer_deductible = calculate_reinsurance_layer(claims_df, percentile=90)  # Use 90th
```

### Add More Confidence Levels

```python
# In config.py:
CONFIDENCE_LEVELS = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
```

### Use Different Climate Variables

```python
# In config.py:
CLIMATE_VARS = ['temp_std', 'rain_std', 'humidity_std']
```

Ensure your climate file has corresponding columns.

### Increase Precision

```python
# In config.py:
FFT_NODES = 2**18  # Double the nodes (slower but more accurate)
SIMULATION_SIZE = 5_000_000  # More simulations for TVaR
```

---

## 📊 Understanding Output

### Console Output Sections

1. **Loading Claims Data**
   - Shows number of claims loaded
   - Basic statistics (min, max, mean, median)

2. **Determining Pareto Parameters**
   - K value (minimum claim)
   - Layer deductible (business decision)

3. **Fitting Pareto Distribution**
   - Baseline q estimate
   - Interpretation (infinite mean/variance?)

4. **Climate-Adjusted Model** (if applicable)
   - GLM coefficients (β₀, β₁, β₂)
   - P-values (significance tests)
   - Coefficient interpretations

5. **Risk Scenarios**
   - q estimates for different climate conditions
   - Baseline, average, 95th %, 99th %

6. **Pricing Results**
   - Pure premium, mean loss, std dev
   - VaR and TVaR at different confidence levels
   - Climate risk adjustment

### CSV Output Format

**molapo_reinsurance_pricing.csv:**
```
Scenario,q_parameter,Pure_Premium,Mean_Loss,Std_Loss,VaR_95,VaR_99,TVaR_95,TVaR_99
Baseline,0.3224,38720621.58,1900101.48,1097032.47,3610208.75,3781212.23,3704669.34,3790772.49
Adverse Climate,0.3208,39043223.00,1900101.48,1097032.47,3610208.75,3781212.23,3705156.02,3790734.55
```

**Key columns:**
- `q_parameter`: Shape parameter used
- `TVaR_99`: Main pricing metric (tail value at risk)
- All values in PULA

---

## 🐛 Troubleshooting

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `FileNotFoundError` | Claims file missing | Check `data/CAT Claims.csv` exists |
| `ValueError: No claims >= K` | All claims filtered out | Check claim amounts are positive |
| `ImportError: No module named 'gemact'` | Missing library | Run `uv pip install gemact` |
| `GLM fit failed` | Numerical issues | Code handles this, check data quality |
| `KeyError: 'claim_amount'` | Column renaming failed | Check column names in Excel file |

---

## 🎯 Quick Decision Guide

**I want to...**

- **Run the analysis**: Execute `main_clean.py`
- **Change parameters**: Edit `config.py`
- **Understand the math**: Read `TUTORIAL.md`
- **Modify a function**: Check its docstring first
- **Add new features**: Start with smallest module first
- **Debug an issue**: Check console output for errors
- **Explain to others**: Use `README.md` and this file

---

## 📚 Code Style Conventions

**Naming:**
- `snake_case` for variables and functions
- `PascalCase` for classes
- `UPPER_CASE` for constants

**Documentation:**
- Every function has a docstring
- Docstrings explain: purpose, arguments, returns, examples

**Comments:**
- `#` for inline comments
- `"""..."""` for multi-line documentation
- Comments explain **why**, code shows **how**

**Example:**
```python
def calculate_q(claims, K):
    """
    Calculate Pareto shape parameter using MLE.
    
    Args:
        claims: Array of claim amounts
        K: Scale parameter (minimum value)
        
    Returns:
        Estimated q (shape parameter)
    """
    # MLE formula: q = n / sum(ln(x/K))
    log_ratios = np.log(claims / K)  # Log transform
    q = len(claims) / np.sum(log_ratios)  # MLE estimate
    return q
```

---

## 🎓 Learning Resources

**Within this project:**
1. `README.md` - Overview and setup
2. `TUTORIAL.md` - Detailed walkthrough
3. This file - Quick reference
4. Function docstrings - API documentation

**External resources:**
- **Python basics**: python.org/about/gettingstarted
- **Pandas**: pandas.pydata.org/docs
- **NumPy**: numpy.org/doc
- **Statistics**: Khan Academy (Probability & Statistics)

---

## ✅ Checklist: Before Running

- [ ] Claims data exists in `data/CAT Claims.csv`
- [ ] Climate data exists (optional)
- [ ] Virtual environment activated
- [ ] Required packages installed (`gemact`, `pandas`, `numpy`, etc.)
- [ ] Python version 3.8+ (check: `python --version`)

---

## 🎉 Success Indicators

**You know it worked when:**
1. ✅ No error messages in console
2. ✅ 3 CSV files created in project folder
3. ✅ Mean excess plot PNG generated
4. ✅ Final summary shows key findings
5. ✅ TVaR values are reasonable (millions of PULA)

**Red flags:**
- ⚠️ q < 0 or q > 10 (unrealistic)
- ⚠️ Negative premiums
- ⚠️ All p-values = 0.000 (numerical issues)
- ⚠️ TVaR < VaR (mathematically impossible)

---

**Last Updated:** December 2025
**Version:** 1.0 (Clean rewrite)
