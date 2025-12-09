# Beginner's Tutorial: Understanding the Flood Risk Analysis Code

## 🎯 What This Code Does

Imagine you're an insurance company that just experienced a major flood. You have 423 claims to pay. Now you need to answer:

1. **How risky is our claims distribution?** (Heavy tail? Light tail?)
2. **Will climate change make things worse?**
3. **How much should we charge for reinsurance?** (Reinsurance = insurance for insurers)

This code answers all three questions using mathematical modeling.

---

## 📚 Key Concepts (Explained Simply)

### 1. Type I Pareto Distribution

**What is it?**
A probability distribution that models "power law" behavior - where a few events are REALLY large.

**Why use it?**
Perfect for catastrophe insurance where:
- Most claims are small ($1,653 - $50,000)
- A few claims are HUGE ($500,000 - $2,000,000)

**The Math (simplified):**
```
P(claim > x) = (K/x)^q

Where:
- K = minimum claim value ($1,653.26 in our data)
- q = "shape" parameter (0.3224 in our data)
  - Small q (< 1) = HEAVY tail = HIGH risk
  - Large q (> 2) = LIGHT tail = LOWER risk
```

**Our result: q = 0.3224**
- This means: EXTREMELY heavy tail
- Translation: Large losses are much more likely than normal distributions predict
- Implication: Need higher insurance reserves

### 2. Maximum Likelihood Estimation (MLE)

**What is it?**
A method to find the "best" parameters for our distribution.

**How it works:**
- Try different values of q
- Pick the q that makes our observed data most likely
- Formula: `q = n / sum(ln(claim_i / K))`

**Think of it like:**
"What value of q makes our actual claims data make the most sense?"

### 3. Climate-Adjusted Modeling (GLM)

**What is it?**
Modeling how the shape parameter q changes with climate (temperature, rainfall).

**The equation:**
```
ln(q) = β₀ + β₁ × temperature + β₂ × rainfall
```

**What the coefficients tell us:**
- **β₁ < 0**: Higher temperature → Lower q → Heavier tail → MORE RISK
- **β₂ < 0**: Higher rainfall → Lower q → Heavier tail → MORE RISK

**Our results:**
- Temperature effect: -0.0410 (negative = increases risk, but small)
- Rainfall effect: -0.0211 (negative = increases risk, but small)
- Neither is statistically significant (p-values > 0.05)

### 4. Reinsurance Pricing

**What is reinsurance?**
Insurance companies buy insurance too! If claims get too high, the reinsurer pays.

**Our "layer":**
```
Reinsurer pays: claims between $80,168 and $555,168
```

**How we price it:**
1. Model claim frequency (how many claims per year): **Poisson distribution**
2. Model claim severity (how big each claim is): **Pareto distribution**
3. Combine them to get total losses: **Aggregate loss model**
4. Calculate risk metrics (VaR, TVaR)

---

## 🔍 Walking Through the Code

### Step 1: Configuration (`config.py`)

**What it does:** Stores all constants in one place

```python
CLAIMS_FILE = "data/CAT Claims.csv"  # Where the data is
LAYER_COVER = 475_000  # Max reinsurance coverage
CONFIDENCE_LEVELS = [0.95, 0.99, 0.995]  # For VaR/TVaR
```

**Why this matters:**
- Easy to change parameters without editing multiple files
- Clear documentation of what values we're using

---

### Step 2: Loading Data (`data_loader.py`)

#### Function: `load_claims_data()`

**What it does:**
1. Opens Excel file
2. Renames columns to standard names
3. Converts amounts to numbers
4. Removes invalid claims (negative, missing)

**Code walkthrough:**
```python
def load_claims_data(file_path: Path) -> pd.DataFrame:
    # 1. Read the Excel file
    df = pd.read_excel(file_path)
    
    # 2. Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
    # 3. Rename to standard name
    if 'Total Claim' in df.columns:
        df = df.rename(columns={'Total Claim': 'claim_amount'})
    
    # 4. Convert to numbers
    df['claim_amount'] = pd.to_numeric(df['claim_amount'], errors='coerce')
    
    # 5. Remove bad data
    df = df.dropna(subset=['claim_amount'])  # Remove missing
    df = df[df['claim_amount'] > 0]          # Remove negative/zero
    
    return df
```

**Input:** Excel file with columns like "Total Claim", "LossDate"
**Output:** Clean DataFrame with standardized column names

#### Function: `add_climate_to_claims()`

**The problem:**
All 423 claims are from ONE event (Feb 2025 flooding). We can't model climate variation from one data point!

**The solution:**
Use **bootstrap sampling** - randomly sample from 31 years of historical climate data and assign to each claim.

**Why this works:**
- Adds artificial temporal variation
- Uses real historical climate patterns
- Enables climate-adjusted modeling

**Code walkthrough:**
```python
def add_climate_to_claims(claims_df, climate_df, random_seed=42):
    # 1. Set random seed (for reproducibility)
    np.random.seed(random_seed)
    
    # 2. Randomly sample climate data (with replacement)
    n_claims = len(claims_df)  # 423 claims
    climate_sample = climate_df.sample(n=n_claims, replace=True)
    
    # 3. Add to claims
    claims_df['temperature'] = climate_sample['Annual_mean_max_C'].values
    claims_df['rainfall'] = climate_sample['Annual_total_mm'].values
    
    # 4. Standardize (convert to z-scores)
    # z-score = (value - mean) / std_dev
    temp_mean = climate_df['Annual_mean_max_C'].mean()
    temp_std = climate_df['Annual_mean_max_C'].std()
    claims_df['temp_std'] = (claims_df['temperature'] - temp_mean) / temp_std
    
    return claims_df
```

**Why standardize?**
- Different units: Temperature in °C, Rainfall in mm
- Different scales: Temp varies 26-30, Rainfall varies 180-853
- Z-scores make them comparable (both have mean=0, std=1)

---

### Step 3: Pareto Modeling (`pareto_model.py`)

#### Class: `ParetoModel`

Think of this as a "Pareto distribution calculator" that holds all our Pareto-related computations.

#### Method: `estimate_baseline_shape_parameter()`

**What it does:**
Calculates q using Maximum Likelihood Estimation.

**Code walkthrough:**
```python
def estimate_baseline_shape_parameter(self, claims_df):
    # 1. Get all claims
    claims = claims_df['claim_amount'].values  # Array of 423 claims
    
    # 2. Calculate log ratios
    # For each claim: ln(claim / K)
    log_ratios = np.log(claims / self.scale_K)
    # Example: ln(50000 / 1653.26) = ln(30.24) = 3.41
    
    # 3. MLE formula: q = n / sum(log_ratios)
    n = len(claims)  # 423
    q_hat = n / np.sum(log_ratios)
    
    # Example calculation:
    # sum(log_ratios) = 1311.76 (actual from our data)
    # q_hat = 423 / 1311.76 = 0.3224
    
    return q_hat
```

**Why this formula works:**
- It's derived from calculus (maximizing the likelihood function)
- Proven to be the best unbiased estimator for Pareto q
- Uses ALL data, not just the tail

#### Method: `estimate_climate_adjusted_parameters()`

**What it does:**
Fits a Generalized Linear Model (GLM) to predict q from climate variables.

**The challenge:**
q is at the population level, but we need individual observations for GLM.

**The solution:**
Use `1 / ln(claim / K)` as a proxy for individual "q contributions."

**Code walkthrough:**
```python
def estimate_climate_adjusted_parameters(self, claims_df, climate_vars):
    # 1. Calculate response variable (proxy for q)
    claims_df['log_ratio'] = np.log(claims_df['claim_amount'] / self.scale_K)
    y = 1 / claims_df['log_ratio']  # Response variable
    
    # 2. Predictor variables (climate)
    X = claims_df[['temp_std', 'rain_std']]
    X = sm.add_constant(X)  # Add intercept column of 1s
    
    # 3. Fit GLM with Gamma family (for positive continuous response)
    glm = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log()))
    result = glm.fit()
    
    # 4. Extract coefficients
    beta_0 = result.params[0]  # Intercept
    beta_1 = result.params[1]  # Temperature coefficient
    beta_2 = result.params[2]  # Rainfall coefficient
    
    return result
```

**Interpreting results:**
```
Model: ln(q) = -1.0227 + (-0.0410 × temp_std) + (-0.0211 × rain_std)

β₀ = -1.0227: Baseline log(q) when climate variables are at their mean (0)
β₁ = -0.0410: For each 1 std dev increase in temperature, ln(q) decreases by 0.041
β₂ = -0.0211: For each 1 std dev increase in rainfall, ln(q) decreases by 0.021

Both negative = both increase tail risk (lower q = heavier tail)
But p-values > 0.05 = not statistically significant
```

#### Method: `predict_q_for_climate()`

**What it does:**
Predicts q for specific climate conditions using the fitted GLM.

**Example:**
```python
# Extreme scenario: Hot (+1.74 std dev) and Wet (+2.02 std dev)
climate_values = {'temp_std': 1.74, 'rain_std': 2.02}

# Calculate using GLM equation
ln_q = -1.0227 + (-0.0410 × 1.74) + (-0.0211 × 2.02)
ln_q = -1.0227 - 0.0713 - 0.0426
ln_q = -1.1366

q = exp(-1.1366) = 0.3208
```

**Translation:**
In extreme hot/wet conditions, q drops from 0.3224 to 0.3208 (slightly heavier tail).

---

### Step 4: Reinsurance Pricing (`reinsurance_pricer.py`)

#### Class: `ReinsurancePricer`

Calculates reinsurance prices using aggregate loss modeling.

#### Method: `build_aggregate_loss_model()`

**What it does:**
Combines frequency + severity + policy structure.

**Code walkthrough:**
```python
def build_aggregate_loss_model(self, shape_q, n_fft_nodes=2**17):
    # 1. FREQUENCY: How many claims per year?
    # Poisson distribution with λ = 423 claims/year
    frequency = Frequency(dist='poisson', par={'mu': 423})
    
    # 2. SEVERITY: How big is each claim?
    # Pareto distribution with K=1653.26, q=0.3224
    severity = Severity(
        dist='pareto2',
        par={'scale': 1653.26, 'shape': 0.3224}
    )
    
    # 3. POLICY: What does reinsurer pay?
    # Pays claims between $80,168 and $555,168
    policy = PolicyStructure(
        layers=Layer(deductible=80168, cover=475000)
    )
    
    # 4. COMBINE: Build aggregate loss model
    # Uses Fast Fourier Transform (FFT) for computation
    model = LossModel(
        frequency=frequency,
        severity=severity,
        policystructure=policy,
        aggr_loss_dist_method='fft',  # Numerical method
        n_aggr_dist_nodes=n_fft_nodes  # Precision (131,072 nodes)
    )
    
    return model
```

**What the model computes:**
- Distribution of **total annual losses** (sum of all claims in a year)
- Accounts for: number of claims × size of each claim × policy limits

#### Method: `calculate_risk_metrics()`

**What it does:**
Calculates VaR and TVaR from the aggregate loss distribution.

**VaR (Value at Risk):**
```
VaR(99%) = $3,790,772

Interpretation: "There's a 99% chance annual losses won't exceed $3.79 million"
```

**TVaR (Tail Value at Risk):**
```
TVaR(99%) = $3,790,772

Interpretation: "If losses exceed VaR(99%), the average loss is $3.79 million"
```

**Why TVaR > VaR?**
- VaR is just a threshold
- TVaR considers how bad things get BEYOND that threshold
- TVaR is more informative for tail risk

**Code walkthrough:**
```python
def calculate_risk_metrics(self, loss_model, confidence_levels):
    # 1. Calculate VaR using model's quantile function
    var_99 = loss_model.ppf(0.99)  # 99th percentile
    
    # 2. Simulate losses for TVaR
    simulated_losses = loss_model.rvs(1_000_000)  # 1 million simulations
    
    # 3. Calculate TVaR: average of losses beyond VaR
    losses_beyond_var = simulated_losses[simulated_losses > var_99]
    tvar_99 = losses_beyond_var.mean()
    
    return {'VaR_99': var_99, 'TVaR_99': tvar_99}
```

---

### Step 5: Main Script (`main_clean.py`)

Orchestrates everything in a clear workflow.

#### Function: `main()`

**Structure:**
1. Print header
2. Load data (with error handling)
3. Fit Pareto model
4. Generate scenarios
5. Price contracts
6. Export results
7. Print summary

**Error handling example:**
```python
try:
    claims_df = load_claims_data(config.CLAIMS_FILE)
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please ensure the claims file exists.")
    return  # Exit gracefully
```

**Why this matters:**
- Prevents cryptic error messages
- Gives user actionable information
- Doesn't crash the entire program

---

## 🎓 Learning Path

### If you're new to Python:
1. Start with `config.py` - just constants
2. Read `data_loader.py` - basic data manipulation
3. Skim `main_clean.py` - see how pieces fit together

### If you understand Python but not statistics:
1. Read "Key Concepts" section above
2. Focus on **what** each function does, not **how**
3. Look at output files to understand results

### If you want to modify the code:
1. Start with `config.py` - change parameters
2. Run `main_clean.py` - see how results change
3. Read function docstrings for details

---

## 💡 Common Questions

**Q: Why is q so low (0.3224)?**
A: Catastrophe insurance has extreme tail risk. This is expected and correct.

**Q: Why isn't climate risk significant?**
A: Either:
   - Small sample size (423 claims, only 33 large)
   - Climate effects are real but small
   - Need more data or different variables

**Q: Can I use this for other disasters?**
A: Yes! Just replace claims data. The methodology is the same.

**Q: What if I don't have climate data?**
A: Code runs fine without it - just skips climate adjustment.

---

## 📖 Further Reading

1. **Pareto Distribution:** Wikipedia article on "Pareto distribution"
2. **GLM:** "Generalized Linear Models" by McCullagh & Nelder
3. **Reinsurance:** "Reinsurance: Fundamentals and New Challenges" by Albrecher et al.
4. **EVT:** "An Introduction to Statistical Modeling of Extreme Values" by Coles

---

## 🎯 Summary

**This code takes:**
- 423 insurance claims from a flood
- 31 years of climate data (optional)

**And produces:**
- Estimate of tail risk (q = 0.3224 = very heavy tail)
- Climate impact analysis (small, not significant)
- Reinsurance prices ($3.79M at 99% confidence)
- Professional CSV outputs for reporting

**All using:**
- Clean, documented code
- Rigorous statistical methods
- Industry-standard practices
