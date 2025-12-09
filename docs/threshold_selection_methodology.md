# Type I Pareto Distribution: Parameter K Determination

## Research Context
This document explains the methodology for determining K in the Type I Pareto distribution for flood insurance claims modeling.

## Type I Pareto Distribution Definition

The Type I Pareto distribution has probability density function:

f(x) = (q * K^q) / x^(q+1)  for x >= K

Where:
- **K** = scale parameter (minimum possible value)
- **q** = shape parameter (controls tail heaviness)

## Parameter K: The Scale Parameter

**K is NOT a threshold we choose** - it is the **minimum value** the distribution can take.

For Type I Pareto:
- K represents the lower bound of the distribution
- All observations must satisfy x >= K
- K is a fixed parameter, not a selection criterion

## Methodology for Determining K

### Approach 1: Minimum Observed Value (SELECTED)

**Description**: Set K equal to the minimum observed claim in the dataset.

**Justification**:
- **Mathematical Definition**: Type I Pareto requires K ≤ min(data)
- **Conservative Approach**: Uses the smallest observation directly
- **Maximum Likelihood**: When K is set to min(data), the MLE for q is:
  
  q_hat = n / Σ ln(x_i / K)
  
  where all x_i >= K

**Advantages**:
- Mathematically rigorous and defensible
- Uses all available data
- No arbitrary threshold selection required
- Consistent with Type I Pareto definition

**Implementation**:
- K = min(claims) = $1,653.26
- All 423 claims used for parameter estimation
- Estimated q = 0.3224

**Interpretation of q = 0.3224**:
- q < 1: Distribution has **infinite mean** (very heavy tail)
- q < 2: Distribution has **infinite variance**
- This indicates extreme tail risk - appropriate for catastrophe insurance

### Approach 2: Conservative K (Alternative)

**Description**: Set K slightly below the minimum observed value (e.g., 95% of minimum).

**Justification**:
- Allows for possibility of smaller claims that weren't observed
- More conservative risk estimation
- Common in insurance when data may not capture full range

**Implementation**: K = 0.95 × min(claims) = $1,570.60

**When to Use**:
- When concerned about left-censoring of data
- When minimum claim may be artifactually high
- For sensitivity analysis

## Distinction: Type I Pareto vs. Peaks Over Threshold (POT)

### Type I Pareto (This Analysis)
- K = minimum value of distribution
- Fits entire claim distribution
- All data used: x >= K for all x
- MLE for q when K is known

### Peaks Over Threshold (Different Method)
- Select a high threshold u
- Fit Generalized Pareto Distribution (GPD) to exceedances: x - u for x > u
- Only models the tail
- Referenced in: Embrechts et al. (1997), McNeil et al. (2015)

**Important**: POT is a different modeling approach - don't confuse it with Type I Pareto parameter estimation.

## Mean Excess Plot

While traditionally used for POT threshold selection, the mean excess plot can still provide diagnostic information about tail behavior.

**Interpretation**:
- Approximately linear region suggests Pareto-type tail
- Helps validate that Pareto is appropriate for the data

## References

1. Kleiber, C., & Kotz, S. (2003). *Statistical Size Distributions in Economics and Actuarial Sciences*. Wiley. (Chapter on Pareto distributions)

2. Arnold, B. C. (2015). *Pareto Distributions* (2nd Edition). CRC Press.

3. Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994). *Continuous Univariate Distributions, Volume 1* (2nd Edition). Wiley. (Chapter 20: Pareto distributions)

4. Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2012). *Loss Models: From Data to Decisions* (4th Edition). Wiley. (Section on Pareto distribution)

## Conclusion

For Type I Pareto distribution fitting:
- K is the **scale parameter** (minimum value), not a threshold we select
- Set K = min(observed data) for maximum likelihood estimation
- All data points are used in the fit (not just tail)
- Resulting q parameter describes tail heaviness across entire distribution

This approach is mathematically rigorous and consistent with the Type I Pareto distribution definition.
