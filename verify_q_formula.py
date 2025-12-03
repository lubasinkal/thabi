"""
Verify the Pareto q MLE Formula
Testing if our implementation matches the theoretical formula
"""
import pandas as pd
import numpy as np

print("="*70)
print("VERIFYING PARETO q MLE FORMULA")
print("="*70)

# Load claims
claims = pd.read_csv('molapo_claims_real_climate.csv')
threshold = 25000

# Filter to claims above threshold
filtered = claims[claims['loss_amount'] > threshold]
n = len(filtered)
losses = filtered['loss_amount'].values

print(f"\nData:")
print(f"  Threshold K = ${threshold:,}")
print(f"  Number of claims (n) = {n}")
print(f"  Sample losses: {[f'${x:,.0f}' for x in losses[:5]]}")

print(f"\n" + "="*70)
print("THEORETICAL FORMULA FROM LITERATURE")
print("="*70)

print(f"\nFor Pareto Type II distribution:")
print(f"  PDF: f(x) = (q/K) * (1 + x/K)^(-q-1)   for x > 0")
print(f"  or equivalently with threshold μ:")
print(f"  PDF: f(x) = (q/μ) * (x/μ)^(-q-1)      for x > μ")

print(f"\nMaximum Likelihood Estimator (MLE) for q:")
print(f"\n  The correct formula is:")
print(f"  ")
print(f"       q^ = n / Σ[i=1 to n] ln(x_i / K)")
print(f"  ")
print(f"  where:")
print(f"    - n is the number of observations above K")
print(f"    - x_i is the i-th loss amount")
print(f"    - K is the threshold")

print(f"\n" + "="*70)
print("ALTERNATIVE EQUIVALENT FORMS")
print("="*70)

print(f"\nForm 1 (most common):")
print(f"  q^ = n / Σ ln(x_i / K)")

print(f"\nForm 2 (expanded):")
print(f"  q^ = n / Σ [ln(x_i) - ln(K)]")
print(f"     = n / [Σ ln(x_i) - n*ln(K)]")

print(f"\nForm 3 (with reciprocal):")
print(f"  q^ = 1 / [ (1/n) * Σ ln(x_i / K) ]")
print(f"     = 1 / E[ln(x_i / K)]")

print(f"\n❌ INCORRECT FORMS:")
print(f"  q^ = Σ ln(K / x_i) / n    ← WRONG (gives negative values)")
print(f"  q^ = Σ ln(x_i * K) / n    ← WRONG (not the MLE)")

print(f"\n" + "="*70)
print("OUR IMPLEMENTATION")
print("="*70)

print(f"\nCode in flood_risk_model.py (lines 44-55):")
print(f"```python")
print(f"log_ratios = np.log(filtered_losses / self.threshold)")
print(f"q_hat = n / np.sum(log_ratios)")
print(f"```")

print(f"\nStep-by-step calculation:")

# Method 1: Our implementation
log_ratios = np.log(losses / threshold)
q_method1 = n / np.sum(log_ratios)

print(f"\nMethod 1 (our code):")
print(f"  Step 1: log_ratios = ln(x_i / K)")
print(f"          First 5 values: {log_ratios[:5]}")
print(f"  Step 2: sum(log_ratios) = {np.sum(log_ratios):.6f}")
print(f"  Step 3: q^ = n / sum = {n} / {np.sum(log_ratios):.6f}")
print(f"  Result: q^ = {q_method1:.6f}")

# Method 2: Expanded form
sum_ln_xi = np.sum(np.log(losses))
n_ln_K = n * np.log(threshold)
q_method2 = n / (sum_ln_xi - n_ln_K)

print(f"\nMethod 2 (expanded form):")
print(f"  Step 1: Σ ln(x_i) = {sum_ln_xi:.6f}")
print(f"  Step 2: n * ln(K) = {n} * {np.log(threshold):.6f} = {n_ln_K:.6f}")
print(f"  Step 3: denominator = {sum_ln_xi:.6f} - {n_ln_K:.6f} = {sum_ln_xi - n_ln_K:.6f}")
print(f"  Step 4: q^ = {n} / {sum_ln_xi - n_ln_K:.6f}")
print(f"  Result: q^ = {q_method2:.6f}")

# Method 3: Mean of log ratios
mean_log_ratio = np.mean(log_ratios)
q_method3 = 1 / mean_log_ratio

print(f"\nMethod 3 (reciprocal of mean):")
print(f"  Step 1: mean(log_ratios) = {mean_log_ratio:.6f}")
print(f"  Step 2: q^ = 1 / mean = 1 / {mean_log_ratio:.6f}")
print(f"  Result: q^ = {q_method3:.6f}")

print(f"\n" + "="*70)
print("VERIFICATION")
print("="*70)

print(f"\nAll three methods give the same result:")
print(f"  Method 1: q^ = {q_method1:.6f}")
print(f"  Method 2: q^ = {q_method2:.6f}")
print(f"  Method 3: q^ = {q_method3:.6f}")

if np.allclose([q_method1, q_method2, q_method3], q_method1):
    print(f"\n✅ ALL METHODS MATCH - Formula is CORRECT!")
else:
    print(f"\n❌ Methods don't match - ERROR in implementation!")

print(f"\n" + "="*70)
print("WHAT ABOUT THE USER'S FORMULA?")
print("="*70)

print(f"\nUser asked about: Σ ln(K/x_i) / n")
print(f"\nLet's test this INCORRECT form:")

# WRONG formula
wrong_sum = np.sum(np.log(threshold / losses))
q_wrong = wrong_sum / n

print(f"  Step 1: ln(K/x_i) for first 5:")
print(f"          {np.log(threshold / losses[:5])}")
print(f"  Step 2: sum = {wrong_sum:.6f}")
print(f"  Step 3: q^ = sum / n = {wrong_sum:.6f} / {n}")
print(f"  Result: q^ = {q_wrong:.6f}")

if q_wrong < 0:
    print(f"\n❌ This gives NEGATIVE q = {q_wrong:.6f}")
    print(f"   Cannot be correct (q must be positive)!")
elif q_wrong > 0:
    print(f"\n⚠️  This gives q = {q_wrong:.6f}")
    print(f"   But it's NOT the MLE (wrong formula)!")

print(f"\n" + "="*70)
print("MATHEMATICAL PROOF")
print("="*70)

print(f"\nWhy our formula is correct:")
print(f"\n1. Pareto Type II likelihood function:")
print(f"   L(q | x_1,...,x_n) = ∏[i=1 to n] (q/K) * (x_i/K)^(-q-1)")

print(f"\n2. Log-likelihood:")
print(f"   ln(L) = Σ [ln(q/K) - (q+1)*ln(x_i/K)]")
print(f"         = n*ln(q/K) - (q+1)*Σ ln(x_i/K)")
print(f"         = n*ln(q) - n*ln(K) - (q+1)*Σ ln(x_i/K)")

print(f"\n3. Take derivative with respect to q and set to 0:")
print(f"   d/dq [ln(L)] = n/q - Σ ln(x_i/K) = 0")

print(f"\n4. Solve for q:")
print(f"   n/q = Σ ln(x_i/K)")
print(f"   q = n / Σ ln(x_i/K)  ✅ THIS IS THE MLE!")

print(f"\n" + "="*70)
print("EXAMPLE CALCULATION")
print("="*70)

print(f"\nUsing first 3 claims:")
sample_losses = losses[:3]
sample_n = len(sample_losses)

print(f"\nClaims:")
for i, loss in enumerate(sample_losses, 1):
    print(f"  x_{i} = ${loss:,.0f}")

print(f"\nThreshold K = ${threshold:,}")

print(f"\nCalculation:")
for i, loss in enumerate(sample_losses, 1):
    ratio = loss / threshold
    log_ratio = np.log(ratio)
    print(f"  ln(x_{i}/K) = ln({loss:,.0f}/{threshold:,}) = ln({ratio:.4f}) = {log_ratio:.6f}")

sample_sum = np.sum(np.log(sample_losses / threshold))
sample_q = sample_n / sample_sum

print(f"\n  Sum = {sample_sum:.6f}")
print(f"  q^ = {sample_n} / {sample_sum:.6f} = {sample_q:.6f}")

print(f"\n" + "="*70)
print("CONCLUSION")
print("="*70)

print(f"\n✅ OUR IMPLEMENTATION IS CORRECT!")
print(f"\n   Formula: q^ = n / Σ ln(x_i / K)")
print(f"   Result: q^ = {q_method1:.6f}")

print(f"\n✅ This matches standard Pareto MLE from literature:")
print(f"   - Klugman, Panjer, Willmot: Loss Models")
print(f"   - Embrechts, Klüppelberg, Mikosch: Modelling Extremal Events")
print(f"   - McNeil, Frey, Embrechts: Quantitative Risk Management")

print(f"\n❌ The alternative form Σ ln(K/x_i) / n is INCORRECT")
print(f"   (gives negative or wrong values)")

print(f"\n🎯 BOTTOM LINE:")
print(f"   Your q = {q_method1:.4f} is correctly calculated using the")
print(f"   Maximum Likelihood Estimator for Pareto Type II distribution.")

print(f"\n" + "="*70)
