# ✅ Pareto MLE Formula - CLARIFIED

## 🎯 **The CORRECT Formula**

For Pareto Type II distribution with threshold K, the Maximum Likelihood Estimator for shape parameter q is:

```
        n
q^ = ─────────
     Σ ln(xi/K)
     i=1 to n
```

**Where**:
- n = number of claims above threshold K
- xi = size of i-th claim (loss amount)
- K = threshold (e.g., $25,000)

---

## 💻 **Implementation**

```python
# Filter claims above threshold
filtered_losses = claims[claims['loss_amount'] > threshold]['loss_amount']
n = len(filtered_losses)

# Calculate MLE
log_ratios = np.log(filtered_losses / threshold)  # ln(xi/K)
q_hat = n / np.sum(log_ratios)                    # n / Σ ln(xi/K)
```

---

## 📊 **Example Calculation**

**Data**:
- Claims: [$84,418, $29,800, $27,270]
- Threshold: $25,000
- n = 3

**Step-by-step**:
```
ln(x₁/K) = ln(84,418/25,000) = ln(3.377) = 1.217
ln(x₂/K) = ln(29,800/25,000) = ln(1.192) = 0.176
ln(x₃/K) = ln(27,270/25,000) = ln(1.091) = 0.087

Sum = 1.217 + 0.176 + 0.087 = 1.479

q^ = n / sum = 3 / 1.479 = 2.028
```

---

## ❌ **INCORRECT Variations**

### **Wrong Form 1**: Σ ln(K/xi) / n
```
This gives: Σ ln(K/xi) / n = -0.646 (NEGATIVE!)
Problem: q must be positive, this is clearly wrong
```

### **Wrong Form 2**: n / Σ ln(K/xi)
```
This gives: n / Σ ln(K/xi) = n / (-161.5) = -1.548
Problem: Still negative!
```

---

## 📚 **References**

This is the standard Pareto MLE found in:

1. **Klugman, Panjer, Willmot** (2012). *Loss Models: From Data to Decisions*
   - Chapter 16.3.2, Equation 16.15

2. **Embrechts, Klüppelberg, Mikosch** (1997). *Modelling Extremal Events*
   - Chapter 3.4, Pareto Distribution

3. **McNeil, Frey, Embrechts** (2015). *Quantitative Risk Management*
   - Chapter 10.2, Generalized Pareto Distribution

---

## ✅ **Your Results**

For Molapo flood data:
- n = 250 claims
- Threshold K = $25,000
- Σ ln(xi/K) = 161.518

**Calculation**:
```
q^ = 250 / 161.518 = 1.5478
```

**Interpretation**:
- q = 1.55 indicates moderately heavy tail
- Expected large losses in tail
- Consistent with flood insurance data

---

## 🔄 **Alternative Equivalent Forms**

All of these are mathematically equivalent:

### **Form 1** (what we use):
```
q^ = n / Σ ln(xi/K)
```

### **Form 2** (expanded):
```
q^ = n / [Σ ln(xi) - n*ln(K)]
```

### **Form 3** (mean reciprocal):
```
q^ = 1 / mean[ln(xi/K)]
   = 1 / [(1/n) * Σ ln(xi/K)]
```

All give the same answer: **q = 1.5478**

---

## 📝 **Note on Documentation**

The formula in `steps.md` has Unicode formatting issues:
```
q^​=∑i=1n​ln(Kxi​​)n​  ← Garbled display
```

Should be read as:
```
q^ = n / Σ[i=1 to n] ln(xi/K)  ← Correct interpretation
```

The implementation is correct!
