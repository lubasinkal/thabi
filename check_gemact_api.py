from gemact import LossModel

methods = [m for m in dir(LossModel) if not m.startswith('_')]
print("Available LossModel methods:")
for m in sorted(methods):
    print(f"  - {m}")

# Check specifically for tail risk measures
print("\nTail risk related methods:")
for m in methods:
    if any(word in m.lower() for word in ['tail', 'var', 'tvar', 'cte', 'es', 'quantile', 'ppf', 'percentile']):
        print(f"  - {m}")
