import pandas as pd

# Read the Excel file
df = pd.read_excel('data/CAT Claims.csv')

print("=== File Structure ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\n=== First 10 rows ===")
print(df.head(10))
print(f"\n=== Data Types ===")
print(df.dtypes)
print(f"\n=== Excess Column Info ===")
if 'Excess' in df.columns:
    print(f"Excess values (unique): {sorted(df['Excess'].unique())}")
    print(f"Excess min: {df['Excess'].min()}, Excess max: {df['Excess'].max()}")
    print(f"Excess value counts:")
    print(df['Excess'].value_counts().sort_index())
else:
    print("Excess column not found!")

print(f"\n=== Sample Data ===")
print(df[['ClaimNum', 'Total Claim', 'Excess', 'Amount Paid', 'LossDate']].head(15))
