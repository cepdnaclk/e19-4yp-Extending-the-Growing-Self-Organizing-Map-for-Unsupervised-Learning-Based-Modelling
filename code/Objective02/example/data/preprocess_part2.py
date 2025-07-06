import pandas as pd

# Step 1: Load the cleaned dataset
df = pd.read_csv("GSE5281_cleaned.csv")

# Step 2: Clean the Age column — extract number from strings like "63 years"
if "Age:" in df.columns:
    df["Age:"] = df["Age:"].astype(str).str.extract(r"(\d+)").astype(float)

# Step 3: Identify categorical columns, EXCLUDING "Sample Name:"
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [col for col in cat_cols if col != "Sample Name:"]

# Step 4: Label encode all categorical columns except Sample Name, and print the mappings
for col in cat_cols:
    df[col] = df[col].astype("category")

    print(f"\n🔹 Mapping for column: {col}")
    for code, category in enumerate(df[col].cat.categories):
        print(f"  {code} → {category}")

    df[col] = df[col].cat.codes

# Step 5: Preview numeric result with Sample Name preserved
print("\n✅ Numerical dataset preview:")
print(df.head())

# Step 6: Save final numeric file (with Sample Name retained)
df.to_csv("GSE5281_numeric.csv", index=False)
print("✅ Fully numeric dataset saved as 'GSE5281_numeric.csv'")

