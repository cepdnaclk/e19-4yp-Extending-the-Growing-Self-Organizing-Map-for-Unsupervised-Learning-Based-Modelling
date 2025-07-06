import pandas as pd

# Step 1: Load the Excel file
file_path = "GSE5281_sample_characteristics.xlsx"  # Make sure it's in your working directory
df = pd.read_excel(file_path)

# Step 2: Drop ID/identifier columns (non-informative)
df = df.drop(columns=["GEO Accession:", "MAGE Identifier:", "Bio-Source Name:"])

# Step 3: Drop constant-value columns (e.g., Organism: if all = 'Human')
df = df.loc[:, df.nunique() > 1]

# Step 4: Preview the result
print("Cleaned dataset preview:")
print(df.head())

# Step 5: Save the cleaned DataFrame to CSV for GSOM or further processing
df.to_csv("GSE5281_cleaned.csv", index=False)
print("✅ Cleaned data saved as 'GSE5281_cleaned.csv'")
