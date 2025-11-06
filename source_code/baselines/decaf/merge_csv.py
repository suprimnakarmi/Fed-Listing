import pandas as pd
import glob
import os

# Path to the directory containing the CSV files
csv_dir = "./"  # Change this if your CSVs are in another folder

# Glob all relevant CSV files
csv_files = sorted(glob.glob(os.path.join(csv_dir, "records_*.csv")))

# List to hold individual DataFrames
df_list = []

# Read and append file source info
for file in csv_files:
    df = pd.read_csv(file)
    # df['Source_File'] = os.path.basename(file)
    df_list.append(df)

# Concatenate all dataframes
merged_df = pd.concat(df_list, ignore_index=True)

# Save merged CSV
merged_df.to_csv("merged_records.csv", index=False)
print("Saved merged dataset to 'merged_records.csv'")
