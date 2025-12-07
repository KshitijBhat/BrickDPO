#!/usr/bin/env python3
"""Merge CSV stats files and parquet files from specified subdirectories."""

import pandas as pd
from pathlib import Path

# Configuration: Specify the base path and list of folder names to merge
BASE_PATH = Path(__file__).parent / "datasets" / "dpo_datasets"
FOLDER_NAMES = [
    # "test_200_299",
    # "test_300_353",
    "train_000_010",
    "train_011_039",
    "train_040_099",
    "train_200_399",
    "train_400_599",
    "train_600_649",
    "train_650_799",
    "train_800_853",
    "train_854_979",
    # Add more folder names here as needed
]

# Output directory for merged files
output_dir = BASE_PATH / "combined_dataset"
output_dir.mkdir(exist_ok=True)

# Get directories from the specified folder names
source_dirs = []
for folder_name in FOLDER_NAMES:
    folder_path = BASE_PATH / folder_name
    if folder_path.exists() and folder_path.is_dir():
        source_dirs.append(folder_path)
    else:
        print(f"Warning: Folder '{folder_name}' not found at {folder_path}")

source_dirs.sort()

print(f"Found {len(source_dirs)} directories to merge from {len(FOLDER_NAMES)} specified folders")

# Files to merge (CSV files only - parquet files are handled separately below)
files_to_merge = [
    "brick_rejections_stats.csv",
    "brick_rejection_reasons.csv", 
    "regeneration_stats.csv",
]

for filename in files_to_merge:
    dfs = []
    for source_dir in source_dirs:
        filepath = source_dir / filename
        if filepath.exists():
            try:
                # Try standard read first
                df = pd.read_csv(filepath)
                dfs.append(df)
                print(f"  Read {len(df)} rows from {source_dir.name}/{filename}")
            except Exception as e:
                # If that fails, try with Python engine and skip bad lines
                try:
                    df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
                    dfs.append(df)
                    print(f"  Read {len(df)} rows from {source_dir.name}/{filename} (with error handling)")
                except Exception as e2:
                    print(f"  Error reading {source_dir.name}/{filename}: {e2}")
                    continue
    
    if dfs:
        # Align columns - use union of all columns, fill missing with NaN
        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)
        all_columns = sorted(list(all_columns))
        
        # Reindex each dataframe to have all columns
        aligned_dfs = []
        for df in dfs:
            aligned_df = df.reindex(columns=all_columns)
            aligned_dfs.append(aligned_df)
        
        merged_df = pd.concat(aligned_dfs, ignore_index=True)
        output_path = output_dir / filename
        merged_df.to_csv(output_path, index=False)
        print(f"  Saved {len(merged_df)} rows to {output_path}\n")
    else:
        print(f"  No files found for {filename}\n")

# Merge parquet files
print("="*50)
print("Merging parquet files...")
parquet_filename = "dpo_data.parquet"
parquet_dfs = []

for source_dir in source_dirs:
    filepath = source_dir / parquet_filename
    if filepath.exists():
        try:
            df = pd.read_parquet(filepath)
            parquet_dfs.append(df)
            print(f"  Read {len(df)} rows from {source_dir.name}/{parquet_filename}")
        except Exception as e:
            print(f"  Error reading {source_dir.name}/{parquet_filename}: {e}")
            continue

if parquet_dfs:
    # Ensure consistent data types across all dataframes before merging
    for df in parquet_dfs:
        # Convert category_id to string if it exists (to preserve leading zeros)
        if 'category_id' in df.columns:
            df['category_id'] = df['category_id'].astype(str)
    
    merged_parquet_df = pd.concat(parquet_dfs, ignore_index=True)
    output_parquet_path = output_dir / parquet_filename
    merged_parquet_df.to_parquet(output_parquet_path, index=False)
    print(f"  Saved {len(merged_parquet_df)} rows to {output_parquet_path}\n")
else:
    print(f"  No parquet files found for {parquet_filename}\n")

print("Done!")

