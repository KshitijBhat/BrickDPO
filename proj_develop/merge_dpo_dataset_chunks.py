#!/usr/bin/env python3
"""Merge CSV stats files and parquet files from all test* subdirectories."""

import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent / "datasets" / "dpo_datasets"
output_dir = base_dir / "combined_dataset"
output_dir.mkdir(exist_ok=True)

# Find all test* directories
test_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("test")]
test_dirs.sort()

print(f"Found {len(test_dirs)} test directories")

# Files to merge
files_to_merge = [
    "brick_rejections_stats.csv",
    "brick_rejection_reasons.csv", 
    "regeneration_stats.csv"
]

for filename in files_to_merge:
    dfs = []
    for test_dir in test_dirs:
        filepath = test_dir / filename
        if filepath.exists():
            try:
                # Try standard read first
                df = pd.read_csv(filepath)
                dfs.append(df)
                print(f"  Read {len(df)} rows from {test_dir.name}/{filename}")
            except Exception as e:
                # If that fails, try with Python engine and skip bad lines
                try:
                    df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
                    dfs.append(df)
                    print(f"  Read {len(df)} rows from {test_dir.name}/{filename} (with error handling)")
                except Exception as e2:
                    print(f"  Error reading {test_dir.name}/{filename}: {e2}")
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
parquet_filename = "dpo_data_process.parquet"
parquet_dfs = []

for test_dir in test_dirs:
    filepath = test_dir / parquet_filename
    if filepath.exists():
        try:
            df = pd.read_parquet(filepath)
            parquet_dfs.append(df)
            print(f"  Read {len(df)} rows from {test_dir.name}/{parquet_filename}")
        except Exception as e:
            print(f"  Error reading {test_dir.name}/{parquet_filename}: {e}")
            continue

if parquet_dfs:
    merged_parquet_df = pd.concat(parquet_dfs, ignore_index=True)
    output_parquet_path = output_dir / parquet_filename
    merged_parquet_df.to_parquet(output_parquet_path, index=False)
    print(f"  Saved {len(merged_parquet_df)} rows to {output_parquet_path}\n")
else:
    print(f"  No parquet files found for {parquet_filename}\n")

print("Done!")

