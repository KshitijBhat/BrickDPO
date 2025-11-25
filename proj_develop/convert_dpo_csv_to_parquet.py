#!/usr/bin/env python3
"""
Script to convert dpo_data.csv files to parquet format.
This script reads CSV files and properly converts list columns before saving as parquet.
"""

import pandas as pd
import ast
import json
from pathlib import Path


def parse_list_column(value):
    """Parse a string representation of a list into an actual list."""
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            # Try parsing as Python literal (handles lists with strings)
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If that fails, try JSON parsing
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # If both fail, return as string
                return value
    return value


def convert_csv_to_parquet(csv_path, parquet_path):
    """Convert a CSV file to parquet format, properly handling list columns."""
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Parse list columns (generated_bricks, stable_partial_structure, average_stability_scores)
    list_columns = ['generated_bricks', 'stable_partial_structure', 'average_stability_scores']
    
    for col in list_columns:
        if col in df.columns:
            print(f"Parsing column: {col}")
            df[col] = df[col].apply(parse_list_column)
    
    # Save as parquet
    print(f"Writing parquet to: {parquet_path}")
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    
    print(f"Successfully converted! Final shape: {df.shape}")
    
    # Verify the parquet file can be read
    print("Verifying parquet file...")
    df_verify = pd.read_parquet(parquet_path)
    print(f"Verified! Read back shape: {df_verify.shape}")
    print(f"First row sample:")
    print(df_verify.iloc[0])
    print()


def main():
    base_dir = Path(__file__).parent / "datasets" / "dpo_datasets"
    
    # Process test_011_039
    csv_path_011 = base_dir / "test_011_039" / "dpo_data.csv"
    parquet_path_011 = base_dir / "test_011_039" / "dpo_data.parquet"
    
    if csv_path_011.exists():
        convert_csv_to_parquet(csv_path_011, parquet_path_011)
    else:
        print(f"Warning: {csv_path_011} does not exist")
    
    # Process test_040_099
    csv_path_040 = base_dir / "test_040_099" / "dpo_data.csv"
    parquet_path_040 = base_dir / "test_040_099" / "dpo_data.parquet"
    
    if csv_path_040.exists():
        convert_csv_to_parquet(csv_path_040, parquet_path_040)
    else:
        print(f"Warning: {csv_path_040} does not exist")


if __name__ == "__main__":
    main()

