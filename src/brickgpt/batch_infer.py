import os
import pandas as pd
import transformers
from tqdm import tqdm
import argparse

import sys
import os
from brickgpt.models import BrickGPT, BrickGPTConfig
from brickgpt.render_bricks import render_bricks

def load_dataset_from_parquet(dataset_path, max_rows=None):
    df = pd.read_parquet(dataset_path)
    if max_rows is not None:
        df = df.iloc[:max_rows]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='4class_stablebrick2text_train.parquet', help='Name of parquet file in proj_develop/datasets/')
    parser.add_argument('--output_dir_prefix', type=str, default='proj_develop/batch_outputs', help='Base directory for all outputs')
    parser.add_argument('--experiment_name', type=str, default='', help='Output folder will have this as suffix')
    parser.add_argument('--caption_column', type=str, default='captions', help='Column name holding list of prompts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_rows', type=int, default=None, help='Limit the number of rows processed')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir_prefix, args.experiment_name)

    print(output_dir)

    dataset_path = os.path.join('proj_develop/datasets', args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    transformers.set_seed(args.seed)

    # Prepare config and BrickGPT object
    cfg = BrickGPTConfig()
    brickgpt = BrickGPT(cfg)

    # Load data
    df = load_dataset_from_parquet(dataset_path, args.max_rows)
    n_structures = df.shape[0]
    print(f"Running inference for {n_structures} rows from {dataset_path}")
    # breakpoint()
    # DPO data collection
    dpo_data = []

    # Prepare dataframes for stats
    rejection_stats = []
    reasons_stats = []
    regen_stats = []

    # Main batch inference loop
    for idx, row in tqdm(df.iterrows(), total=n_structures):
        structure_id = row.get('structure_id', idx)
        object_id = row.get('object_id', idx)
        category_id = row.get('category_id', idx)
        prompts = row[args.caption_column]
        for prompt in prompts:
            output = brickgpt(prompt)

            # Collected DPO dataset
            dpo_data_point = output.get('dpo_response_dict', None)
            if dpo_data_point is not None:
                dpo_data_dict = {}
                dpo_data_dict['structure_id'] = structure_id
                dpo_data_dict['object_id'] = object_id
                dpo_data_dict['category_id'] = category_id
                dpo_data_dict.update(dpo_data_point)

                dpo_data.append(dpo_data_dict)

            # Log stats
            rejection_stats.append({'structure_id': structure_id, 'prompt': prompt, 'brick_rejections': output['rejection_reasons'].total()})
            reasons_stats.append({'structure_id': structure_id, 'prompt': prompt, **output['rejection_reasons']})
            regen_stats.append({'structure_id': structure_id, 'prompt': prompt, 'regenerations': output['n_regenerations']})

    # Save DPO data
    dpo_output_path = os.path.join(output_dir, 'dpo_data.parquet')
    dpo_df = pd.DataFrame(dpo_data)
    dpo_df.to_csv(os.path.join(output_dir, 'dpo_data.csv'), index=False)
    dpo_df.to_parquet(dpo_output_path, index=False)
    print(f"Saved DPO data to {dpo_output_path}")

    # Save stats to CSV for easy later analysis
    pd.DataFrame(rejection_stats).to_csv(os.path.join(output_dir, 'brick_rejections_stats.csv'), index=False)
    pd.DataFrame(reasons_stats).to_csv(os.path.join(output_dir, 'brick_rejection_reasons.csv'), index=False)
    pd.DataFrame(regen_stats).to_csv(os.path.join(output_dir, 'regeneration_stats.csv'), index=False)

    print(f"Saved stats to {output_dir}/brick_rejections_stats.csv, brick_rejection_reasons.csv, regeneration_stats.csv")

if __name__ == '__main__':
    main()