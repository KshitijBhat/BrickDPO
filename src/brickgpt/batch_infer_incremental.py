import os
import pandas as pd
import transformers
from tqdm import tqdm
import argparse

import sys
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
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
    parser.add_argument('--idx_start', type=int, default=None, help='Inclusive start index within the dataset')
    parser.add_argument('--idx_end', type=int, default=None, help='Exclusive end index within the dataset')
    args = parser.parse_args()

    dataset_path = os.path.join('proj_develop/datasets', args.dataset)
    transformers.set_seed(args.seed)

    # Prepare config and BrickGPT object
    cfg = BrickGPTConfig()
    brickgpt = BrickGPT(cfg)

    # Load data
    df = load_dataset_from_parquet(dataset_path, args.max_rows)
    total_rows = df.shape[0]

    idx_start = args.idx_start if args.idx_start is not None else 0
    idx_end = args.idx_end if args.idx_end is not None else total_rows

    if not (0 <= idx_start <= idx_end <= total_rows):
        raise SystemExit(f"Index range [{idx_start}, {idx_end}) is invalid for dataset with {total_rows} rows.")

    if idx_start != 0 or idx_end != total_rows:
        df = df.iloc[idx_start:idx_end]

    exp_name = args.experiment_name.strip() if args.experiment_name else ''
    if args.idx_start is not None or args.idx_end is not None:
        idx_suffix = f"idx_{idx_start}_{idx_end}"
        exp_name = f"{exp_name}_{idx_suffix}" if exp_name else idx_suffix

    output_dir = os.path.join(args.output_dir_prefix, exp_name)

    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dpo_csv_path = Path(os.path.join(output_dir, 'dpo_data.csv'))
    dpo_parquet_path = Path(os.path.join(output_dir, 'dpo_data.parquet'))
    rejection_csv_path = Path(os.path.join(output_dir, 'brick_rejections_stats.csv'))
    reasons_csv_path = Path(os.path.join(output_dir, 'brick_rejection_reasons.csv'))
    regen_csv_path = Path(os.path.join(output_dir, 'regeneration_stats.csv'))

    for path in [dpo_csv_path, dpo_parquet_path, rejection_csv_path, reasons_csv_path, regen_csv_path]:
        if path.exists():
            path.unlink()

    n_structures = df.shape[0]
    idx_range_desc = f"indices {idx_start} to {idx_end - 1}" if n_structures else f"empty slice [{idx_start}, {idx_end})"
    print(f"Running inference for {n_structures} rows from {dataset_path} ({idx_range_desc})")
    # breakpoint()
    # DPO data collection
    dpo_data = []

    # Prepare dataframes for stats
    rejection_stats = []
    reasons_stats = []
    regen_stats = []

    last_dpo_idx = 0
    last_rejection_idx = 0
    last_reasons_idx = 0
    last_regen_idx = 0
    dpo_parquet_writer = None

    def flush_dpo(data_list, start_idx):
        nonlocal dpo_parquet_writer
        new_rows = data_list[start_idx:]
        if not new_rows:
            return start_idx
        df = pd.DataFrame(new_rows)
        if df.empty:
            return start_idx
        header_needed = not dpo_csv_path.exists()
        df.to_csv(dpo_csv_path, mode='a', header=header_needed, index=False)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if dpo_parquet_writer is None:
            dpo_parquet_writer = pq.ParquetWriter(str(dpo_parquet_path), table.schema)
        dpo_parquet_writer.write_table(table)
        return len(data_list)

    def flush_csv(data_list, start_idx, path):
        new_rows = data_list[start_idx:]
        if not new_rows:
            return start_idx
        df = pd.DataFrame(new_rows)
        if df.empty:
            return start_idx
        header_needed = not path.exists()
        df.to_csv(path, mode='a', header=header_needed, index=False)
        return len(data_list)

    interrupted = False

    try:
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

            last_dpo_idx = flush_dpo(dpo_data, last_dpo_idx)
            last_rejection_idx = flush_csv(rejection_stats, last_rejection_idx, rejection_csv_path)
            last_reasons_idx = flush_csv(reasons_stats, last_reasons_idx, reasons_csv_path)
            last_regen_idx = flush_csv(regen_stats, last_regen_idx, regen_csv_path)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted by user; partial outputs saved to disk.")
    finally:
        if dpo_parquet_writer is not None:
            dpo_parquet_writer.close()

    if dpo_csv_path.exists():
        print(f"Saved DPO data to {dpo_parquet_path}")
    if rejection_csv_path.exists() and reasons_csv_path.exists() and regen_csv_path.exists():
        print(f"Saved stats to {output_dir}/brick_rejections_stats.csv, brick_rejection_reasons.csv, regeneration_stats.csv")

    if interrupted:
        raise SystemExit(130)

if __name__ == '__main__':
    main()