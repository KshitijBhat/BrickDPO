import os
import json
import time
import pandas as pd
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from datetime import datetime

from brickgpt.models import BrickGPT, BrickGPTConfig


def load_hf_model_with_subfolder(model_path, subfolder, device):
    """Load a merged model from HuggingFace with subfolder structure."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder=subfolder)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        subfolder=subfolder,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model


def load_dataset_from_parquet(dataset_path, start_idx=0, max_rows=None):
    """Load dataset from parquet file with optional row limiting."""
    df = pd.read_parquet(dataset_path)
    if max_rows is not None:
        df = df.iloc[start_idx : start_idx + max_rows]
    elif start_idx > 0:
        df = df.iloc[start_idx:]
    return df


def main():
    parser = argparse.ArgumentParser(description='Fast evaluation inference script for BrickGPT')
    parser.add_argument(
        '--dataset',
        type=str,
        default='4class_stablebrick2text_train.parquet',
        help='Name of parquet file in proj_develop/datasets/'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default=None,
        help='Output filename (without extension). If not provided, will use dataset name + timestamp'
    )
    parser.add_argument(
        '--caption_column',
        type=str,
        default='captions',
        help='Column name holding list of prompts'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Starting index for processing rows'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=None,
        help='Limit the number of rows processed'
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='AvaLovelace/BrickGPT',
        help='Model checkpoint for inference'
    )
    parser.add_argument(
        '--use_hf',
        action='store_true',
        help='Load merged model from HuggingFace with subfolder structure'
    )
    parser.add_argument(
        '--hf_subfolder',
        type=str,
        default='merged_dpo_brickgpt',
        help='Subfolder when using --use_hf (default: merged_dpo_brickgpt)'
    )
    args = parser.parse_args()

    # Setup paths
    dataset_path = os.path.join('proj_develop/datasets', args.dataset)
    output_dir = 'proj_develop/inference'
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    if args.output_name is None:
        dataset_basename = os.path.splitext(args.dataset)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'eval_{dataset_basename}_{timestamp}'
    else:
        output_name = args.output_name

    output_path = os.path.join(output_dir, f'{output_name}.jsonl')

    # Set seed
    transformers.set_seed(args.seed)

    # Initialize BrickGPT
    cfg = BrickGPTConfig(model_name_or_path=args.model_name_or_path)
    brickgpt = BrickGPT(cfg)

    # If using HuggingFace merged model, override the tokenizer and model
    if args.use_hf:
        print(f"Loading HuggingFace model from subfolder: {args.hf_subfolder}")
        tokenizer, model = load_hf_model_with_subfolder(
            args.model_name_or_path,
            args.hf_subfolder,
            brickgpt.device
        )
        brickgpt.llm.tokenizer = tokenizer
        brickgpt.llm.model = model

    # Load dataset
    df = load_dataset_from_parquet(dataset_path, start_idx=args.start_idx, max_rows=args.max_rows)
    n_structures = df.shape[0]

    print(f"Running evaluation inference on {n_structures} structures")
    print(f"Model: {cfg.model_name_or_path}")
    if args.use_hf:
        print(f"  Loaded from HuggingFace subfolder: {args.hf_subfolder}")
    print(f"Dataset: {dataset_path}")
    print(f"Starting index: {args.start_idx}")
    print(f"Output file: {output_path}")
    print("-" * 80)

    # Track overall statistics
    total_inferences = 0
    total_time = 0

    # Open output file for writing
    with open(output_path, 'w') as f:
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=n_structures, desc="Processing structures"):
            structure_id = row.get('structure_id', idx)
            object_id = row.get('object_id', idx)
            category_id = row.get('category_id', idx)
            prompts = row[args.caption_column]

            # Handle both list and single prompt cases
            if not isinstance(prompts, list):
                prompts = [prompts]

            # Run inference for each prompt
            for prompt_idx, prompt in enumerate(prompts):
                # Time the inference
                start_time = time.time()
                output = brickgpt(prompt)
                end_time = time.time()
                inference_time = end_time - start_time

                # Extract data from output
                final_bricks = output['bricks']
                n_regenerations = output['n_regenerations']
                rejection_reasons = dict(output['rejection_reasons'])  # Convert Counter to dict

                # Get stability scores for all blocks
                stability_scores = brickgpt._stability_scores(final_bricks).tolist()

                # Prepare result record
                result = {
                    'structure_id': structure_id,
                    'object_id': object_id,
                    'category_id': category_id,
                    'prompt': prompt,
                    'prompt_idx': prompt_idx,
                    'final_sequence': final_bricks.to_txt(),
                    'n_bricks': len(final_bricks),
                    'n_regenerations': n_regenerations,
                    'rejection_reasons': rejection_reasons,
                    'total_rejections': sum(rejection_reasons.values()),
                    'inference_time_seconds': inference_time,
                    'stability_scores': stability_scores,
                    'mean_stability_score': float(brickgpt._stability_scores(final_bricks).mean()),
                    'is_stable': brickgpt._is_stable(final_bricks),
                }

                # Write to JSONL file (one JSON object per line)
                f.write(json.dumps(result) + '\n')
                f.flush()  # Ensure data is written immediately

                total_inferences += 1
                total_time += inference_time

    # Print summary statistics
    print("-" * 80)
    print(f"Evaluation complete!")
    print(f"Total inferences: {total_inferences}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per inference: {total_time/total_inferences:.2f}s")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
