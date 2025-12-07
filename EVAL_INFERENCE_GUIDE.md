# Evaluation Inference Guide

This guide explains how to use the fast evaluation inference script (`eval_infer.py`) to benchmark your BrickGPT models.

## Overview

The `eval_infer.py` script is designed for **fast evaluation** of BrickGPT models. Unlike `batch_infer.py` (which stores intermediate generations for DPO training), this script only stores final results, making it ideal for computing metrics and evaluating model performance.

## Key Differences from `batch_infer.py`

| Feature | `batch_infer.py` | `eval_infer.py` |
|---------|------------------|-----------------|
| Purpose | DPO training data collection | Fast model evaluation |
| Stores intermediate generations | ✅ Yes | ❌ No |
| Output format | 3 CSV files + parquet | Single JSONL file |
| Output size | Large (all generations) | Small (final results only) |
| Speed | Slower (I/O overhead) | Faster (minimal I/O) |

## Usage

### Basic Usage

```bash
python -m brickgpt.eval_infer --dataset your_dataset.parquet
```

### Common Options

```bash
# Evaluate on a specific dataset with custom model
python -m brickgpt.eval_infer \
    --dataset 4class_stablebrick2text_test.parquet \
    --model_name_or_path path/to/your/model \
    --output_name my_experiment_v1

# Evaluate only first 100 rows for quick testing
python -m brickgpt.eval_infer \
    --dataset test_dataset.parquet \
    --max_rows 100 \
    --seed 42

# Resume from a specific index
python -m brickgpt.eval_infer \
    --dataset large_dataset.parquet \
    --start_idx 1000 \
    --max_rows 500
```

### All Available Arguments

- `--dataset`: Name of parquet file in `proj_develop/datasets/` (default: `4class_stablebrick2text_train.parquet`)
- `--output_name`: Custom output filename without extension (default: auto-generated from dataset + timestamp)
- `--caption_column`: Column name containing prompts (default: `captions`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--start_idx`: Starting row index (default: `0`)
- `--max_rows`: Limit number of rows to process (default: `None` = all rows)
- `--model_name_or_path`: Model checkpoint path (default: `AvaLovelace/BrickGPT`)

## Output Format

Results are saved as **JSONL** (JSON Lines) in `proj_develop/inference/`.

Each line is a JSON object containing:

```json
{
  "structure_id": 12345,
  "object_id": 67890,
  "category_id": 3,
  "prompt": "A red car",
  "prompt_idx": 0,
  "final_sequence": "2x4 (0,0,0)\n4x2 (2,0,0)\n...",
  "n_bricks": 47,
  "n_regenerations": 3,
  "rejection_reasons": {
    "collision": 12,
    "out_of_bounds": 5,
    "ill_formatted": 1
  },
  "total_rejections": 18,
  "inference_time_seconds": 12.34,
  "stability_scores": [0.0, 0.0, 0.1, ...],
  "mean_stability_score": 0.023,
  "is_stable": true
}
```

### Field Descriptions

- `structure_id`, `object_id`, `category_id`: Identifiers from the dataset
- `prompt`: The text prompt used for generation
- `prompt_idx`: Index of prompt (if multiple prompts per structure)
- `final_sequence`: Final generated brick structure in text format
- `n_bricks`: Number of bricks in final structure
- `n_regenerations`: How many times the structure was regenerated due to instability
- `rejection_reasons`: Dictionary of rejection reasons and their counts
- `total_rejections`: Sum of all rejections
- `inference_time_seconds`: Total time for this inference
- `stability_scores`: Stability score for each brick (lower is better)
- `mean_stability_score`: Average stability score across all bricks
- `is_stable`: Boolean indicating if final structure is stable

## Analyzing Results

### Using the Utility Script

```bash
# Print summary statistics
python -m brickgpt.eval_utils proj_develop/inference/eval_results.jsonl

# Show structures with many regenerations
python -m brickgpt.eval_utils proj_develop/inference/eval_results.jsonl \
    --show-failed --min-regen 20

# Export to CSV for external analysis
python -m brickgpt.eval_utils proj_develop/inference/eval_results.jsonl --export-csv
```

### Programmatic Analysis

```python
from brickgpt.eval_utils import load_eval_results, print_eval_summary

# Load results
df = load_eval_results('proj_develop/inference/eval_results.jsonl')

# Print summary
print_eval_summary(df)

# Custom analysis
print(f"Success rate: {(df['is_stable']).mean() * 100:.1f}%")
print(f"Avg time: {df['inference_time_seconds'].mean():.2f}s")

# Find problematic structures
problematic = df[df['n_regenerations'] > 50]
print(problematic[['structure_id', 'prompt', 'n_regenerations']])
```

## Example Workflow

### 1. Run evaluation on test set

```bash
python -m brickgpt.eval_infer \
    --dataset test_set.parquet \
    --model_name_or_path ./my_finetuned_model \
    --output_name baseline_test \
    --seed 42
```

### 2. Analyze results

```bash
python -m brickgpt.eval_utils \
    proj_develop/inference/eval_baseline_test_20250107_143022.jsonl \
    --show-failed --export-csv
```

### 3. Compare with another model

```bash
# Run inference with different model
python -m brickgpt.eval_infer \
    --dataset test_set.parquet \
    --model_name_or_path ./my_improved_model \
    --output_name improved_test \
    --seed 42

# Load both and compare
python
>>> from brickgpt.eval_utils import load_eval_results
>>> baseline = load_eval_results('proj_develop/inference/eval_baseline_test_*.jsonl')
>>> improved = load_eval_results('proj_develop/inference/eval_improved_test_*.jsonl')
>>> print(f"Baseline stability: {baseline['is_stable'].mean():.2%}")
>>> print(f"Improved stability: {improved['is_stable'].mean():.2%}")
```

## Performance Tips

1. **Use `--max_rows`** for quick iterations during development
2. **JSONL format** allows streaming processing for very large evaluations
3. Results are written incrementally, so you can monitor progress in real-time:
   ```bash
   # In another terminal while inference is running
   tail -f proj_develop/inference/eval_*.jsonl | jq .
   ```
4. The script flushes after each write, so partial results are always saved even if interrupted

## Metrics You Can Compute

From the output data, you can calculate:

- **Success Rate**: `% is_stable == True`
- **Generation Efficiency**: `mean(n_regenerations)`, `mean(total_rejections)`
- **Inference Speed**: `mean(inference_time_seconds)`
- **Structure Quality**: `mean(mean_stability_score)`
- **Rejection Analysis**: Distribution of rejection reasons
- **Prompt Performance**: Group by prompt to see which prompts are harder

## Troubleshooting

**Q: I'm getting OOM errors**
- Use `--max_rows` to process in batches
- Reduce `max_bricks` or `world_dim` in BrickGPTConfig

**Q: Results file is too large**
- The `final_sequence` and `stability_scores` fields take most space
- Consider post-processing to remove these if you don't need them

**Q: How do I compare two models?**
- Use the same dataset and seed for both runs
- Load both JSONL files into DataFrames and compare metrics

**Q: Can I interrupt and resume?**
- Yes! Use `--start_idx` to resume from where you left off
- JSONL format preserves all previously written results
