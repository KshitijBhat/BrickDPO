## Running batch inference

You can run batch inference over a dataset of structures using:

```zsh
uv run batch_infer
```

```zsh
/home/kneepolean/.local/bin/uv run infer
```

This script processes each structure in the dataset and records BrickGPT's rejection and regeneration statistics.

The **four most important arguments** for batch inference are:

- `--dataset`: name of the parquet file inside `proj_develop/datasets/`
- `--output_dir_prefix`: base directory where all outputs will be written
- `--experiment_name`: name of the subfolder for this run
- `--max_samples`: number of structures to process

See `uv run batch_infer -h` for the full list of options.

### Dataset structure

Each structure in the dataset includes **five prompts** (stored in the `captions` column). BrickGPT generates one output per prompt and logs statistics for each promp
brick_rejections_stats.csv — total rejection count per prompt
brick_rejection_reasons.csv — rejection counts broken down by reason
regeneration_stats.csv — number of regenerations per prompt
Visualizing results
To visualize the saved statistics, run:

python proj_develop/eval_stats.py
t-structure pair.

### Output files

After inference completes, three CSV files will be written to:

```
proj_develop/batch_outputs/<experiment_name>/
```

These include:

- `brick_rejections_stats.csv` — total rejection count per prompt  
- `brick_rejection_reasons.csv` — rejection counts broken down by reason  
- `regeneration_stats.csv` — number of regenerations per prompt

### Visualizing results

To visualize the saved statistics, run:

```zsh
python proj_develop/eval_stats.py
```

This script loads the CSVs from the experiment folder and produces summary plots (histograms, reason distributions, regeneration counts).


## Sreeharsha important commands:
```zsh
python proj_develop/eval_stats.py --output_dir_prefix proj_develop/datasets/dpo_datasets/combined_dataset
```

* For AWS:
```bash

to active the virtual environment:

```zsh
source .venv-eval/bin/activate

# Convert csv to parquet:
```zsh
uv run python proj_develop/convert_dpo_csv_to_parquet.py
```


```

## Kshitij: How to run the dpo data collection?
It is the same as the batch infer script.

The args are experiment name prefix, start_idx and the number of rows you want to run it for.

Example:
```
uv run batch_infer --dataset /home/vader/Music/ckpts/10623-ConditionedBrickGPT/proj_develop/datasets/4class_stablebrick2text_train.parquet --experiment_name train_400_599 --start_idx 400 --max_rows 200
```