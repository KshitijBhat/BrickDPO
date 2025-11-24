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

Each structure in the dataset includes **five prompts** (stored in the `captions` column). BrickGPT generates one output per prompt and logs statistics for each prompt-structure pair.

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
