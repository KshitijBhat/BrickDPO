"""
Script to prepare a DPO (Direct Preference Optimization) dataset in HuggingFace format.
Fixed to output proper JSONL format for TRL DPOTrainer.
"""
import argparse
import pandas as pd
import os
from brickgpt.models import create_instruction

def main():
    # ------------------------------
    # Add argparse CLI interface
    # ------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Optional maximum sequence length for truncation (default: None)"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="datasets/dpo_datasets/combined_test_dataset/dpo_data.parquet",
        help="Path to input parquet file",
        required=True
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="datasets/dpo_datasets/combined_test_dataset/dpo_hf",
        help="Path to output JSONL file",
        required=True
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    max_seq_len = args.max_seq_len

    print(f"max_seq_len = {max_seq_len}")

    output_file_suffix = "" if max_seq_len is None else f"_maxlen{max_seq_len}"
    output_file = f"{output_file}{output_file_suffix}.jsonl"

    # ------------------------------
    # Start parsing 
    # ------------------------------
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    dpo_data = []

    for _, row in df.iterrows():
        caption = row.get('caption', '')
        bricks_candidates = row.get('generated_bricks', [])


        # The last entry of the array is the chosen response
        chosen_content = bricks_candidates[-1]

        if max_seq_len is not None and len(chosen_content) > max_seq_len:
            continue
        
        # All the entries df['generated_bricks'][0:-1] are the rejected ones
        rejected_candidates = bricks_candidates[:-1]

        # Pair each rejected candidate with the single chosen response
        for rejected_content in rejected_candidates:
            
            # Construct messages in the chat format
            # Prompt contains the conversation history (system + user)
            prompt_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": create_instruction(caption)}
            ]
            
            # Chosen/rejected contain only the assistant response
            chosen_messages = [
                {"role": "assistant", "content": chosen_content}
            ]
            
            rejected_messages = [
                {"role": "assistant", "content": rejected_content}
            ]

            entry = {
                "prompt": prompt_messages,
                "chosen": chosen_messages,
                "rejected": rejected_messages
            }
            
            dpo_data.append(entry)

    print(f"Created {len(dpo_data)} DPO training pairs")
    
    # Convert to DataFrame
    dataset = pd.DataFrame(dpo_data)
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save as JSONL (orient='records', lines=True)
    # force_ascii=False ensures special characters are written as-is, not escaped
    dataset.to_json(output_file, orient='records', lines=True, force_ascii=False)
    
    print(f"Successfully processed data. Saved {len(dataset)} DPO entries to {output_file}")
    
    # Verify the format
    print("\nVerifying format...")
    try:
        # Read back with lines=True
        test_df = pd.read_json(output_file, lines=True)
        print("Verification successful: File can be read as JSONL.")
        print(f"Sample entry:\n{test_df.iloc[0]}")
        print(f"Sample entry dict:\n{test_df.iloc[0].to_dict()}")
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    main()