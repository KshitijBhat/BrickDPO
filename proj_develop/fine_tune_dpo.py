import os
import torch
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import DPOConfig, DPOTrainer

# @dataclass
class ScriptArguments:
    """
    Arguments for the DPO training script.
    """
    # Model arguments
    model_name_or_path: str = "sft_brickgpt"  # Path to SFT meta-llama/Llama-3.2-1B-Instruct model
    dataset_path: str = "datasets/dpo_datasets/combined_dataset/dpo_hf.parquet"
    output_dir: str = "dpo_output"
    
    # LoRA arguments (Matched to your finetune.zsh)
    use_peft: bool = True
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = ["q_proj", "v_proj"]

    # DPO Specifics
    beta: float = 0.1
    max_length: int = 8192
    max_prompt_length: int = 4096


def main():
    # Use hardcoded arguments instead of command line parsing
    script_args = ScriptArguments()
    
    # Configure DPO training arguments
    dpo_args = DPOConfig(
        output_dir=script_args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        report_to="none",
        beta=script_args.beta,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
    )

    # 1. Load Dataset
    # The trainer expects columns: 'prompt', 'chosen', 'rejected'
    print(f"Loading dataset from {script_args.dataset_path}...")
    train_dataset = load_dataset("parquet", data_files=script_args.dataset_path, split="train")
    
    # 2. Load Tokenizer
    # We load this first to handle special tokens before loading the model
    print(f"Loading tokenizer from {script_args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    
    # Llama 3 specific pad token handling based on your config files
    # Your tokenizer_config.json shows pad_token_id is 128004 (<|finetune_right_pad_id|>)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback if specific token isn't set, though your config suggests it exists
            tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
            
    # Ensure padding side is correct for DPO (usually left padding is safer for generation)
    # However, for pure training, right padding is often standard. 
    # TRL handles this, but explicit setting is good practice.
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right"

    # 3. Load Model
    print(f"Loading model from {script_args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16, # Matching your SFT script
        attn_implementation="flash_attention_2", # Llama 3 optimization
        device_map="auto"
    )

    # 4. LoRA Configuration
    peft_config = None
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=script_args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # 5. Initialize DPO Trainer
    # Note: When using PEFT, we do NOT need to load a ref_model explicitly.
    # The trainer creates a reference from the frozen base model adapter.
    trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. Train
    print("Starting DPO training...")
    trainer.train()
    
    # 7. Save
    print(f"Saving model to {script_args.output_dir}")
    trainer.save_model(script_args.output_dir)
    # Save tokenizer as well to ensure special tokens map is preserved
    tokenizer.save_pretrained(script_args.output_dir) 

if __name__ == "__main__":
    main()