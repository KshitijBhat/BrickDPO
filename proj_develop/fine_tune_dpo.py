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

@dataclass
class ScriptArguments:
    """
    Arguments for the DPO training script.
    """
    # Model arguments
    model_name_or_path: Optional[str] = field(
        default="AvaLovelace/BrickGPT", 
        metadata={"help": "Path to the SFT checkpoint (your input model)"}
    )
    dataset_path: Optional[str] = field(
        default="dpo_dataset.parquet", 
        metadata={"help": "Path to the DPO dataset parquet file"}
    )
    output_dir: Optional[str] = field(
        default="dpo_output", 
        metadata={"help": "Where to save the model"}
    )
    
    # LoRA arguments (Matched to your finetune.zsh)
    use_peft: bool = field(default=True, metadata={"help": "Whether to use PEFT/LoRA"})
    lora_r: int = field(default=32, metadata={"help": "LoRA R value"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA Alpha value"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA Dropout"})
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"], 
        metadata={"help": "Target modules for LoRA"}
    )

    # DPO Specifics
    beta: float = field(default=0.1, metadata={"help": "The beta parameter for DPO loss"})
    max_length: int = field(default=8192, metadata={"help": "Max sequence length (matches your SFT)"})
    max_prompt_length: int = field(default=4096, metadata={"help": "Max prompt length"})


def main():
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, dpo_args = parser.parse_args_into_dataclasses()

    # 1. Load Dataset
    # The trainer expects columns: 'prompt', 'chosen', 'rejected'
    print(f"Loading dataset from {script_args.dataset_path}...")
    dataset = load_dataset("parquet", data_files={"train": script_args.dataset_path})
    
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
        attn_implementation="flash_attention_2",
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
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=script_args.beta,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
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