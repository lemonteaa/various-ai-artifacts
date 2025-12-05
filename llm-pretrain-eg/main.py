import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import math
from tqdm import tqdm
import os

# Configuration
MODEL_CONFIG = {
    "architectures": ["Ernie4_5_ForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 131072,
    "model_type": "ernie4_5",
    "num_attention_heads": 16,
    "num_hidden_layers": 18,
    "num_key_value_heads": 2,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 500000.0,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.54.0.dev0",
    "use_bias": False,
    "use_cache": True,
    "vocab_size": 103424
}

TRAINING_CONFIG = {
    "dataset_name": "HuggingFaceFW/fineweb",
    "dataset_config": "sample-10BT",  # Small sample for iteration
    "tokenizer_name": "baidu/ERNIE-4.5-0.3B-PT",  # Compatible tokenizer
    "sequence_length": 2048,
    "batch_size": 4,  # Adjust based on memory
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 50,
    "max_steps": 1000,
    "save_steps": 500,
    "log_steps": 10,
    "output_dir": "./checkpoints",
    "gradient_accumulation_steps": 2,
    "bf16": True,  # Use bfloat16 for H100
}

def setup_model():
    """Initialize the model with specified configuration"""
    config = AutoConfig.for_model(**MODEL_CONFIG)
    model = AutoModelForCausalLM.from_config(config)
    
    # Initialize weights (simplified)
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=MODEL_CONFIG["initializer_range"])
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    model.apply(init_weights)
    return model

def setup_tokenizer():
    """Setup tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        TRAINING_CONFIG["tokenizer_name"],
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_dataset(tokenizer):
    """Setup streaming dataset with preprocessing"""
    dataset = load_dataset(
        TRAINING_CONFIG["dataset_name"],
        name=TRAINING_CONFIG["dataset_config"],
        split="train",
        streaming=True
    )
    
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding=False,
            max_length=TRAINING_CONFIG["sequence_length"]
        )
    
    # Tokenize and group texts
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # We drop the small remainder
        total_length = (total_length // TRAINING_CONFIG["sequence_length"]) * TRAINING_CONFIG["sequence_length"]
        
        # Split by chunks of max_seq_length
        result = {
            k: [t[i : i + TRAINING_CONFIG["sequence_length"]] 
                for i in range(0, total_length, TRAINING_CONFIG["sequence_length"])]
            for k, t in concatenated_examples.items()
        }
        return result
    
    grouped_dataset = tokenized_dataset.map(
        group_texts,
        batched=True
    )
    
    return grouped_dataset

def setup_dataloader(dataset):
    """Create DataLoader with appropriate collator"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=setup_tokenizer(),
        mlm=False,
    )
    
    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
    )
    return dataloader

def setup_optimizer(model):
    """Setup AdamW optimizer with weight decay"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": TRAINING_CONFIG["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=TRAINING_CONFIG["learning_rate"]
    )
    return optimizer

def train():
    """Main training loop"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = setup_tokenizer()
    model = setup_model().to(device)
    
    if TRAINING_CONFIG["bf16"]:
        model = model.bfloat16()
    
    dataset = setup_dataset(tokenizer)
    dataloader = setup_dataloader(dataset)
    optimizer = setup_optimizer(model)
    
    # Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAINING_CONFIG["warmup_steps"],
        num_training_steps=TRAINING_CONFIG["max_steps"]
    )
    
    # Training loop
    model.train()
    accumulated_loss = 0
    optimizer.zero_grad()
    
    data_iter = iter(dataloader)
    
    for step in tqdm(range(1, TRAINING_CONFIG["max_steps"] + 1), desc="Training"):
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reinitialize iterator when dataset ends
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / TRAINING_CONFIG["gradient_accumulation_steps"]
        
        # Backward pass
        loss.backward()
        
        accumulated_loss += loss.item()
        
        # Gradient accumulation
        if step % TRAINING_CONFIG["gradient_accumulation_steps"] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if step % TRAINING_CONFIG["log_steps"] == 0:
            avg_loss = accumulated_loss / TRAINING_CONFIG["log_steps"]
            print(f"Step {step}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
            accumulated_loss = 0
        
        # Save checkpoint
        if step % TRAINING_CONFIG["save_steps"] == 0:
            os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
            checkpoint_path = os.path.join(TRAINING_CONFIG["output_dir"], f"checkpoint-{step}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"Checkpoint saved at step {step}")

if __name__ == "__main__":
    train()
