#import subprocess
#import sys
#
#def install(package):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Hack!
#install("torchdata")

#from importlib import metadata
#print("[DEBUG]")
#for dist in metadata.distributions():
#    print(f"{dist.name}=={dist.version}")
#print("====")

import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
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
import argparse
import tempfile
import shutil

# Import wandb
import wandb

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
    "batch_size": 8,  # Adjust based on memory
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 50,
    "max_steps": 21000,
    "save_steps": 4000,
    "log_steps": 50,
    "output_dir": "./checkpoints", # Local temp dir for saving before upload
    "gradient_accumulation_steps": 2,
    "bf16": True,  # Use bfloat16 for H100
    "wandb_project": "llm-pretraining", # W&B Project Name
    "wandb_entity": "lemontea-tom", # Set your W&B entity (team/username) if needed
    "run_name": None, # Optional run name for W&B
    "resume_from_artifact": None, # W&B Artifact path to resume from (e.g., "entity/project/checkpoint-name:v0")
}

WANDB_API_KEY = "<provide your key here, secret from env var>"
SKIP_DATA_ROWS = 3000000 # 3 million

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
    # Ensure pad token is set for DataCollatorForLanguageModeling
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_dataset(tokenizer):
    """Setup streaming dataset with preprocessing"""
    # issue 4804: streaming dataset with concatenating splits raises an error, (known limitation)
    dataset = load_dataset(
        TRAINING_CONFIG["dataset_name"],
        name=TRAINING_CONFIG["dataset_config"],
        split="train",
        streaming=True
    )
    #dataset = dataset.skip(SKIP_DATA_ROWS) # dataset lib limitation: .skip is inefficient (doesn't skip shard)

    def tokenize_function(example):
        # Use padding=False for efficiency with group_texts
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
            k: [t[i: i + TRAINING_CONFIG["sequence_length"]]
                for i in range(0, total_length, TRAINING_CONFIG["sequence_length"])]
            for k, t in concatenated_examples.items()
        }
        # Add labels for causal LM (shifted inputs)
        result["labels"] = result["input_ids"].copy()
        return result

    grouped_dataset = tokenized_dataset.map(
        group_texts,
        batched=True
    )

    return grouped_dataset

def setup_dataloader(dataset):
    """Create DataLoader with appropriate collator"""
    # Re-initialize tokenizer here to avoid pickling issues if needed later
    tokenizer = setup_tokenizer()
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, # Causal LM
    )

    # TODO: beta level feature
    dataloader = StatefulDataLoader(
        dataset,
        collate_fn=data_collator,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False, # Streaming dataset handles shuffling
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

def save_model_artifact(run, model, tokenizer, step, optimizer=None, scheduler=None, dataloader=None):
    """Save model checkpoint locally and upload as W&B artifact."""
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=TRAINING_CONFIG["output_dir"])
    checkpoint_path = os.path.join(temp_dir, f"checkpoint-{step}")

    try:
        # Save model, tokenizer, and training states
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        if optimizer:
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        if scheduler:
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))
        # Save training step
        with open(os.path.join(checkpoint_path, "training_state.json"), 'w') as f:
            import json
            json.dump({"step": step}, f)
        # Save dataloader/dataset state (must use is not None due to implicitly being an iterable implies attempt the len() which it doesn't have)
        if dataloader is not None:
            print("[DEBUG] Saving dataloader state")
            print(dataloader)
            torch.save(dataloader.state_dict(), os.path.join(checkpoint_path, "dataloader.pt"))

        # Create W&B Artifact
        artifact_name = f"checkpoint-{step}"
        artifact = wandb.Artifact(name=artifact_name, type='model')
        artifact.add_dir(checkpoint_path) # Add entire checkpoint directory

        # Log the artifact to W&B
        run.log_artifact(artifact)
        print(f"Checkpoint saved and artifact '{artifact_name}' uploaded at step {step}")

    finally:
        # Clean up local temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def load_checkpoint_from_artifact(run, artifact_path, device):
    """Load model, tokenizer, and training states from a W&B artifact."""
    print(f"Resuming training from artifact: {artifact_path}")
    # Use W&B API to download the artifact
    #api = wandb.Api(api_key=WANDB_API_KEY)
    #artifact = api.artifact(artifact_path)
    # Update: Use from the run object to enable lineage tracking
    artifact = run.use_artifact(artifact_path, type='model')

    # Download artifact to a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        artifact_dir = artifact.download(root=temp_dir)
        print(f"Artifact downloaded to: {artifact_dir}")

        # Load model and tokenizer
        #model = setup_model() # Initialize model architecture
        #model.load_state_dict(torch.load(os.path.join(artifact_dir, "pytorch_model.bin"), map_location=device))
        #model.to(device)
        # Update: use huggingface transformer + safetensor ecosystem directly
        # TODO: currently use config from checkpoint. Provide option to override with current run config?
        model = AutoModelForCausalLM.from_pretrained(artifact_dir, local_files_only=True)
        model.to(device)
        tokenizer = setup_tokenizer()
        # Note: Tokenizer state is usually static, but loading from artifact ensures consistency
        # tokenizer = AutoTokenizer.from_pretrained(artifact_dir)

        # Load training states if they exist
        start_step = 0
        optimizer_state = None
        scheduler_state = None

        if os.path.exists(os.path.join(artifact_dir, "optimizer.pt")):
            optimizer_state = torch.load(os.path.join(artifact_dir, "optimizer.pt"), map_location=device)
        if os.path.exists(os.path.join(artifact_dir, "scheduler.pt")):
            scheduler_state = torch.load(os.path.join(artifact_dir, "scheduler.pt"), map_location=device)
        if os.path.exists(os.path.join(artifact_dir, "training_state.json")):
            import json
            with open(os.path.join(artifact_dir, "training_state.json"), 'r') as f:
                training_state = json.load(f)
                start_step = training_state.get("step", 0)
        # TODO: beta feature, load dataloader/dataset state
        dataloader_state = None
        if os.path.exists(os.path.join(artifact_dir, "dataloader.pt")):
            dataloader_state = torch.load(os.path.join(artifact_dir, "dataloader.pt"))

        print(f"Checkpoint loaded. Resuming from step {start_step}")
        return model, tokenizer, start_step, optimizer_state, scheduler_state, dataloader_state

    finally:
        # Clean up downloaded artifact directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def train():
    """Main training loop"""
    # Initialize W&B
    run = wandb.init(
        project=TRAINING_CONFIG["wandb_project"],
        entity=TRAINING_CONFIG["wandb_entity"],
        name=TRAINING_CONFIG["run_name"],
        config=TRAINING_CONFIG # Log hyperparameters
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_step = 0
    model = None
    tokenizer = None
    optimizer_state = None
    scheduler_state = None

    # Check for resumption
    if TRAINING_CONFIG["resume_from_artifact"]:
        model, tokenizer, start_step, optimizer_state, scheduler_state, dataloader_state = \
            load_checkpoint_from_artifact(run, TRAINING_CONFIG["resume_from_artifact"], device)
        if TRAINING_CONFIG["bf16"] and model.dtype != torch.bfloat16:
             model = model.bfloat16()
    else:
        # Standard initialization
        tokenizer = setup_tokenizer()
        model = setup_model().to(device)
        if TRAINING_CONFIG["bf16"]:
            model = model.bfloat16()

    # Setup data, optimizer, scheduler
    dataset = setup_dataset(tokenizer)
    dataloader = setup_dataloader(dataset)
    if dataloader_state:
        dataloader.load_state_dict(dataloader_state)
    optimizer = setup_optimizer(model)
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAINING_CONFIG["warmup_steps"],
        num_training_steps=TRAINING_CONFIG["max_steps"]
    )
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    # Training loop
    model.train()
    accumulated_loss = 0
    optimizer.zero_grad()

    data_iter = iter(dataloader)

    # Use tqdm starting from the resumed step
    pbar = tqdm(range(start_step + 1, TRAINING_CONFIG["max_steps"] + 1), desc="Training", initial=start_step, total=TRAINING_CONFIG["max_steps"])

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reinitialize iterator when dataset ends (for streaming)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / TRAINING_CONFIG["gradient_accumulation_steps"]

        # Backward pass
        loss.backward()

        accumulated_loss += loss.item() * TRAINING_CONFIG["gradient_accumulation_steps"] # Adjust for accumulation

        # Gradient accumulation
        if step % TRAINING_CONFIG["gradient_accumulation_steps"] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % TRAINING_CONFIG["log_steps"] == 0:
            avg_loss = accumulated_loss / TRAINING_CONFIG["log_steps"]
            current_lr = scheduler.get_last_lr()[0]
            # Log metrics to W&B
            run.log({
                "train/loss": avg_loss,
                "train/learning_rate": current_lr,
                "train/step": step
            }, step=step)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "LR": f"{current_lr:.2e}"})
            accumulated_loss = 0

        # Save checkpoint as W&B Artifact
        if (step - start_step) % TRAINING_CONFIG["save_steps"] == 0:
            save_model_artifact(run, model, tokenizer, step, optimizer, scheduler, dataloader)

    # Final save
    if (TRAINING_CONFIG["max_steps"] - start_step) % TRAINING_CONFIG["save_steps"] != 0:
        save_model_artifact(run, model, tokenizer, TRAINING_CONFIG["max_steps"], optimizer, scheduler, dataloader)

    print("Training completed.")
    run.finish()

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Train LLM with W&B integration")
    #parser.add_argument("--resume_from_artifact", type=str, help="W&B artifact path to resume from (e.g., entity/project/artifact_name:v1)")
    #parser.add_argument("--run_name", type=str, help="Optional name for the W&B run")
    #args = parser.parse_args()
    #args.resume_from_artifact = None
    #args.run_name = 'initial_run'
    args = { 'run_name': 'continue_train_001', 'resume_from_artifact': 'lemontea-tom/llm-pretraining/checkpoint-13000:v0' }

    # Update config with command line arguments if provided
    if args['resume_from_artifact']:
        TRAINING_CONFIG["resume_from_artifact"] = args['resume_from_artifact']
    if args['run_name']:
        TRAINING_CONFIG["run_name"] = args['run_name']

    wandb.login(key=WANDB_API_KEY, verify=True)
    try:
        train()
    except Exception:
        print(traceback.format_exc())
