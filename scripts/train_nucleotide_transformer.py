from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, AutoConfig
)
import numpy as np
import torch
import os
from accelerate import Accelerator
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback


DATASET_PATH = "./datasets/nucleotide_transformer_downstream_tasks_revised"
MODEL_PATH = "./models/InstaDeepAI/nucleotide-transformer-2.5b-1000g"
MODEL_ID = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"  # Fallback
QUANTIZATION_TYPE = "none"  # Can be "4bit", "8bit", or "none"
BASE_SAVE_PATH = f"./trained_models/"
TRAINING_TASK = [] # Leave empty to train all tasks, or ["promoter_all", "enhancer_all"] for the specific tasks you want to train 


def load_model(model_id, quantization_type="4bit", trained_model_path=None, local_model_path=None, reinitialize_classifier=True):
    """
    Load the nucleotide transformer model with optional quantization.
    
    Args:
        model_id (str): HuggingFace model identifier (fallback)
        quantization_type (str): Either "4bit", "8bit", or "none"
        trained_model_path (str): Path to trained model weights (optional)
        local_model_path (str): Path to local model directory (optional)
        reinitialize_classifier (bool): Whether to reinitialize classifier weights (default: True)
    
    Returns:
        torch.nn.Module: Model (quantized or full precision)
    """
    # Check if we should load a trained model
    if trained_model_path and os.path.exists(trained_model_path):
        print(f"Loading trained model from {trained_model_path}...")
        # Load trained model configuration
        config = AutoConfig.from_pretrained(trained_model_path)
        
        if quantization_type == "none":
            print(f"Loading trained model without quantization (full precision)...")
            # IMPORTANT: Avoid device_map="auto" for training to prevent tensor sharding across GPUs
            model = AutoModelForSequenceClassification.from_pretrained(
                trained_model_path,
                config=config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            print(f"Loading trained model with {quantization_type} quantization...")
            empty_model = AutoModelForSequenceClassification.from_config(config)
            
            # Initialize classifier weights for newly created model
            if reinitialize_classifier:
                initialize_classifier_weights(empty_model)
            
            print(f"Creating {quantization_type} quantization configuration...")
            bnb_config = create_quantization_config(quantization_type)
            
            print(f"Loading and quantizing trained model with {quantization_type} precision...")
            model = load_and_quantize_model(
                empty_model, 
                weights_location=trained_model_path, 
                bnb_quantization_config=bnb_config
            )
    else:
        # Determine which model to load: local first, then HuggingFace
        model_path = None
        if local_model_path and os.path.exists(local_model_path):
            model_path = local_model_path
            print(f"Loading model from local directory: {model_path}")
        else:
            model_path = model_id
            print(f"Local model not found, loading from Hugging Face: {model_path}")
            if local_model_path:
                print(f"Consider running download_nucleotide_transformer.py to save model locally at {local_model_path}")
        
        # Load model configuration
        print(f"Loading model configuration...")
        config = AutoConfig.from_pretrained(model_path)
        
        if quantization_type == "none":
            print(f"Loading model without quantization (full precision)...")
            # IMPORTANT: Avoid device_map="auto" for training to prevent tensor sharding across GPUs
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Initialize classifier weights for newly loaded model
            # This ensures proper initialization even when loading from pretrained weights
            # if the classifier was not properly initialized in the original model
            if reinitialize_classifier:
                initialize_classifier_weights(model)
        else:
            if model_path == model_id:
                # Download from HuggingFace
                print(f"Downloading model weights for {model_id}...")
                weights_location = snapshot_download(repo_id=model_id)
            else:
                # Use local model
                weights_location = model_path
                print(f"Using local model weights from {weights_location}")
            
            empty_model = AutoModelForSequenceClassification.from_config(config)
            
            # Initialize classifier weights for newly created model
            if reinitialize_classifier:
                initialize_classifier_weights(empty_model)
            
            print(f"Creating {quantization_type} quantization configuration...")
            bnb_config = create_quantization_config(quantization_type)
            
            print(f"Loading and quantizing model with {quantization_type} precision...")
            model = load_and_quantize_model(
                empty_model, 
                weights_location=weights_location, 
                bnb_quantization_config=bnb_config
            )
    
    return model

class TensorBoardCallback(TrainerCallback):
    """Custom callback to log additional metrics to TensorBoard"""
    
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, state.global_step)
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key.startswith("eval_"):
                    self.writer.add_scalar(f"eval/{key[5:]}", value, state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()


class GradientClippingCallback(TrainerCallback):
    """Custom callback to apply additional gradient clipping and monitoring"""
    
    def __init__(self, max_grad_norm=1.0, clip_threshold=10.0):
        self.max_grad_norm = max_grad_norm
        self.clip_threshold = clip_threshold
        self.gradient_norms = []
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Apply gradient clipping after each step"""
        if model is not None:
            # Get all gradients
            total_norm = 0.0
            param_count = 0
            has_nan = False
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Check for NaN gradients first
                    if torch.isnan(param.grad).any():
                        print(f"⚠️  WARNING: NaN gradient detected in {name}")
                        has_nan = True
                        # Zero out NaN gradients
                        param.grad.data = torch.zeros_like(param.grad.data)
                        continue
                    
                    # Calculate norm safely
                    param_norm = param.grad.data.norm(2)
                    if not torch.isnan(param_norm) and not torch.isinf(param_norm):
                        total_norm += param_norm.item() ** 2
                        param_count += 1
            
            if param_count > 0 and not has_nan:
                total_norm = total_norm ** (1. / 2)
                self.gradient_norms.append(total_norm)
                
                # Apply gradient clipping if norm is too large
                if total_norm > self.clip_threshold:
                    print(f"⚠️  WARNING: Large gradient norm detected: {total_norm:.2f}, applying clipping...")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                
                # Log gradient norm every 10 steps
                if state.global_step % 10 == 0:
                    print(f"Step {state.global_step}: Gradient norm = {total_norm:.4f}")
            elif has_nan:
                print(f"⚠️  WARNING: NaN gradients detected, skipping gradient norm calculation")
                self.gradient_norms.append(float('nan'))
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add gradient norm to logs"""
        if logs and self.gradient_norms:
            recent_norm = self.gradient_norms[-1] if self.gradient_norms else 0.0
            logs['custom_grad_norm'] = recent_norm

def freeze_backbone_parameters(model, freeze_classifier=False):
    """
    Freeze all backbone parameters and only allow classifier to be trained.
    
    Args:
        model: The model to freeze
        freeze_classifier: If True, also freeze the classifier (useful for inference)
    
    Returns:
        tuple: (frozen_params, trainable_params) - counts of frozen and trainable parameters
    """
    frozen_params = 0
    trainable_params = 0
    
    print("Freezing backbone parameters...")
    
    for name, param in model.named_parameters():
        # Freeze all parameters except the classifier
        if 'classifier' in name:
            if not freeze_classifier:
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"  ✓ Trainable: {name} ({param.numel():,} parameters)")
            else:
                param.requires_grad = False
                frozen_params += param.numel()
                print(f"  ✗ Frozen: {name} ({param.numel():,} parameters)")
        else:
            # Freeze backbone parameters
            param.requires_grad = False
            frozen_params += param.numel()
            if frozen_params <= 10:  # Only print first few for brevity
                print(f"  ✗ Frozen: {name} ({param.numel():,} parameters)")
    
    print(f"\nParameter Summary:")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {frozen_params + trainable_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / (frozen_params + trainable_params):.2f}%")
    
    return frozen_params, trainable_params

"""
Multi-GPU Training Support for Nucleotide Transformer

This script automatically detects available GPUs and configures training accordingly:
- Single GPU: Uses standard Hugging Face Trainer
- Multiple GPUs: Uses Accelerate library for distributed training
- CPU: Falls back to CPU training

The script now supports classifier-only training by freezing the backbone model.

Usage:
1. Single GPU: python train_nucleotide_transformer.py
2. Multi-GPU: 
   - First time: accelerate config
   - Then: accelerate launch train_nucleotide_transformer.py
   - Or use the generated launch script: ./launch_multi_gpu.sh

The script will automatically:
- Detect the number of available GPUs
- Configure appropriate batch sizes and gradient accumulation
- Set up distributed training if multiple GPUs are available
- Freeze backbone parameters and only train the classifier
- Create a launch script for easy multi-GPU training
"""



# Check for available GPUs and configure multi-GPU training
def setup_multi_gpu():
    """Setup multi-GPU training if multiple GPUs are available."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s) available")
        
        if num_gpus > 1:
            print(f"Setting up multi-GPU training with {num_gpus} GPUs")
            # Initialize accelerator for multi-GPU training
            accelerator = Accelerator()
            return accelerator, True
        else:
            print("Single GPU detected, using standard training")
            return None, False
    else:
        print("No CUDA GPUs available, using CPU training")
        return None, False

# Setup multi-GPU training
accelerator, use_multi_gpu = setup_multi_gpu()

# Load dataset from local directory

if os.path.exists(DATASET_PATH):
    print(f"Loading dataset from local directory: {DATASET_PATH}")
    # Load from individual arrow files to preserve structure
    train_data = load_dataset('arrow', data_files=f'{DATASET_PATH}/train/data-00000-of-00001.arrow')['train']
    test_data = load_dataset('arrow', data_files=f'{DATASET_PATH}/test/data-00000-of-00001.arrow')['train']
    raw = {'train': train_data, 'test': test_data}
else:
    print(f"Local dataset not found at {DATASET_PATH}, downloading from Hugging Face...")
    raw = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    print("Dataset downloaded. Consider running download_nucleotide_transformer.py to save it locally for future use.")

def train_single_task(task_name: str):
    print(f"\n===== Training task: {task_name} =====")
    
    # Filter for the specific task (use only the original train split)
    full_train = raw["train"].filter(lambda ex: ex["task"] == task_name)
    
    # Split into train (80%) and validation (20%). Try to stratify by label if supported
    try:
        split = full_train.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    except Exception:
        split = full_train.train_test_split(test_size=0.2, seed=42)
    
    train = split["train"]
    val   = split["test"]
    
    num_labels = len(set(train["label"]))
    id2label = {i: f"CLASS_{i}" for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}
    
    # Model configuration - try local first, fallback to Hugging Face
    
    # Load tokenizer from local directory if available
    if os.path.exists(MODEL_PATH):
        print(f"Loading tokenizer from local directory: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    else:
        print(f"Local model not found at {MODEL_PATH}, loading from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print("Model loaded from Hugging Face. Consider running download_nucleotide_transformer.py to save it locally for future use.")
    
    # Find the maximum sequence length in the dataset for this task
    print("Finding maximum sequence length in the dataset...")
    train_max_length = max(len(seq) for seq in train["sequence"])
    print(f"Maximum sequence length in training data: {train_max_length}")
    
    # Also check validation data
    val_max_length = max(len(seq) for seq in val["sequence"])
    print(f"Maximum sequence length in validation data: {val_max_length}")
    
    # Use the maximum of both
    actual_max_length = max(train_max_length, val_max_length)
    print(f"Using max_length: {actual_max_length}")
    
    def tokenize(batch):
        return tokenizer(
            batch["sequence"],
            padding=False,  # no padding needed - all sequences same length
            truncation=True,
            max_length=actual_max_length,  # use actual max length from dataset
            return_attention_mask=True,
        )
    
    print("model max length: ", tokenizer.model_max_length)
    
    train_tok = train.map(tokenize, batched=True, remove_columns=[c for c in train.column_names if c not in {"label"}])
    val_tok   = val.map(tokenize,   batched=True, remove_columns=[c for c in val.column_names   if c not in {"label"}])
    
    # Simple data collator - no padding needed since all sequences have same length
    class SimpleDataCollator:
        def __call__(self, features):
            batch = {
                "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
                "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
                "labels": torch.tensor([f["label"] for f in features], dtype=torch.long)
            }
            return batch
    
    data_collator = SimpleDataCollator()

    # Load model with quantization support
    print(f"Loading model with {QUANTIZATION_TYPE} quantization...")
    # do not use quantization during training
    # Pass local model path to load_model function
    # Load model - only reinitialize if loading from config, not from pretrained weights
    model = load_model(MODEL_ID, "none", local_model_path=MODEL_PATH, reinitialize_classifier=False)

    # Configure the model for classification task
    model.config.num_labels = num_labels
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.problem_type = "single_label_classification"

    if torch.cuda.is_available():
        # Move model to GPU
        model = model.to('cuda')
        print(f"Final model device: {next(model.parameters()).device}")

    # print model's classifier
    print(f"Model classifier: {model.classifier}")

    # Ensure classifier dtype matches backbone (avoids Half vs Float mismatch)
    backbone_dtype = next(model.parameters()).dtype
    model.classifier = model.classifier.to(dtype=backbone_dtype)
    
    # FREEZE BACKBONE PARAMETERS - Only train the classifier
    print("\n" + "="*60)
    print("FREEZING BACKBONE PARAMETERS - CLASSIFIER-ONLY TRAINING")
    print("="*60)
    frozen_params, trainable_params = freeze_backbone_parameters(model, freeze_classifier=False)
    
    # Verify that only classifier parameters are trainable
    print("\nVerifying parameter freezing:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  ✓ Trainable: {name}")
        else:
            if 'classifier' not in name:  # Don't print all frozen backbone params
                continue
            print(f"  ✗ Frozen: {name}")
    
    # Note: Classifier weights are now properly initialized in load_bnb_nucleotide_transformer.py

    # (Optional) if you have limited memory:
    # model.gradient_checkpointing_enable()
    # model.config.use_cache = False

    # Use sklearn metrics instead of evaluate library to avoid loading issues
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        
        # Check for NaN values in logits
        if np.isnan(logits).any():
            print("⚠️  WARNING: NaN values detected in evaluation logits!")
            return {"accuracy": 0.0, "f1_macro": 0.0}
        
        if model.config.problem_type == "regression":
            # MAE example for regression:
            preds = logits.squeeze(-1)
            mae = np.mean(np.abs(preds - labels))
            return {"mae": mae}
        else:
            preds = np.argmax(logits, axis=-1)
            out = {"accuracy": accuracy_score(labels, preds)}
            # Only compute F1 when classes > 1
            if num_labels > 1:
                out["f1_macro"] = f1_score(labels, preds, average="macro")
            return out

    args = TrainingArguments(
        output_dir=f"{BASE_SAVE_PATH}/{task_name}",
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        learning_rate=1e-4,
        warmup_steps=50,
        num_train_epochs=10,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",  # Add cosine learning rate scheduling
    )

    # Create callbacks
    tensorboard_callback = TensorBoardCallback(f"./tensorboard_logs/{task_name}")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[tensorboard_callback],
    )

    # Add some debugging to check batch dimensions
    print("Testing data collator with a small batch...")
    sample_batch = data_collator([train_tok[i] for i in range(2)])
    print(f"Sample batch input_ids shape: {sample_batch['input_ids'].shape}")
    print(f"Sample batch attention_mask shape: {sample_batch['attention_mask'].shape}")
    print(f"Sample batch labels shape: {sample_batch['labels'].shape}")
    
    # Verify all sequences in batch have same length (expected since no padding)
    input_ids = sample_batch['input_ids']
    attention_mask = sample_batch['attention_mask']
    print(f"All input_ids have same length: {all(len(seq) == len(input_ids[0]) for seq in input_ids)}")
    print(f"All attention_masks have same length: {all(len(seq) == len(attention_mask[0]) for seq in attention_mask)}")
    print(f"Input_ids and attention_mask shapes match: {input_ids.shape == attention_mask.shape}")
    print(f"Sequence length: {len(input_ids[0])} (should be {actual_max_length})")

    # Print training configuration
    print(f"\nTraining Configuration:")
    print(f"Multi-GPU training: {use_multi_gpu}")
    print(f"Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"Per-device batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * (torch.cuda.device_count() if torch.cuda.is_available() else 1)}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Training epochs: {args.num_train_epochs}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Start training
    print("\nStarting classifier-only training...")
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal evaluation metrics: {metrics}")

    # Save the trained model
    print(f"Saving trained model to {f"{BASE_SAVE_PATH}/{task_name}"}...")
    os.makedirs(f"{BASE_SAVE_PATH}/{task_name}", exist_ok=True)

    # Save model and tokenizer
    trainer.save_model(f"{BASE_SAVE_PATH}/{task_name}")
    tokenizer.save_pretrained(f"{BASE_SAVE_PATH}/{task_name}")

    # Also save the model configuration
    model.config.save_pretrained(f"{BASE_SAVE_PATH}/{task_name}")

    print(f"Model saved successfully to {f"{BASE_SAVE_PATH}/{task_name}"}")
    print("You can now load this trained model using load_bnb_nucleotide_transformer.py with --trained-model-path")
    
    print(f"\n📊 TensorBoard logs saved to: ./tensorboard_logs/{task_name}")
    print("To view training progress, run:")
    print(f"  tensorboard --logdir=./tensorboard_logs/{task_name}")
    print("Then open http://localhost:6006 in your browser")

# Create a launch script for multi-GPU training
def create_launch_script():
    """Create a launch script for multi-GPU training using accelerate."""
    script_content = '''#!/bin/bash
# Multi-GPU training launch script
# This script uses accelerate to launch training on multiple GPUs

# Initialize accelerate configuration (run this once)
# accelerate config

# Launch training with accelerate
accelerate launch train_nucleotide_transformer.py

# Alternative: Launch with torchrun for more control
# torchrun --nproc_per_node=2 train_nucleotide_transformer.py
'''
    
    with open("launch_multi_gpu.sh", "w") as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod("launch_multi_gpu.sh", 0o755)
    print("\nCreated launch_multi_gpu.sh script for multi-GPU training")
    print("To use multi-GPU training:")
    print("1. Run: accelerate config (first time only)")
    print("2. Run: ./launch_multi_gpu.sh")

if use_multi_gpu:
    create_launch_script()

# Determine all tasks and run training for each
if len(TRAINING_TASK) == 0:
    all_tasks = sorted(set(raw["train"]["task"]))
else:
    all_tasks = TRAINING_TASK
print(f"\nDiscovered {len(all_tasks)} tasks: {all_tasks}")
for t in all_tasks:
    train_single_task(t)
