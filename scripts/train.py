#!/usr/bin/env python3
"""
Training script for genomic language models.

Example usage:
    python scripts/train.py --model nucleotide-transformer --config config/training.yaml
    python scripts/train.py --model nucleotide-transformer --freeze-backbone --epochs 3
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import NucleotideTransformer, NucleotideTransformerConfig
from src.data import DataLoader, DataConfig
from src.training import GenomicTrainer, TrainingConfig
from src.utils import load_config, save_config
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    parser = argparse.ArgumentParser(description="Train genomic language models")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="nucleotide-transformer",
                       help="Model type to train")
    parser.add_argument("--model-name", type=str, 
                       default="InstaDeepAI/nucleotide-transformer-2.5b-1000g",
                       help="HuggingFace model name or path")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Local path to model")
    parser.add_argument("--num-labels", type=int, default=2,
                       help="Number of classification labels")
    
    # Data arguments
    parser.add_argument("--dataset", type=str,
                       default="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
                       help="Dataset name or path")
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="Local path to dataset")
    parser.add_argument("--task", type=str, default=None,
                       help="Specific task within dataset")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--train-samples", type=int, default=None,
                       help="Limit training samples")
    parser.add_argument("--eval-samples", type=int, default=None,
                       help="Limit evaluation samples")
    
    # Training arguments
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                       help="Evaluation batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true",
                       help="Use BF16 training")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Use gradient checkpointing")
    
    # Model-specific arguments
    parser.add_argument("--freeze-backbone", action="store_true",
                       help="Freeze backbone parameters")
    parser.add_argument("--reinitialize-classifier", action="store_true",
                       default=True, help="Reinitialize classifier weights")
    
    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file (overrides other arguments)")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save-config", type=str, default=None,
                       help="Save configuration to file")
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config:
        config_dict = load_config(args.config)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config_dict[key] = value
    else:
        config_dict = vars(args)
    
    # Create model configuration
    model_config = NucleotideTransformerConfig(
        model_name=config_dict.get('model_name'),
        model_path=config_dict.get('model_path'),
        num_labels=config_dict.get('num_labels', 2),
        max_length=config_dict.get('max_length', 512),
        reinitialize_classifier=config_dict.get('reinitialize_classifier', True)
    )
    
    # Create and load model
    print(f"Loading model: {model_config.model_name}")
    model = NucleotideTransformer(model_config)
    model.load()
    
    # Create data configuration
    data_config = DataConfig(
        dataset_name=config_dict.get('dataset'),
        dataset_path=config_dict.get('dataset_path'),
        task_name=config_dict.get('task'),
        max_length=config_dict.get('max_length', 512),
        train_samples=config_dict.get('train_samples'),
        eval_samples=config_dict.get('eval_samples'),
        seed=config_dict.get('seed', 42)
    )
    
    # Load and preprocess data
    print(f"Loading dataset: {data_config.dataset_name}")
    data_loader = DataLoader(data_config)
    data_loader.load()
    
    # Check if task is specified
    if not data_config.task_name:
        # List available tasks and use the first one
        available_tasks = data_loader.get_available_tasks()
        print(f"Available tasks: {available_tasks}")
        if not available_tasks:
            raise ValueError("No tasks found in dataset")
        data_config.task_name = available_tasks[0]
        print(f"No task specified, using: {data_config.task_name}")
    
    # Prepare task-specific data
    train_dataset, eval_dataset = data_loader.prepare_task_data(
        data_config.task_name, 
        model.tokenizer
    )
    
    # Update model config with correct number of labels
    num_labels = data_loader.get_num_labels(data_config.task_name)
    id2label, label2id = data_loader.get_label_mappings(data_config.task_name)
    print(f"Task '{data_config.task_name}' has {num_labels} labels")
    
    # Update model configuration
    model.model.config.num_labels = num_labels
    model.model.config.id2label = id2label
    model.model.config.label2id = label2id
    model.model.config.problem_type = "single_label_classification"
    
    # Create training configuration
    training_config = TrainingConfig(
        output_dir=config_dict.get('output_dir', './results'),
        num_train_epochs=config_dict.get('epochs', 3),
        per_device_train_batch_size=config_dict.get('batch_size', 8),
        per_device_eval_batch_size=config_dict.get('eval_batch_size', 8),
        learning_rate=config_dict.get('learning_rate', 5e-5),
        weight_decay=config_dict.get('weight_decay', 0.01),
        warmup_steps=config_dict.get('warmup_steps', 500),
        gradient_accumulation_steps=config_dict.get('gradient_accumulation', 1),
        fp16=config_dict.get('fp16', False),
        bf16=config_dict.get('bf16', False),
        gradient_checkpointing=config_dict.get('gradient_checkpointing', False),
        freeze_backbone=config_dict.get('freeze_backbone', False),
        reinitialize_classifier=config_dict.get('reinitialize_classifier', True),
        seed=config_dict.get('seed', 42)
    )
    
    # Save configuration if requested
    if args.save_config:
        save_config({
            'model': model_config.__dict__,
            'data': data_config.__dict__,
            'training': training_config.__dict__
        }, args.save_config)
    
    # Create trainer and train
    print("Initializing trainer...")
    trainer = GenomicTrainer(model, training_config, compute_metrics)
    
    print("Starting training...")
    trainer.train(train_dataset, eval_dataset)
    
    # Final evaluation
    print("\nFinal evaluation...")
    metrics = trainer.evaluate(eval_dataset)
    
    print("\nTraining completed!")
    print(f"Results saved to: {training_config.output_dir}")


if __name__ == "__main__":
    main()
