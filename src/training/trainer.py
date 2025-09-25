"""Training utilities for genomic models."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datasets import Dataset

from ..models.base import BaseModel
from ..utils.memory import print_gpu_memory_usage, get_model_memory_footprint


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_dir: str = "./logs"
    logging_steps: int = 10
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Custom parameters
    freeze_backbone: bool = True
    reinitialize_classifier: bool = True
    use_gradient_clipping_callback: bool = True
    gradient_clip_threshold: float = 10.0
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            evaluation_strategy=self.evaluation_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            max_grad_norm=self.max_grad_norm,
            seed=self.seed,
            push_to_hub=self.push_to_hub,
            hub_model_id=self.hub_model_id,
            hub_token=self.hub_token,
            report_to=self.report_to
        )


class GenomicTrainer:
    """Trainer for genomic language models."""
    
    def __init__(self,
                 model: BaseModel,
                 config: TrainingConfig,
                 compute_metrics=None):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            compute_metrics: Optional metrics computation function
        """
        self.model = model
        self.config = config
        self.compute_metrics = compute_metrics
        self.trainer = None
        
    def train(self,
              train_dataset: Dataset,
              eval_dataset: Optional[Dataset] = None,
              callbacks: Optional[List[TrainerCallback]] = None) -> None:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callbacks: Optional list of training callbacks
        """
        # Load model if not already loaded
        if self.model.model is None:
            self.model.load()
            
        # Set up model for training
        self._prepare_model_for_training()
        
        # Print initial memory usage
        print("Initial memory state:")
        print_gpu_memory_usage()
        
        # Create training arguments
        training_args = self.config.to_training_arguments()
        
        # Set up callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
            
        # Create trainer
        self.trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.model.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # Start training
        print("Starting training...")
        train_result = self.trainer.train()
        
        # Save final model
        print(f"Saving final model to {self.config.output_dir}")
        self.trainer.save_model()
        self.model.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        self.trainer.save_metrics("train", metrics)
        
        print("Training completed!")
        return train_result
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            # Create a trainer just for evaluation
            training_args = self.config.to_training_arguments()
            self.trainer = Trainer(
                model=self.model.model,
                args=training_args,
                tokenizer=self.model.tokenizer,
                compute_metrics=self.compute_metrics
            )
            
        print("Evaluating model...")
        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        # Save metrics
        self.trainer.save_metrics("eval", metrics)
        
        return metrics
    
    def _prepare_model_for_training(self) -> None:
        """Prepare the model for training."""
        model = self.model.model
        
        # Freeze backbone if requested
        if self.config.freeze_backbone:
            print("Freezing backbone parameters...")
            frozen, trainable = self.model.freeze_backbone(freeze_classifier=False)
            print(f"Frozen parameters: {frozen:,}")
            print(f"Trainable parameters: {trainable:,}")
            
        # Reinitialize classifier if requested
        if self.config.reinitialize_classifier:
            print("Reinitializing classifier weights...")
            self.model.initialize_classifier_weights()
            
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
                
        # Set model to training mode
        model.train()
        
        # Print model memory footprint
        footprint = get_model_memory_footprint(model)
        print(f"Model memory footprint: {footprint}")
    
    def _get_default_callbacks(self) -> List[TrainerCallback]:
        """Get default training callbacks."""
        from .callbacks import TensorBoardCallback, GradientClippingCallback
        
        callbacks = []
        
        # Add TensorBoard callback
        if "tensorboard" in self.config.report_to:
            tb_callback = TensorBoardCallback(self.config.logging_dir)
            callbacks.append(tb_callback)
            
        # Add gradient clipping callback
        if self.config.use_gradient_clipping_callback:
            clip_callback = GradientClippingCallback(
                max_grad_norm=self.config.max_grad_norm,
                clip_threshold=self.config.gradient_clip_threshold
            )
            callbacks.append(clip_callback)
            
        return callbacks
    
    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Save a training checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
        """
        if self.trainer:
            self.trainer.save_model(checkpoint_dir)
            self.model.tokenizer.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved to: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        from transformers import AutoModelForSequenceClassification
        
        print(f"Loading checkpoint from: {checkpoint_dir}")
        self.model.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
            device_map="auto"
        )
        self.model.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        print("Checkpoint loaded successfully")
