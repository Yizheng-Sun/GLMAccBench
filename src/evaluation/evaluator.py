"""Model evaluation utilities."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
import json
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer, TrainingArguments
from datasets import Dataset

from ..models.base import BaseModel
from ..utils.memory import get_model_memory_footprint


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    batch_size: int = 8
    max_length: int = 512
    metrics: List[str] = None  # List of metrics to compute
    output_dir: str = "./evaluation_results"
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    device: str = "auto"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1"]


class Evaluator:
    """Model evaluator for genomic models."""
    
    def __init__(self, 
                 model: BaseModel,
                 config: EvaluationConfig):
        """
        Initialize the evaluator.
        
        Args:
            model: The model to evaluate
            config: Evaluation configuration
        """
        self.model = model
        self.config = config
        self.results: Dict[str, Any] = {}
        
    def evaluate(self,
                 eval_dataset: Dataset,
                 compute_metrics: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_dataset: Dataset to evaluate on
            compute_metrics: Optional custom metrics function
            
        Returns:
            Dictionary of evaluation results
        """
        # Load model if not loaded
        if self.model.model is None:
            self.model.load()
            
        # Use default metrics if not provided
        if compute_metrics is None:
            compute_metrics = self._create_compute_metrics()
            
        # Create trainer for evaluation
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_eval_batch_size=self.config.batch_size,
            remove_unused_columns=False,
            do_eval=True,
            do_train=False
        )
        
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            tokenizer=self.model.tokenizer,
            compute_metrics=compute_metrics
        )
        
        print("Starting evaluation...")
        
        # Run evaluation
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        
        # Get predictions if requested
        if self.config.save_predictions:
            predictions = trainer.predict(eval_dataset)
            self.results['predictions'] = {
                'logits': predictions.predictions.tolist(),
                'labels': predictions.label_ids.tolist()
            }
            
        # Add model information
        self.results.update({
            'eval_metrics': eval_results,
            'model_info': self._get_model_info(),
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save results
        self._save_results()
        
        print("Evaluation completed!")
        self._print_results(eval_results)
        
        return eval_results
    
    def benchmark(self,
                  datasets: Dict[str, Dataset]) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark the model on multiple datasets.
        
        Args:
            datasets: Dictionary of dataset_name -> Dataset
            
        Returns:
            Dictionary of results for each dataset
        """
        benchmark_results = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"\nEvaluating on {dataset_name}...")
            results = self.evaluate(dataset)
            benchmark_results[dataset_name] = results
            
        # Save combined results
        self.results['benchmark'] = benchmark_results
        self._save_results()
        
        # Print summary
        self._print_benchmark_summary(benchmark_results)
        
        return benchmark_results
    
    def compare_models(self,
                       models: Dict[str, BaseModel],
                       eval_dataset: Dataset) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: Dictionary of model_name -> Model
            eval_dataset: Dataset to evaluate on
            
        Returns:
            Dictionary of results for each model
        """
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Create evaluator for this model
            evaluator = Evaluator(model, self.config)
            results = evaluator.evaluate(eval_dataset)
            
            comparison_results[model_name] = {
                'metrics': results,
                'memory_footprint': get_model_memory_footprint(model.model)
            }
            
        # Save comparison
        self.results['model_comparison'] = comparison_results
        self._save_results()
        
        # Print comparison table
        self._print_comparison_table(comparison_results)
        
        return comparison_results
    
    def _create_compute_metrics(self) -> Callable:
        """Create default compute metrics function."""
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            
            # Get predicted classes
            if len(predictions.shape) > 1:
                preds = np.argmax(predictions, axis=1)
            else:
                preds = predictions
                
            # Compute metrics
            metrics = {}
            
            if "accuracy" in self.config.metrics:
                metrics["accuracy"] = accuracy_score(labels, preds)
                
            if any(m in self.config.metrics for m in ["precision", "recall", "f1"]):
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, preds, average='weighted'
                )
                
                if "precision" in self.config.metrics:
                    metrics["precision"] = precision
                if "recall" in self.config.metrics:
                    metrics["recall"] = recall
                if "f1" in self.config.metrics:
                    metrics["f1"] = f1
                    
            if self.config.save_confusion_matrix:
                cm = confusion_matrix(labels, preds)
                self.results['confusion_matrix'] = cm.tolist()
                
            return metrics
        
        return compute_metrics
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        info = {
            'model_name': self.model.config.model_name,
            'num_labels': self.model.config.num_labels
        }
        
        if self.model.model is not None:
            info.update(get_model_memory_footprint(self.model.model))
            
        return info
    
    def _save_results(self) -> None:
        """Save evaluation results to file."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        filepath = os.path.join(self.config.output_dir, filename)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"Results saved to: {filepath}")
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print evaluation results."""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key:20s}: {value:.4f}")
            else:
                print(f"{key:20s}: {value}")
                
        print("="*50)
    
    def _print_benchmark_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print benchmark summary table."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Create header
        metrics = list(next(iter(results.values())).keys())
        header = f"{'Dataset':20s} | " + " | ".join([f"{m:10s}" for m in metrics[:4]])
        print(header)
        print("-"*60)
        
        # Print results for each dataset
        for dataset_name, dataset_results in results.items():
            row = f"{dataset_name:20s} | "
            for metric in metrics[:4]:
                if metric in dataset_results:
                    value = dataset_results[metric]
                    if isinstance(value, float):
                        row += f"{value:10.4f} | "
                    else:
                        row += f"{str(value):10s} | "
            print(row)
            
        print("="*60)
    
    def _print_comparison_table(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print model comparison table."""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        # Create header
        header = f"{'Model':20s} | {'Accuracy':10s} | {'F1':10s} | {'Params (M)':12s} | {'Memory (MB)':12s}"
        print(header)
        print("-"*70)
        
        # Print results for each model
        for model_name, model_results in results.items():
            metrics = model_results.get('metrics', {})
            memory = model_results.get('memory_footprint', {})
            
            accuracy = metrics.get('eval_accuracy', metrics.get('accuracy', 0))
            f1 = metrics.get('eval_f1', metrics.get('f1', 0))
            params = memory.get('total_parameters', 0) / 1e6
            size_mb = memory.get('parameter_size_mb', 0)
            
            row = f"{model_name:20s} | {accuracy:10.4f} | {f1:10.4f} | {params:12.2f} | {size_mb:12.2f}"
            print(row)
            
        print("="*70)
