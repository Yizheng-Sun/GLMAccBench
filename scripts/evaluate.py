#!/usr/bin/env python3
"""
Evaluation script for genomic language models.

Example usage:
    python scripts/evaluate.py --model-path ./results/checkpoint-1000
    python scripts/evaluate.py --quantized-model-path ./quantized_models/w4a16
    python scripts/evaluate.py --compare model1.path model2.path --dataset test_data
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import NucleotideTransformer, NucleotideTransformerConfig
from src.data import DataLoader, DataConfig
from src.evaluation import Evaluator, EvaluationConfig
from src.utils import load_config
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


def load_model(model_path: str, model_name: str = None) -> NucleotideTransformer:
    """Load a model from path."""
    config = NucleotideTransformerConfig(
        model_name=model_name or model_path,
        model_path=model_path
    )
    model = NucleotideTransformer(config)
    model.load()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate genomic language models")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to trained model checkpoint")
    parser.add_argument("--quantized-model-path", type=str, default=None,
                       help="Path to quantized model")
    parser.add_argument("--model-name", type=str,
                       default="InstaDeepAI/nucleotide-transformer-2.5b-1000g",
                       help="Base model name (if not in checkpoint)")
    
    # Data arguments
    parser.add_argument("--dataset", type=str,
                       default="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
                       help="Dataset name or path")
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="Local path to dataset")
    parser.add_argument("--task", type=str, default=None,
                       help="Specific task within dataset")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate on")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--eval-samples", type=int, default=None,
                       help="Limit evaluation samples")
    
    # Evaluation arguments
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Evaluation batch size")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--metrics", nargs="+",
                       default=["accuracy", "precision", "recall", "f1"],
                       help="Metrics to compute")
    parser.add_argument("--save-predictions", action="store_true",
                       help="Save model predictions")
    parser.add_argument("--save-confusion-matrix", action="store_true",
                       help="Save confusion matrix")
    
    # Comparison mode
    parser.add_argument("--compare", nargs="+", default=None,
                       help="Compare multiple models (provide paths)")
    
    # Benchmark mode
    parser.add_argument("--benchmark", nargs="+", default=None,
                       help="Benchmark on multiple datasets")
    
    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config:
        config_dict = load_config(args.config)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config_dict[key.replace('-', '_')] = value
    else:
        config_dict = {k.replace('-', '_'): v for k, v in vars(args).items()}
    
    # Create evaluation configuration
    eval_config = EvaluationConfig(
        batch_size=config_dict.get('batch_size', 8),
        max_length=config_dict.get('max_length', 512),
        metrics=config_dict.get('metrics', ["accuracy", "precision", "recall", "f1"]),
        output_dir=config_dict.get('output_dir', "./evaluation_results"),
        save_predictions=config_dict.get('save_predictions', False),
        save_confusion_matrix=config_dict.get('save_confusion_matrix', False)
    )
    
    # Handle comparison mode
    if args.compare:
        print("Running model comparison...")
        
        # Load dataset
        data_config = DataConfig(
            dataset_name=config_dict.get('dataset'),
            dataset_path=config_dict.get('dataset_path'),
            task_name=config_dict.get('task'),
            max_length=config_dict.get('max_length', 512),
            eval_samples=config_dict.get('eval_samples')
        )
        data_loader = DataLoader(data_config)
        dataset = data_loader.load()
        
        # Load first model for tokenizer
        first_model = load_model(args.compare[0])
        dataset = data_loader.preprocess(first_model.tokenizer)
        _, eval_dataset = data_loader.get_splits()
        
        # Load all models
        models = {}
        for i, model_path in enumerate(args.compare):
            model_name = f"model_{i+1}"
            if os.path.basename(model_path):
                model_name = os.path.basename(model_path)
            
            print(f"Loading {model_name} from {model_path}")
            models[model_name] = load_model(model_path)
        
        # Create evaluator and compare
        evaluator = Evaluator(first_model, eval_config)
        results = evaluator.compare_models(models, eval_dataset)
        
        print("\nComparison completed!")
        return
    
    # Handle benchmark mode
    if args.benchmark:
        print("Running benchmark evaluation...")
        
        # Load model
        if args.quantized_model_path:
            model_path = args.quantized_model_path
        elif args.model_path:
            model_path = args.model_path
        else:
            raise ValueError("Must provide either --model-path or --quantized-model-path")
        
        model = load_model(model_path, args.model_name)
        
        # Load datasets
        datasets = {}
        for dataset_name in args.benchmark:
            print(f"Loading dataset: {dataset_name}")
            
            data_config = DataConfig(
                dataset_name=dataset_name,
                max_length=config_dict.get('max_length', 512),
                eval_samples=config_dict.get('eval_samples')
            )
            data_loader = DataLoader(data_config)
            dataset = data_loader.load()
            dataset = data_loader.preprocess(model.tokenizer)
            _, eval_dataset = data_loader.get_splits()
            
            datasets[dataset_name] = eval_dataset
        
        # Create evaluator and benchmark
        evaluator = Evaluator(model, eval_config)
        results = evaluator.benchmark(datasets)
        
        print("\nBenchmark completed!")
        return
    
    # Standard evaluation mode
    print("Running standard evaluation...")
    
    # Load model
    if args.quantized_model_path:
        print(f"Loading quantized model from: {args.quantized_model_path}")
        model = load_model(args.quantized_model_path, args.model_name)
    elif args.model_path:
        print(f"Loading model from: {args.model_path}")
        model = load_model(args.model_path, args.model_name)
    else:
        raise ValueError("Must provide either --model-path or --quantized-model-path")
    
    # Load and preprocess dataset
    data_config = DataConfig(
        dataset_name=config_dict.get('dataset'),
        dataset_path=config_dict.get('dataset_path'),
        task_name=config_dict.get('task'),
        eval_split=config_dict.get('split', 'test'),
        max_length=config_dict.get('max_length', 512),
        eval_samples=config_dict.get('eval_samples')
    )
    
    print(f"Loading dataset: {data_config.dataset_name}")
    data_loader = DataLoader(data_config)
    dataset = data_loader.load()
    dataset = data_loader.preprocess(model.tokenizer)
    
    # Get evaluation split
    if config_dict.get('split') == 'train':
        eval_dataset, _ = data_loader.get_splits()
    else:
        _, eval_dataset = data_loader.get_splits()
    
    # Create evaluator and evaluate
    evaluator = Evaluator(model, eval_config)
    results = evaluator.evaluate(eval_dataset, compute_metrics)
    
    print("\nEvaluation completed!")
    print(f"Results saved to: {eval_config.output_dir}")


if __name__ == "__main__":
    main()
