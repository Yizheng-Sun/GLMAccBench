#!/usr/bin/env python3
"""
Download script for models and datasets.

Example usage:
    python scripts/download.py --model InstaDeepAI/nucleotide-transformer-2.5b-1000g
    python scripts/download.py --dataset InstaDeepAI/nucleotide_transformer_downstream_tasks_revised
    python scripts/download.py --all
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import NucleotideTransformer, NucleotideTransformerConfig
from src.data import DataLoader


def download_model(model_name: str, output_path: str):
    """Download a model from HuggingFace Hub."""
    print(f"Downloading model: {model_name}")
    print(f"Output path: {output_path}")
    
    config = NucleotideTransformerConfig(model_name=model_name)
    model = NucleotideTransformer(config)
    
    # Download model
    model_path = model.download_from_hub(output_path)
    
    print(f"✅ Model downloaded successfully to: {model_path}")
    
    # Verify by loading
    print("Verifying download by loading model...")
    config_verify = NucleotideTransformerConfig(
        model_name=model_name,
        model_path=output_path
    )
    model_verify = NucleotideTransformer(config_verify)
    model_verify.load()
    
    print("✅ Model verified successfully!")
    
    # Print model info
    memory_stats = model_verify.get_memory_footprint()
    print("\nModel Information:")
    print(f"  Parameters: {memory_stats.get('parameter_count', 0):,}")
    print(f"  Size (MB): {memory_stats.get('parameter_size_mb', 0):.2f}")


def download_dataset(dataset_name: str, output_path: str):
    """Download a dataset from HuggingFace Hub."""
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output path: {output_path}")
    
    # Download dataset
    DataLoader.download_dataset(dataset_name, output_path)
    
    # Verify by loading
    print("Verifying download by loading dataset...")
    from src.data import DataConfig
    
    config = DataConfig(
        dataset_name=dataset_name,
        dataset_path=output_path
    )
    loader = DataLoader(config)
    dataset = loader.load()
    
    print("✅ Dataset verified successfully!")
    
    # Print dataset info
    print("\nDataset Information:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")


def main():
    parser = argparse.ArgumentParser(description="Download models and datasets")
    
    # Download targets
    parser.add_argument("--model", type=str, default=None,
                       help="Model name to download from HuggingFace")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset name to download from HuggingFace")
    parser.add_argument("--all", action="store_true",
                       help="Download default model and dataset")
    
    # Output paths
    parser.add_argument("--model-output", type=str, default="./models",
                       help="Output directory for models")
    parser.add_argument("--dataset-output", type=str, default="./datasets",
                       help="Output directory for datasets")
    
    # Default resources
    parser.add_argument("--default-model", type=str,
                       default="InstaDeepAI/nucleotide-transformer-2.5b-1000g",
                       help="Default model to download with --all")
    parser.add_argument("--default-dataset", type=str,
                       default="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
                       help="Default dataset to download with --all")
    
    args = parser.parse_args()
    
    # Check that at least one download target is specified
    if not args.model and not args.dataset and not args.all:
        parser.error("Must specify --model, --dataset, or --all")
    
    # Handle --all flag
    if args.all:
        print("Downloading default model and dataset...")
        
        # Download model
        model_output = os.path.join(
            args.model_output,
            os.path.basename(args.default_model)
        )
        download_model(args.default_model, model_output)
        
        print("\n" + "="*50 + "\n")
        
        # Download dataset
        dataset_output = os.path.join(
            args.dataset_output,
            os.path.basename(args.default_dataset)
        )
        download_dataset(args.default_dataset, dataset_output)
        
    else:
        # Download specified model
        if args.model:
            model_output = os.path.join(
                args.model_output,
                os.path.basename(args.model)
            )
            download_model(args.model, model_output)
        
        # Download specified dataset
        if args.dataset:
            if args.model:
                print("\n" + "="*50 + "\n")
            
            dataset_output = os.path.join(
                args.dataset_output,
                os.path.basename(args.dataset)
            )
            download_dataset(args.dataset, dataset_output)
    
    print("\n✅ All downloads completed successfully!")


if __name__ == "__main__":
    main()
