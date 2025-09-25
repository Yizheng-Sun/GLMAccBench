#!/usr/bin/env python3
"""
Script to download the InstaDeepAI/nucleotide-transformer-2.5b-1000g model and dataset from Hugging Face
and save them locally for offline use.
"""

import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch

def download_model(model_name: str, local_dir: str = "./models/nucleotide-transformer-2.5b-1000g"):
    """
    Download the nucleotide transformer model from Hugging Face to local directory.
    
    Args:
        model_name (str): The Hugging Face model identifier
        local_dir (str): Local directory to save the model
    """
    print(f"Downloading model: {model_name}")
    print(f"Local directory: {local_dir}")
    
    # Create local directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=local_dir,
            local_files_only=False
        )
        
        print("Downloading model...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=local_dir,
            local_files_only=False,
            torch_dtype=torch.float16,  # Use float16 to save memory
            device_map="auto"  # Automatically handle device placement
        )
        
        print("Saving tokenizer locally...")
        tokenizer.save_pretrained(local_dir)
        
        print("Saving model locally...")
        model.save_pretrained(local_dir)
        
        print(f"✅ Model successfully downloaded and saved to: {local_dir}")
        print(f"Model size: {model.num_parameters():,} parameters")
        
        # Print model info
        print("\nModel information:")
        print(f"- Model name: {model_name}")
        print(f"- Local path: {os.path.abspath(local_dir)}")
        print(f"- Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"- Model config: {model.config}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        raise

def download_dataset(dataset_name: str, local_dir: str = "./datasets/nucleotide_transformer_downstream_tasks_revised"):
    """
    Download the nucleotide transformer downstream tasks dataset from Hugging Face to local directory.
    
    Args:
        dataset_name (str): The Hugging Face dataset identifier
        local_dir (str): Local directory to save the dataset
    """
    print(f"Downloading dataset: {dataset_name}")
    print(f"Local directory: {local_dir}")
    
    # Create local directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        print("Loading dataset...")
        dataset = load_dataset(
            dataset_name,
            cache_dir=local_dir,
            trust_remote_code=True  # Some datasets may require this
        )
        
        print("Saving dataset locally...")
        # Save each split separately to ensure proper structure preservation
        for split_name, split_data in dataset.items():
            split_dir = os.path.join(local_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            split_data.save_to_disk(split_dir)
        
        # Also save the full dataset for compatibility
        dataset.save_to_disk(local_dir)
        
        print(f"✅ Dataset successfully downloaded and saved to: {local_dir}")
        
        # Print dataset info
        print("\nDataset information:")
        print(f"- Dataset name: {dataset_name}")
        print(f"- Local path: {os.path.abspath(local_dir)}")
        print(f"- Available splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"- {split_name}: {len(split_data)} examples")
            if len(split_data) > 0:
                print(f"  - Features: {list(split_data.features.keys())}")
                print(f"  - Example keys: {list(split_data[0].keys()) if hasattr(split_data[0], 'keys') else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download nucleotide transformer model and dataset from Hugging Face")
    parser.add_argument(
        "--model-name", 
        default="InstaDeepAI/nucleotide-transformer-2.5b-1000g",
        help="Hugging Face model identifier (default: InstaDeepAI/nucleotide-transformer-2.5b-1000g)"
    )
    parser.add_argument(
        "--dataset-name",
        default="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
        help="Hugging Face dataset identifier (default: InstaDeepAI/nucleotide_transformer_downstream_tasks_revised)"
    )
    parser.add_argument(
        "--model-dir",
        default="./models/nucleotide-transformer-2.5b-1000g",
        help="Local directory to save the model (default: ./models/nucleotide-transformer-2.5b-1000g)"
    )
    parser.add_argument(
        "--dataset-dir",
        default="./datasets/nucleotide_transformer_downstream_tasks_revised",
        help="Local directory to save the dataset (default: ./datasets/nucleotide_transformer_downstream_tasks_revised)"
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        default=True,
        help="Download the model (default: True)"
    )
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        default=True,
        help="Download the dataset (default: True)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download even if files already exist locally"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting download process...")
    print("=" * 50)
    
    # Download model if requested
    if args.download_model:
        print("\n📦 MODEL DOWNLOAD")
        print("-" * 30)
        
        # Check if model already exists locally
        if os.path.exists(args.model_dir) and not args.force_download:
            print(f"⚠️  Model already exists at: {args.model_dir}")
            print("Use --force-download to re-download")
        else:
            download_model(args.model_name, args.model_dir)
    
    # Download dataset if requested
    if args.download_dataset:
        print("\n📊 DATASET DOWNLOAD")
        print("-" * 30)
        
        # Check if dataset already exists locally
        if os.path.exists(args.dataset_dir) and not args.force_download:
            print(f"⚠️  Dataset already exists at: {args.dataset_dir}")
            print("Use --force-download to re-download")
        else:
            download_dataset(args.dataset_name, args.dataset_dir)
    
    print("\n" + "=" * 50)
    print("✅ Download process completed!")

if __name__ == "__main__":
    main()
