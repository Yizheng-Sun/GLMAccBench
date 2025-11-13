from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import os
import argparse
import sys
from datasets import load_dataset

def download_and_save_model(model_path, local_save_path):
    """Download and save any model and tokenizer locally"""
    print(f"Downloading and saving model from: {model_path}")
    
    try:
        # Download the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Try different model types to handle various architectures
        model = None
        model_classes = [
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForSequenceClassification,
            AutoModelForCausalLM
        ]
        
        for model_class in model_classes:
            try:
                model = model_class.from_pretrained(model_path)
                print(f"Successfully loaded model using {model_class.__name__}")
                break
            except Exception as e:
                print(f"Failed to load with {model_class.__name__}: {str(e)}")
                continue
        
        if model is None:
            raise Exception("Could not load model with any of the available model classes")
        
        # Save them locally
        os.makedirs(local_save_path, exist_ok=True)
        tokenizer.save_pretrained(local_save_path)
        model.save_pretrained(local_save_path)
        
        print(f"Model and tokenizer saved to: {local_save_path}")
        return tokenizer, model
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return None, None

def download_and_save_dataset(dataset_path, local_save_path, dataset_config=None):
    """Download and save any dataset locally"""
    print(f"Downloading and saving dataset from: {dataset_path}")
    
    try:
        # Load the dataset
        if dataset_config:
            dataset = load_dataset(dataset_path, dataset_config)
        else:
            dataset = load_dataset(dataset_path)
        
        # Save dataset locally
        os.makedirs(local_save_path, exist_ok=True)
        dataset.save_to_disk(local_save_path)
        
        print(f"Dataset saved to: {local_save_path}")
        return dataset
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return None

def load_model_from_local(local_path):
    """Load model and tokenizer from local storage"""
    if not os.path.exists(local_path):
        print(f"Local model not found at {local_path}. Please download first.")
        return None, None
    
    print(f"Loading model and tokenizer from local storage: {local_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        
        # Try different model types to handle various architectures
        model = None
        model_classes = [
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForSequenceClassification,
            AutoModelForCausalLM
        ]
        
        for model_class in model_classes:
            try:
                model = model_class.from_pretrained(local_path)
                print(f"Successfully loaded local model using {model_class.__name__}")
                break
            except Exception as e:
                continue
        
        if model is None:
            raise Exception("Could not load local model with any of the available model classes")
            
        print("Model and tokenizer loaded successfully from local storage!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading local model: {str(e)}")
        return None, None

def load_dataset_from_local(local_path):
    """Load dataset from local storage"""
    if not os.path.exists(local_path):
        print(f"Local dataset not found at {local_path}. Please download first.")
        return None
    
    print(f"Loading dataset from local storage: {local_path}")
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(local_path)
        print("Dataset loaded successfully from local storage!")
        return dataset
    except Exception as e:
        print(f"Error loading local dataset: {str(e)}")
        return None

def main():
    """Main function to handle command-line arguments and download/load models or datasets"""
    parser = argparse.ArgumentParser(description="Download and save HuggingFace models or datasets locally")
    
    # Add mutually exclusive group for model or dataset
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="HuggingFace model path to download (e.g., 'bert-base-uncased')")
    group.add_argument("--dataset", type=str, help="HuggingFace dataset path to download (e.g., 'imdb')")
    
    parser.add_argument("--local-path", type=str, default=None, 
                       help="Local directory path to save the model or dataset")
    parser.add_argument("--dataset-config", type=str, default=None,
                       help="Dataset configuration name (optional, for datasets with multiple configs)")
    parser.add_argument("--load-only", action="store_true",
                       help="Only load from local path without downloading")
    
    args = parser.parse_args()
    
    print("=== HuggingFace Model and Dataset Downloader ===\n")
    
    if args.model:
        print(f"Working with model: {args.model}")
        if args.local_path is None and args.model is not None:
            local_path = f"./models/{args.model}"
        else:
            local_path = os.path.abspath(args.local_path)
        
        if args.load_only:
            # Load from local only
            tokenizer, model = load_model_from_local(local_path)
            if tokenizer is not None and model is not None:
                print(f"Model loaded successfully from: {local_path}")
            else:
                print("Failed to load model from local path")
                sys.exit(1)
        else:
            # Check if local files exist
            model_exists = os.path.exists(local_path)
            print(f"Local model exists: {model_exists}")
            
            if not model_exists:
                tokenizer, model = download_and_save_model(args.model, local_path)
                if tokenizer is None or model is None:
                    print("Failed to download model")
                    sys.exit(1)
            else:
                print("Model already exists locally. Loading from local storage...")
                tokenizer, model = load_model_from_local(local_path)
                if tokenizer is None or model is None:
                    print("Failed to load existing local model. You may want to re-download.")
                    sys.exit(1)
    
    elif args.dataset:
        print(f"Working with dataset: {args.dataset}")
        if args.local_path is None and args.dataset is not None:
            local_path = f"./datasets/{args.dataset}"
        else:
            local_path = os.path.abspath(args.local_path)
        
        if args.load_only:
            # Load from local only
            dataset = load_dataset_from_local(local_path)
            if dataset is not None:
                print(f"Dataset loaded successfully from: {local_path}")
            else:
                print("Failed to load dataset from local path")
                sys.exit(1)
        else:
            # Check if local files exist
            dataset_exists = os.path.exists(local_path)
            print(f"Local dataset exists: {dataset_exists}")
            
            if not dataset_exists:
                dataset = download_and_save_dataset(args.dataset, local_path, args.dataset_config)
                if dataset is None:
                    print("Failed to download dataset")
                    sys.exit(1)
            else:
                print("Dataset already exists locally. Loading from local storage...")
                dataset = load_dataset_from_local(local_path)
                if dataset is None:
                    print("Failed to load existing local dataset. You may want to re-download.")
                    sys.exit(1)
        
        # Print dataset info if successfully loaded
        if dataset is not None:
            print("\n=== Dataset Information ===")
            print(f"Dataset keys: {list(dataset.keys())}")
            print(f"Number of splits: {len(dataset)}")
            
            # Example: Access a specific split (if available)
            for split_name in dataset.keys():
                print(f"Split '{split_name}': {len(dataset[split_name])} examples")
                if len(dataset[split_name]) > 0:
                    print(f"First example keys: {list(dataset[split_name][0].keys())}")
                break
    
    print(f"\nOperation completed successfully!")
    print(f"Local path: {local_path}")

if __name__ == "__main__":
    main()
