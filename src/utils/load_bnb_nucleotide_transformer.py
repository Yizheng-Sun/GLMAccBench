"""
Model loading script with quantization options for nucleotide-transformer-2.5b-1000g
Supports both 4-bit and 8-bit quantization for memory efficiency.
"""

import argparse
import torch
import os
from transformers import AutoModelForSequenceClassification, AutoConfig
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from huggingface_hub import snapshot_download


def create_quantization_config(quantization_type="4bit"):
    """
    Create quantization configuration based on the specified type.
    
    Args:
        quantization_type (str): Either "4bit", "8bit", or "none"
    
    Returns:
        BnbQuantizationConfig or None: Configured quantization settings, or None for no quantization
    """
    if quantization_type == "4bit":
        return BnbQuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization_type == "8bit":
        return BnbQuantizationConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif quantization_type == "none":
        return None
    else:
        raise ValueError("quantization_type must be either '4bit', '8bit', or 'none'")


def initialize_classifier_weights(model):
    """
    Properly initialize classifier weights to prevent gradient explosion.
    
    Args:
        model: The model with a classifier to initialize
    """
    print("Initializing classifier weights with proper scaling...")
    
    if hasattr(model, 'classifier'):
        if hasattr(model.classifier, 'dense') and hasattr(model.classifier, 'out_proj'):
            # For EsmClassificationHead structure (common in ESM models)
            print("Initializing EsmClassificationHead...")
            
            # Initialize dense layer with smaller weights
            if hasattr(model.classifier.dense, 'weight'):
                torch.nn.init.normal_(model.classifier.dense.weight, mean=0.0, std=0.02)
                print("✅ Initialized dense layer weights (std=0.02)")
            
            if hasattr(model.classifier.dense, 'bias') and model.classifier.dense.bias is not None:
                torch.nn.init.zeros_(model.classifier.dense.bias)
                print("✅ Initialized dense layer bias to zero")
            
            # Initialize output projection with even smaller weights
            if hasattr(model.classifier.out_proj, 'weight'):
                torch.nn.init.normal_(model.classifier.out_proj.weight, mean=0.0, std=0.01)
                print("✅ Initialized out_proj weights (std=0.01)")
            
            if hasattr(model.classifier.out_proj, 'bias') and model.classifier.out_proj.bias is not None:
                torch.nn.init.zeros_(model.classifier.out_proj.bias)
                print("✅ Initialized out_proj bias to zero")
                
        elif hasattr(model.classifier, 'weight'):
            # For simple Linear classifier
            print("Initializing Linear classifier...")
            torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.01)
            print("✅ Initialized classifier weights (std=0.01)")
            
            if hasattr(model.classifier, 'bias') and model.classifier.bias is not None:
                torch.nn.init.zeros_(model.classifier.bias)
                print("✅ Initialized classifier bias to zero")
        else:
            print("⚠️  Warning: Unknown classifier structure, using default initialization")
    else:
        print("⚠️  Warning: No classifier found in model")


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


def print_gpu_memory_usage():
    """Print current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        # Clear cache first for accurate measurement
        torch.cuda.empty_cache()
        
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # Convert to GB
        
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    else:
        print("CUDA is not available")


def main():
    """Main function to run the model loading script."""
    parser = argparse.ArgumentParser(description="Load nucleotide transformer with quantization")
    parser.add_argument(
        "--quantization", 
        choices=["4bit", "8bit", "none"], 
        default="4bit",
        help="Quantization precision: 4bit, 8bit, or none for full precision (default: 4bit)"
    )
    parser.add_argument(
        "--model-id", 
        default="InstaDeepAI/nucleotide-transformer-2.5b-1000g",
        help="HuggingFace model identifier (fallback if local model not found)"
    )
    parser.add_argument(
        "--local-model-path",
        default="./models/nucleotide-transformer-2.5b-1000g",
        help="Path to local model directory (default: ./models/nucleotide-transformer-2.5b-1000g)"
    )
    parser.add_argument(
        "--trained-model-path",
        type=str,
        default=None,
        help="Path to trained model weights directory (if loading a custom trained model)"
    )
    parser.add_argument(
        "--reinitialize-classifier",
        action="store_true",
        default=True,
        help="Whether to reinitialize classifier weights with proper scaling (default: True)"
    )
    parser.add_argument(
        "--keep-classifier",
        action="store_true",
        help="Keep original classifier weights (overrides --reinitialize-classifier)"
    )
    
    args = parser.parse_args()
    
    if args.quantization == "none":
        print(f"Starting model loading without quantization (full precision)...")
    else:
        print(f"Starting model loading with {args.quantization} quantization...")
    
    if args.trained_model_path:
        print(f"Trained model path: {args.trained_model_path}")
    else:
        print(f"Local model path: {args.local_model_path}")
        print(f"Fallback model ID: {args.model_id}")
    
    # Determine whether to reinitialize classifier
    reinitialize = args.reinitialize_classifier and not args.keep_classifier
    
    # Load the model (with or without quantization)
    model = load_model(args.model_id, args.quantization, args.trained_model_path, args.local_model_path, reinitialize)
    
    print("Model loaded successfully!")
    print_gpu_memory_usage()
    
    return model


if __name__ == "__main__":
    model = main()