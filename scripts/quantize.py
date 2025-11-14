#!/usr/bin/env python3
"""
Quantization script for genomic language models.

Example usage:
    # BitsAndBytes quantization (fast, no calibration needed)
    python scripts/quantize.py --method bnb --bits 4
    
    # GPTQ quantization (requires calibration data)
    python scripts/quantize.py --method gptq --calibration-samples 512
    
    # Other methods
    python scripts/quantize.py --method w4a16 --output quantized_models/w4a16
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import NucleotideTransformer, NucleotideTransformerConfig
from src.quantization import (
    BitsAndBytesQuantizer, BnBConfig,
    LLMCompressorQuantizer, W4A4Config, W4A16Config, W8A8Config, GPTQConfig, AWQConfig
)
from src.data import DataLoader, DataConfig
from src.utils import load_config, save_config, print_gpu_memory_usage


def get_quantizer(method: str, bits: int, config_dict: dict):
    """Get the appropriate quantizer based on method."""
    
    if method == "bnb":
        config = BnBConfig(
            bits=bits,
            calibration_samples=config_dict.get('calibration_samples', 512),
            output_path=config_dict.get('output_path')
        )
        return BitsAndBytesQuantizer(config)
        
    elif method == "w4a4":
        config = W4A4Config(
            calibration_samples=config_dict.get('calibration_samples', 512),
            output_path=config_dict.get('output_path'),
            ignore=config_dict.get('ignore_layers')
        )
        return LLMCompressorQuantizer(config)
        
    elif method == "w4a16":
        config = W4A16Config(
            calibration_samples=config_dict.get('calibration_samples', 512),
            output_path=config_dict.get('output_path'),
            ignore=config_dict.get('ignore_layers')
        )
        return LLMCompressorQuantizer(config)
        
    elif method == "w8a8":
        config = W8A8Config(
            calibration_samples=config_dict.get('calibration_samples', 512),
            output_path=config_dict.get('output_path'),
            smoothquant_alpha=config_dict.get('smoothquant_alpha', 0.8),
            ignore=config_dict.get('ignore_layers')
        )
        return LLMCompressorQuantizer(config)
        
    elif method == "gptq":
        config = GPTQConfig(
            bits=bits,
            calibration_samples=config_dict.get('calibration_samples', 512),
            output_path=config_dict.get('output_path'),
            group_size=config_dict.get('group_size', 128),
            damp_percent=config_dict.get('damp_percent', 0.01),
            ignore=config_dict.get('ignore_layers')
        )
        return LLMCompressorQuantizer(config)
        
    elif method == "awq":
        config = AWQConfig(
            bits=bits,
            calibration_samples=config_dict.get('calibration_samples', 512),
            output_path=config_dict.get('output_path'),
            group_size=config_dict.get('group_size', 128),
            zero_point=config_dict.get('zero_point', True),
            ignore=config_dict.get('ignore_layers')
        )
        return LLMCompressorQuantizer(config)
        
    else:
        raise ValueError(f"Unknown quantization method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Quantize genomic language models")
    
    # Model arguments
    parser.add_argument("--model-name", type=str,
                       default="InstaDeepAI/nucleotide-transformer-2.5b-1000g",
                       help="HuggingFace model name or path")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Local path to model")
    parser.add_argument("--quantized-model-path", type=str, default=None,
                       help="Path to pre-quantized model to load")
    
    # Quantization arguments
    parser.add_argument("--method", type=str, required=True,
                       choices=["bnb", "w4a4", "w4a16", "w8a8", "gptq", "awq"],
                       help="Quantization method")
    parser.add_argument("--bits", type=int, default=4,
                       choices=[4, 8], help="Number of bits for quantization")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for quantized model")
    
    # Calibration arguments
    parser.add_argument("--calibration-dataset", type=str,
                       default="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
                       help="Dataset for calibration")
    parser.add_argument("--calibration-samples", type=int, default=512,
                       help="Number of calibration samples")
    
    # Method-specific arguments
    parser.add_argument("--smoothquant-alpha", type=float, default=0.8,
                       help="Alpha parameter for SmoothQuant (W8A8)")
    parser.add_argument("--group-size", type=int, default=128,
                       help="Group size for GPTQ/AWQ")
    parser.add_argument("--damp-percent", type=float, default=0.01,
                       help="Damping percentage for GPTQ")
    parser.add_argument("--zero-point", action="store_true",
                       help="Use zero point for AWQ")
    parser.add_argument("--ignore-layers", nargs="+", default=None,
                       help="Layers to ignore during quantization")
    
    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--save-config", type=str, default=None,
                       help="Save configuration to file")
    
    # Other arguments
    parser.add_argument("--test", action="store_true",
                       help="Test the quantized model after quantization")
    
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
    
    # Set output path if not specified
    if not config_dict.get('output'):
        config_dict['output_path'] = f"./quantized_models/{args.method}_{args.bits}bit"
    else:
        config_dict['output_path'] = config_dict['output']
    
    # Handle pre-quantized model loading
    if args.quantized_model_path:
        print(f"Loading pre-quantized model from: {args.quantized_model_path}")
        
        # Get quantizer to load the model
        quantizer = get_quantizer(args.method, args.bits, config_dict)
        model, tokenizer = quantizer.load_quantized(args.quantized_model_path)
        
        # Print memory stats
        print("\nMemory footprint:")
        memory_stats = quantizer.get_memory_footprint(model)
        for key, value in memory_stats.items():
            print(f"  {key}: {value}")
            
        print("\nPre-quantized model loaded successfully!")
        return
    
    # Create model configuration
    model_config = NucleotideTransformerConfig(
        model_name=config_dict.get('model_name'),
        model_path=config_dict.get('model_path')
    )
    
    # Create and load model
    print(f"Loading model: {model_config.model_name}")
    model = NucleotideTransformer(model_config)
    model.load()
    
    # Print initial memory usage
    print("\nInitial memory state:")
    print_gpu_memory_usage()
    
    # Get quantizer
    print(f"\nInitializing {args.method.upper()} quantizer...")
    quantizer = get_quantizer(args.method, args.bits, config_dict)
    
    # Load calibration dataset if needed (not for BnB)
    calibration_data = None
    if args.method != "bnb":
        print(f"Loading calibration dataset: {config_dict.get('calibration_dataset')}")
        data_config = DataConfig(
            dataset_name=config_dict.get('calibration_dataset'),
            train_samples=config_dict.get('calibration_samples', 512)
        )
        data_loader = DataLoader(data_config)
        data_loader.load()
        
        # Get first available task for calibration data
        available_tasks = data_loader.get_available_tasks()
        if not available_tasks:
            raise ValueError("No tasks found in calibration dataset")
        task_name = available_tasks[0]
        print(f"Using task '{task_name}' for calibration")
        
        # Prepare calibration data
        train_dataset, _ = data_loader.prepare_task_data(task_name, model.tokenizer)
        calibration_data = train_dataset
    
    # Perform quantization
    print(f"\nQuantizing model with {args.method.upper()} ({args.bits}-bit)...")
    quantized_model, tokenizer = quantizer.quantize(model, calibration_data)
    
    # Print final memory usage
    print("\nFinal memory state:")
    print_gpu_memory_usage()
    
    # Save configuration if requested
    if args.save_config:
        save_config({
            'method': args.method,
            'bits': args.bits,
            'model': model_config.__dict__,
            'quantization': quantizer.config.__dict__
        }, args.save_config)
    
    # Test the model if requested
    if args.test:
        print("\nTesting quantized model...")
        test_text = "ATCGATCGATCGATCG"
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = quantized_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
        print(f"Test input: {test_text}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predicted class: {predictions.item() if predictions.numel() == 1 else predictions.tolist()}")
    
    print(f"\nQuantization completed! Model saved to: {config_dict.get('output_path', 'model not saved')}")


if __name__ == "__main__":
    import torch  # Import torch for testing
    main()
