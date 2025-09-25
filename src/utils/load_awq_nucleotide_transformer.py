"""
AWQ Quantization script for nucleotide-transformer-2.5b-1000g
Applies 4-bit AWQ quantization using llm-compressor to reduce model size and memory usage.

Features:
- Quantize a new nucleotide transformer model using AWQ
- Load and test pre-quantized models directly
- Support for local and HuggingFace model loading
- Automatic calibration dataset preparation
- Memory usage monitoring throughout the process

Usage:
1. Quantize a new model:
   python load_awq_nucleotide_transformer.py
   
2. Load a pre-quantized model:
   python load_awq_nucleotide_transformer.py --quantized-model-path ./path/to/quantized/model
"""

import argparse
import torch
import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation


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


def load_nucleotide_model(model_id, local_model_path=None, quantized_model_path=None):
    """
    Load the nucleotide transformer model for quantization or load pre-quantized model.
    
    Args:
        model_id (str): HuggingFace model identifier (fallback)
        local_model_path (str): Path to local model directory (optional)
        quantized_model_path (str): Path to pre-quantized model directory (optional)
    
    Returns:
        tuple: (model, tokenizer, is_already_quantized)
    """
    # Check if we should load a pre-quantized model
    if quantized_model_path and os.path.exists(quantized_model_path):
        print(f"Loading pre-quantized model from: {quantized_model_path}")
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                quantized_model_path,
                torch_dtype="auto",
                device_map="auto"  # Use device_map for quantized models
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            
            print(f"Pre-quantized model loaded successfully!")
            print(f"Model type: {type(model)}")
            print(f"Model config: {model.config}")
            
            # Print GPU memory usage after loading quantized model
            print("\nGPU Memory After Loading Quantized Model:")
            print_gpu_memory_usage()
            
            # Check if model appears to be quantized
            is_quantized = any(hasattr(module, 'weight') and 
                             hasattr(module.weight, 'dtype') and 
                             'int' in str(module.weight.dtype).lower() 
                             for module in model.modules())
            
            if is_quantized:
                print("✅ Model appears to be quantized (detected integer weights)")
            else:
                print("⚠️  Model loaded but quantization status unclear")
            
            return model, tokenizer, True
            
        except Exception as e:
            print(f"❌ Error loading quantized model from {quantized_model_path}: {e}")
            print("Falling back to loading unquantized model for quantization...")
    
    # Load unquantized model for quantization
    # Determine which model to load: local first, then HuggingFace
    model_path = None
    if local_model_path and os.path.exists(local_model_path):
        model_path = local_model_path
        print(f"Loading model from local directory: {model_path}")
    else:
        model_path = model_id
        print(f"Local model not found, loading from Hugging Face: {model_path}")
        if local_model_path:
            print(f"Consider downloading the model locally to {local_model_path} for faster loading")
    
    print(f"Loading model and tokenizer for quantization...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=None  # Keep on CPU initially for quantization
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")
    
    return model, tokenizer, False


def prepare_calibration_data(tokenizer, dataset_path=None, num_samples=256, max_length=512):
    """
    Prepare calibration dataset for AWQ quantization.
    
    Args:
        tokenizer: The model tokenizer
        dataset_path (str): Path to local nucleotide dataset (optional)
        num_samples (int): Number of calibration samples (AWQ typically uses fewer samples than GPTQ)
        max_length (int): Maximum sequence length
    
    Returns:
        datasets.Dataset: Processed calibration dataset
    """
    print(f"Preparing calibration data with {num_samples} samples...")
    
    # Try to use local nucleotide transformer dataset first
    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading nucleotide dataset from local directory: {dataset_path}")
        try:
            # Load from individual arrow files to preserve structure
            train_data = load_dataset('arrow', data_files=f'{dataset_path}/train/data-00000-of-00001.arrow')['train']
            
            # Filter for a specific task (use promoter_all as it's commonly used)
            ds = train_data.filter(lambda ex: ex["task"] == "promoter_all")
            print(f"Using 'promoter_all' task data with {len(ds)} total samples")
            
            # Take a subset for calibration
            if len(ds) > num_samples:
                ds = ds.select(range(num_samples))
            else:
                print(f"Warning: Dataset only has {len(ds)} samples, using all available")
            
            ds = ds.shuffle(seed=42)
            
            def preprocess_nucleotide(example):
                """Convert nucleotide sequence to text format expected by tokenizer"""
                return {"text": example["sequence"]}
            
            ds = ds.map(preprocess_nucleotide)
            
        except Exception as e:
            print(f"Error loading local nucleotide dataset: {e}")
            print("Falling back to HuggingFace dataset...")
            ds = None
    else:
        ds = None
    
    # Fallback to HuggingFace nucleotide dataset
    if ds is None:
        print("Loading nucleotide dataset from HuggingFace...")
        try:
            raw_ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", split=f"train[:{num_samples * 2}]")
            
            # Filter for a specific task
            ds = raw_ds.filter(lambda ex: ex["task"] == "promoter_all")
            
            # Take required number of samples
            if len(ds) > num_samples:
                ds = ds.select(range(num_samples))
            
            ds = ds.shuffle(seed=42)
            
            def preprocess_nucleotide(example):
                """Convert nucleotide sequence to text format expected by tokenizer"""
                return {"text": example["sequence"]}
            
            ds = ds.map(preprocess_nucleotide)
            
        except Exception as e:
            print(f"Error loading nucleotide dataset from HuggingFace: {e}")
            print("Using synthetic nucleotide sequences for calibration...")
            
            # Generate synthetic nucleotide sequences as fallback
            import random
            random.seed(42)
            
            def generate_nucleotide_sequence(length=500):  # Shorter sequences for AWQ
                """Generate a random nucleotide sequence"""
                bases = ['A', 'T', 'G', 'C']
                return ''.join(random.choices(bases, k=length))
            
            synthetic_data = [{"text": generate_nucleotide_sequence()} for _ in range(num_samples)]
            ds = load_dataset("json", data=synthetic_data, split="train")
    
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,  # Add special tokens for nucleotide sequences
        )
    
    print("Tokenizing calibration data...")
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    
    print(f"Calibration dataset prepared with {len(ds)} samples")
    print(f"Sample tokenized length: {len(ds[0]['input_ids']) if len(ds) > 0 else 'N/A'}")
    
    return ds


def apply_awq_quantization(model, dataset, max_seq_length=512, num_calibration_samples=256):
    """
    Apply AWQ quantization to the model.
    
    Args:
        model: The model to quantize
        dataset: Calibration dataset
        max_seq_length (int): Maximum sequence length
        num_calibration_samples (int): Number of calibration samples
    
    Returns:
        torch.nn.Module: Quantized model
    """
    print("Configuring AWQ quantization...")
    
    # Configure the quantization algorithm
    # Use W4A16_ASYM (4-bit weights, 16-bit activations, asymmetric quantization)
    # Ignore the classifier head (similar to ignoring lm_head in language models)
    recipe = [
        AWQModifier(
            ignore=["classifier"],  # Don't quantize the classification head
            scheme="W4A16_ASYM",    # 4-bit asymmetric quantization
            targets=["Linear"]      # Target linear layers
        ),
    ]
    
    print("Applying AWQ quantization...")
    print("This may take some time depending on model size...")
    print("AWQ typically requires less calibration time than GPTQ...")
    
    # Apply quantization
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
    )
    
    print("AWQ quantization completed successfully!")
    return model


def test_quantized_model(model, tokenizer):
    """
    Test the quantized model with a sample nucleotide sequence.
    
    Args:
        model: The quantized model
        tokenizer: The tokenizer
    """
    print("\n" + "="*60)
    print("TESTING QUANTIZED MODEL")
    print("="*60)
    
    # Prepare model for generation/inference
    dispatch_for_generation(model)
    
    # Test with a sample nucleotide sequence
    test_sequence = "ATCGATCGATCGATCG" * 10  # Repeat pattern to make it longer
    print(f"Test sequence: {test_sequence[:50]}...")
    
    # Tokenize the test sequence
    sample = tokenizer(test_sequence, return_tensors="pt", truncation=True, max_length=512)
    sample = {key: value.to(model.device) for key, value in sample.items()}
    
    try:
        # For sequence classification, get predictions
        with torch.no_grad():
            outputs = model(**sample)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=-1)
            
        print(f"Model output shape: {logits.shape}")
        print(f"Predictions: {predictions}")
        print(f"Predicted class: {torch.argmax(predictions, dim=-1).item()}")
        print("Model test completed successfully!")
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        print("This might be expected if the model requires specific input formats")


def save_quantized_model(model, tokenizer, model_id, output_dir=None):
    """
    Save the quantized model to disk.
    
    Args:
        model: The quantized model
        tokenizer: The tokenizer
        model_id (str): Original model identifier
        output_dir (str): Custom output directory (optional)
    """
    if output_dir is None:
        # Create output directory name based on model ID
        model_name = model_id.rstrip("/").split("/")[-1]
        output_dir = f"{model_name}-awq-asym"
    
    print(f"\nSaving quantized model to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Quantized model saved successfully to {output_dir}")
    print(f"Model size reduction: Approximately 75% (4-bit vs 16-bit weights)")
    
    return output_dir


def main():
    """Main function to run AWQ quantization on nucleotide transformer."""
    parser = argparse.ArgumentParser(description="Apply AWQ quantization to nucleotide transformer")
    parser.add_argument(
        "--model-id", 
        default="InstaDeepAI/nucleotide-transformer-2.5b-1000g",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--local-model-path",
        default="./models/nucleotide-transformer-2.5b-1000g",
        help="Path to local model directory"
    )
    parser.add_argument(
        "--quantized-model-path",
        type=str,
        default=None,
        help="Path to pre-quantized model directory (if provided, skips quantization and loads the model directly)"
    )
    parser.add_argument(
        "--dataset-path",
        default="./datasets/nucleotide_transformer_downstream_tasks_revised",
        help="Path to local nucleotide dataset for calibration"
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256, AWQ typically uses fewer samples than GPTQ)"
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Maximum sequence length for calibration (default: 512, shorter than GPTQ for faster processing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for quantized model (default: auto-generated)"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip testing the quantized model"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    if args.quantized_model_path:
        print("LOADING PRE-QUANTIZED NUCLEOTIDE TRANSFORMER (AWQ)")
    else:
        print("AWQ QUANTIZATION FOR NUCLEOTIDE TRANSFORMER")
    print("="*80)
    print(f"Model ID: {args.model_id}")
    print(f"Local model path: {args.local_model_path}")
    if args.quantized_model_path:
        print(f"Quantized model path: {args.quantized_model_path}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Calibration samples: {args.num_calibration_samples}")
    print(f"Max sequence length: {args.max_sequence_length}")
    print("="*80)
    
    # Print initial GPU memory usage
    print("\nInitial GPU Memory Usage:")
    print_gpu_memory_usage()
    
    # Step 1: Load model and tokenizer
    print("\n" + "="*60)
    print("STEP 1: LOADING MODEL")
    print("="*60)
    model, tokenizer, is_already_quantized = load_nucleotide_model(
        args.model_id, 
        args.local_model_path, 
        args.quantized_model_path
    )
    
    print("\nGPU Memory After Model Loading:")
    print_gpu_memory_usage()
    
    if is_already_quantized:
        print("\n" + "="*60)
        print("PRE-QUANTIZED MODEL LOADED - SKIPPING QUANTIZATION")
        print("="*60)
        quantized_model = model
        
        # Test the pre-quantized model (optional)
        if not args.skip_test:
            test_quantized_model(quantized_model, tokenizer)
        
        print("\n" + "="*80)
        print("PRE-QUANTIZED MODEL READY FOR USE!")
        print("="*80)
        print(f"Model loaded from: {args.quantized_model_path}")
        print(f"The model is ready for inference")
        print("="*80)
        
    else:
        # Step 2: Prepare calibration data
        print("\n" + "="*60)
        print("STEP 2: PREPARING CALIBRATION DATA")
        print("="*60)
        calibration_dataset = prepare_calibration_data(
            tokenizer,
            args.dataset_path,
            args.num_calibration_samples,
            args.max_sequence_length
        )
        
        # Step 3: Apply AWQ quantization
        print("\n" + "="*60)
        print("STEP 3: APPLYING AWQ QUANTIZATION")
        print("="*60)
        quantized_model = apply_awq_quantization(
            model,
            calibration_dataset,
            args.max_sequence_length,
            args.num_calibration_samples
        )
        
        print("\nGPU Memory After Quantization:")
        print_gpu_memory_usage()
        
        # Step 4: Test quantized model (optional)
        if not args.skip_test:
            test_quantized_model(quantized_model, tokenizer)
        
        # Step 5: Save quantized model
        print("\n" + "="*60)
        print("STEP 4: SAVING QUANTIZED MODEL")
        print("="*60)
        output_path = save_quantized_model(quantized_model, tokenizer, args.model_id, args.output_dir)
        
        print("\n" + "="*80)
        print("AWQ QUANTIZATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Quantized model saved to: {output_path}")
        print(f"Expected memory reduction: ~75% (4-bit weights)")
        print(f"AWQ benefits: Faster inference with minimal accuracy loss")
        print(f"The quantized model can be loaded using standard transformers:")
        print(f"  model = AutoModelForSequenceClassification.from_pretrained('{output_path}')")
        print("="*80)
    
    return quantized_model


if __name__ == "__main__":
    model = main()
