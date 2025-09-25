"""Memory utility functions."""

import torch
from typing import Dict, Optional


def print_gpu_memory_usage(prefix: str = "") -> None:
    """
    Print current GPU memory usage.
    
    Args:
        prefix: Optional prefix for the printout
    """
    if torch.cuda.is_available():
        # Clear cache first for accurate measurement
        torch.cuda.empty_cache()
        
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # Convert to GB
        
        prefix_str = f"{prefix} - " if prefix else ""
        print(f"{prefix_str}GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    else:
        print("CUDA is not available")


def get_memory_stats() -> Dict[str, float]:
    """
    Get current memory statistics.
    
    Returns:
        Dictionary with memory statistics
    """
    stats = {}
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        stats['gpu_available'] = True
    else:
        stats['gpu_available'] = False
        
    return stats


def get_model_memory_footprint(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate the memory footprint of a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with memory statistics
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size in MB
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # Check for quantization
    is_quantized = False
    quantized_layers = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
            dtype_str = str(module.weight.dtype).lower()
            if 'int' in dtype_str or 'quantized' in dtype_str:
                is_quantized = True
                quantized_layers += 1
    
    stats = {
        'total_parameters': param_count,
        'trainable_parameters': trainable_count,
        'frozen_parameters': param_count - trainable_count,
        'parameter_size_mb': param_size_mb,
        'is_quantized': is_quantized,
        'quantized_layers': quantized_layers
    }
    
    # Add GPU memory if model is on CUDA
    if next(model.parameters(), None) is not None and next(model.parameters()).is_cuda:
        stats.update(get_memory_stats())
        
    return stats


def estimate_batch_size(model: torch.nn.Module, 
                       sequence_length: int = 512,
                       available_memory_gb: Optional[float] = None) -> int:
    """
    Estimate optimal batch size based on model size and available memory.
    
    Args:
        model: The model to analyze
        sequence_length: Expected sequence length
        available_memory_gb: Available GPU memory in GB (auto-detected if None)
        
    Returns:
        Estimated optimal batch size
    """
    if available_memory_gb is None and torch.cuda.is_available():
        # Get available memory (leave 2GB buffer)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        used_memory = torch.cuda.memory_allocated() / (1024**3)
        available_memory_gb = total_memory - used_memory - 2.0
        
    if available_memory_gb is None or available_memory_gb <= 0:
        return 1  # Default to batch size of 1
        
    # Estimate memory per sample (rough approximation)
    model_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4  # Assume float32
    
    # Check if model is quantized
    for module in model.modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
            if 'int8' in str(module.weight.dtype).lower():
                bytes_per_param = 1
                break
            elif 'int4' in str(module.weight.dtype).lower():
                bytes_per_param = 0.5
                break
                
    # Rough estimate: model size + activation memory
    model_memory_gb = (model_params * bytes_per_param) / (1024**3)
    
    # Estimate activation memory per sample (very rough)
    activation_memory_per_sample_mb = (sequence_length * 1024 * 4) / (1024 * 1024)  # Assume 1024 hidden dim
    activation_memory_per_sample_gb = activation_memory_per_sample_mb / 1024
    
    # Calculate batch size (conservative estimate)
    batch_size = int((available_memory_gb - model_memory_gb) / (activation_memory_per_sample_gb * 2))
    
    # Ensure at least batch size of 1
    return max(1, batch_size)
