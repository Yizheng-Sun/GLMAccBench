"""Base classes for quantization methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..models.base import BaseModel


@dataclass
class QuantizationConfig:
    """Base configuration for quantization."""
    
    method: str  # Quantization method name
    bits: int = 4  # Number of bits for quantization
    calibration_samples: int = 512  # Number of samples for calibration
    output_path: Optional[str] = None  # Path to save quantized model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BaseQuantizer(ABC):
    """Abstract base class for all quantizers."""
    
    def __init__(self, config: QuantizationConfig):
        """
        Initialize the quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config
        
    @abstractmethod
    def quantize(self, 
                 model: BaseModel,
                 calibration_data: Optional[Any] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Quantize a model.
        
        Args:
            model: The model to quantize
            calibration_data: Optional calibration dataset
            
        Returns:
            Tuple of (quantized_model, tokenizer)
        """
        pass
    
    @abstractmethod
    def load_quantized(self, model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pre-quantized model.
        
        Args:
            model_path: Path to the quantized model
            
        Returns:
            Tuple of (quantized_model, tokenizer)
        """
        pass
    
    def save_quantized(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_path: str) -> None:
        """
        Save a quantized model.
        
        Args:
            model: The quantized model
            tokenizer: The tokenizer
            output_path: Directory to save the model
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        print(f"Saving quantized model to: {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print("✅ Quantized model saved successfully")
    
    def get_memory_footprint(self, model: PreTrainedModel) -> Dict[str, float]:
        """
        Get memory footprint of the quantized model.
        
        Args:
            model: The quantized model
            
        Returns:
            Dictionary with memory statistics
        """
        param_count = sum(p.numel() for p in model.parameters())
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        stats = {
            "parameter_count": param_count,
            "parameter_size_mb": param_size_mb,
            "quantization_method": self.config.method,
            "quantization_bits": self.config.bits
        }
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            
        return stats
    
    def prepare_calibration_data(self, dataset, tokenizer, max_samples: Optional[int] = None) -> list:
        """
        Prepare calibration dataset for quantization.
        
        Args:
            dataset: The dataset to use for calibration
            tokenizer: The tokenizer to use
            max_samples: Maximum number of samples to use
            
        Returns:
            List of tokenized samples
        """
        max_samples = max_samples or self.config.calibration_samples
        
        texts = []
        for idx, item in enumerate(dataset):
            if idx >= max_samples:
                break
                
            # Handle different dataset formats
            if 'text' in item:
                text = item['text']
            elif 'sequence' in item:
                text = item['sequence']
            else:
                # Try to find the first string field
                for value in item.values():
                    if isinstance(value, str):
                        text = value
                        break
                else:
                    continue
                    
            texts.append(text)
        
        # Tokenize the texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return tokenized
