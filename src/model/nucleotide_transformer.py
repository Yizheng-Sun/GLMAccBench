"""Nucleotide Transformer model implementation."""

import os
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from huggingface_hub import snapshot_download

from .base import BaseModel, ModelConfig


@dataclass
class NucleotideTransformerConfig(ModelConfig):
    """Configuration specific to Nucleotide Transformer models."""
    
    model_name: str = None
    max_length: int = None
    reinitialize_classifier: bool = None


class NucleotideTransformer(BaseModel):
    """Implementation of Nucleotide Transformer model."""
    
    def __init__(self, config: NucleotideTransformerConfig):
        """
        Initialize the Nucleotide Transformer.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.config: NucleotideTransformerConfig = config
        self.model_path = config.model_path
        
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the nucleotide transformer model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Determine model source
        model_path = self.model_path
        
        print(f"Loading Nucleotide Transformer from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        
        # Configure torch dtype
        torch_dtype = self._get_torch_dtype()
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.config.num_labels
        )
        print(f"Model loaded successfully!")
        print(f"Model type: {type(self.model)}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model, self.tokenizer
    
    def load_quantized(self, quantized_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pre-quantized model.
        
        Args:
            quantized_path: Path to the quantized model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if not os.path.exists(quantized_path):
            raise FileNotFoundError(f"Quantized model not found at: {quantized_path}")
            
        print(f"Loading pre-quantized model from: {quantized_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(quantized_path)
        
        # Load quantized model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            quantized_path,
            torch_dtype="auto",
            device_map=self.config.device_map
        )
        
        # Check if model is quantized
        is_quantized = self._check_quantization()
        if is_quantized:
            print("✅ Model appears to be quantized")
        else:
            print("⚠️  Model loaded but quantization status unclear")
            
        return self.model, self.tokenizer
    
    def save(self, output_path: str) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            output_path: Directory to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
            
        os.makedirs(output_path, exist_ok=True)
        
        print(f"Saving model to: {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"✅ Model saved successfully")
    
    def _get_torch_dtype(self):
        """Get the appropriate torch dtype."""
        if self.config.torch_dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        elif self.config.torch_dtype == "float16":
            return torch.float16
        elif self.config.torch_dtype == "bfloat16":
            return torch.bfloat16
        elif self.config.torch_dtype == "float32":
            return torch.float32
        else:
            return "auto"
    
    def _check_quantization(self) -> bool:
        """Check if the model appears to be quantized."""
        if self.model is None:
            return False
            
        for module in self.model.modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                dtype_str = str(module.weight.dtype).lower()
                if 'int' in dtype_str or 'quantized' in dtype_str:
                    return True
                    
        return False
