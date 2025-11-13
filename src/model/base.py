"""Base classes for models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class ModelConfig:
    """Base configuration for models."""
    
    model_name: str
    model_path: Optional[str] = None  # Local path to model
    num_labels: int = 2
    max_length: int = 512
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        
    @abstractmethod
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        pass
    
    @abstractmethod
    def save(self, output_path: str) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            output_path: Directory to save the model
        """
        pass
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """
        Get memory footprint of the model.
        
        Returns:
            Dictionary with memory statistics
        """
        if self.model is None:
            return {"error": "Model not loaded"}
            
        param_count = sum(p.numel() for p in self.model.parameters())
        param_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        
        stats = {
            "parameter_count": param_count,
            "parameter_size_mb": param_size_mb
        }
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            
        return stats
    
    def initialize_classifier_weights(self, std_dense: float = 0.02, std_out: float = 0.01) -> None:
        """
        Initialize classifier weights to prevent gradient explosion.
        
        Args:
            std_dense: Standard deviation for dense layer initialization
            std_out: Standard deviation for output layer initialization
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        if hasattr(self.model, 'classifier'):
            classifier = self.model.classifier
            
            # Handle different classifier structures
            if hasattr(classifier, 'dense') and hasattr(classifier, 'out_proj'):
                # EsmClassificationHead structure
                if hasattr(classifier.dense, 'weight'):
                    torch.nn.init.normal_(classifier.dense.weight, mean=0.0, std=std_dense)
                if hasattr(classifier.dense, 'bias') and classifier.dense.bias is not None:
                    torch.nn.init.zeros_(classifier.dense.bias)
                    
                if hasattr(classifier.out_proj, 'weight'):
                    torch.nn.init.normal_(classifier.out_proj.weight, mean=0.0, std=std_out)
                if hasattr(classifier.out_proj, 'bias') and classifier.out_proj.bias is not None:
                    torch.nn.init.zeros_(classifier.out_proj.bias)
                    
            elif hasattr(classifier, 'weight'):
                # Simple Linear classifier
                torch.nn.init.normal_(classifier.weight, mean=0.0, std=std_out)
                if hasattr(classifier, 'bias') and classifier.bias is not None:
                    torch.nn.init.zeros_(classifier.bias)
    
    def freeze_backbone(self, freeze_classifier: bool = False) -> Tuple[int, int]:
        """
        Freeze backbone parameters, optionally freeze classifier too.
        
        Args:
            freeze_classifier: Whether to freeze the classifier as well
            
        Returns:
            Tuple of (frozen_params, trainable_params)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        frozen_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name and not freeze_classifier:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
                
        return frozen_params, trainable_params


