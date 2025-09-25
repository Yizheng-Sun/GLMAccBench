"""LLM Compressor quantization implementations."""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer

from .base import BaseQuantizer, QuantizationConfig
from ..models.base import BaseModel


@dataclass
class W4A4Config(QuantizationConfig):
    """Configuration for W4A4 quantization."""
    method: str = "w4a4"
    bits: int = 4
    ignore: list = None  # Layers to ignore


@dataclass  
class W4A16Config(QuantizationConfig):
    """Configuration for W4A16 quantization."""
    method: str = "w4a16"
    bits: int = 4
    ignore: list = None


@dataclass
class W8A8Config(QuantizationConfig):
    """Configuration for W8A8 quantization with SmoothQuant."""
    method: str = "w8a8"
    bits: int = 8
    smoothquant_alpha: float = 0.8
    ignore: list = None


@dataclass
class GPTQConfig(QuantizationConfig):
    """Configuration for GPTQ quantization."""
    method: str = "gptq"
    bits: int = 4
    group_size: int = 128
    damp_percent: float = 0.01
    ignore: list = None


@dataclass
class AWQConfig(QuantizationConfig):
    """Configuration for AWQ quantization."""
    method: str = "awq"
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    ignore: list = None


class LLMCompressorQuantizer(BaseQuantizer):
    """LLM Compressor quantization implementation."""
    
    def __init__(self, config: QuantizationConfig):
        """
        Initialize the LLM Compressor quantizer.
        
        Args:
            config: Quantization configuration (W4A4, W4A16, W8A8, GPTQ, or AWQ)
        """
        super().__init__(config)
        
    def quantize(self,
                 model: BaseModel,
                 calibration_data: Optional[Any] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Quantize a model using LLM Compressor.
        
        Args:
            model: The model to quantize
            calibration_data: Calibration dataset for quantization
            
        Returns:
            Tuple of (quantized_model, tokenizer)
        """
        try:
            from llmcompressor import oneshot
            from llmcompressor.utils import dispatch_for_generation
        except ImportError:
            raise ImportError("llmcompressor is required for this quantization method. Install with: pip install llmcompressor")
        
        # Load model if not already loaded
        if model.model is None:
            model.load()
            
        # Prepare calibration data if provided
        calibration_samples = None
        if calibration_data is not None:
            calibration_samples = self.prepare_calibration_data(
                calibration_data, 
                model.tokenizer,
                self.config.calibration_samples
            )
            
        print(f"Applying {self.config.method.upper()} quantization...")
        
        # Create the appropriate modifier based on config type
        modifier = self._create_modifier()
        
        # Apply quantization
        quantized_model = oneshot(
            model=model.model,
            recipe=modifier,
            dataset=calibration_samples,
            max_samples=self.config.calibration_samples
        )
        
        # Enable generation mode if available
        try:
            quantized_model = dispatch_for_generation(quantized_model)
        except:
            pass  # Not all models support generation dispatch
            
        print(f"✅ Model quantized with {self.config.method.upper()}")
        
        # Save if output path specified
        if self.config.output_path:
            self.save_quantized(quantized_model, model.tokenizer, self.config.output_path)
            
        # Print memory footprint
        memory_stats = self.get_memory_footprint(quantized_model)
        print(f"Memory footprint: {memory_stats}")
        
        return quantized_model, model.tokenizer
    
    def load_quantized(self, model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pre-quantized LLM Compressor model.
        
        Args:
            model_path: Path to the quantized model
            
        Returns:
            Tuple of (quantized_model, tokenizer)
        """
        print(f"Loading {self.config.method.upper()} quantized model from: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load quantized model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Check quantization status
        is_quantized = self._check_quantization(model)
        if is_quantized:
            print(f"✅ {self.config.method.upper()} quantized model loaded")
        else:
            print(f"⚠️  Model loaded but quantization status unclear")
            
        return model, tokenizer
    
    def _create_modifier(self):
        """Create the appropriate modifier based on config type."""
        try:
            from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
            from llmcompressor.modifiers.awq import AWQModifier
        except ImportError:
            raise ImportError("llmcompressor modifiers not available")
            
        if isinstance(self.config, W4A4Config):
            return QuantizationModifier(
                targets="Linear",
                scheme="W4A4",
                ignore=self.config.ignore or ["esm.embeddings", "esm.contact_head", "lm_head", "classifier"]
            )
            
        elif isinstance(self.config, W4A16Config):
            return QuantizationModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=self.config.ignore or ["esm.embeddings", "esm.contact_head", "lm_head", "classifier"]
            )
            
        elif isinstance(self.config, W8A8Config):
            return [
                SmoothQuantModifier(
                    alpha=self.config.smoothquant_alpha,
                    ignore=self.config.ignore or ["esm.embeddings", "esm.contact_head", "lm_head", "classifier"]
                ),
                GPTQModifier(
                    targets="Linear",
                    bits=8,
                    ignore=self.config.ignore or ["esm.embeddings", "esm.contact_head", "lm_head", "classifier"]
                )
            ]
            
        elif isinstance(self.config, GPTQConfig):
            return GPTQModifier(
                targets="Linear",
                bits=self.config.bits,
                group_size=self.config.group_size,
                damp_percent=self.config.damp_percent,
                ignore=self.config.ignore or ["esm.embeddings", "esm.contact_head", "lm_head", "classifier"]
            )
            
        elif isinstance(self.config, AWQConfig):
            return AWQModifier(
                targets="Linear",
                bits=self.config.bits,
                group_size=self.config.group_size,
                zero_point=self.config.zero_point,
                ignore=self.config.ignore or ["esm.embeddings", "esm.contact_head", "lm_head", "classifier"]
            )
            
        else:
            raise ValueError(f"Unknown quantization config type: {type(self.config)}")
    
    def _check_quantization(self, model: PreTrainedModel) -> bool:
        """Check if the model appears to be quantized."""
        for module in model.modules():
            if hasattr(module, 'weight'):
                # Check for quantized weight attributes
                if hasattr(module.weight, 'quantized') and module.weight.quantized:
                    return True
                # Check for integer dtype
                if hasattr(module.weight, 'dtype'):
                    dtype_str = str(module.weight.dtype).lower()
                    if 'int' in dtype_str:
                        return True
                        
        return False
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """Get information about the quantization configuration."""
        info = {
            "method": self.config.method.upper(),
            "bits": self.config.bits,
        }
        
        if isinstance(self.config, W8A8Config):
            info["smoothquant_alpha"] = self.config.smoothquant_alpha
        elif isinstance(self.config, GPTQConfig):
            info.update({
                "group_size": self.config.group_size,
                "damp_percent": self.config.damp_percent
            })
        elif isinstance(self.config, AWQConfig):
            info.update({
                "group_size": self.config.group_size,
                "zero_point": self.config.zero_point
            })
            
        if hasattr(self.config, 'ignore') and self.config.ignore:
            info["ignored_layers"] = self.config.ignore
            
        return info
