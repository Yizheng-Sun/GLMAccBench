"""BitsAndBytes quantization implementation."""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from huggingface_hub import snapshot_download
import os

from .base import BaseQuantizer, QuantizationConfig
from ..model.base import BaseModel


@dataclass
class BnBConfig(QuantizationConfig):
    """Configuration for BitsAndBytes quantization."""
    
    method: str = "bnb"
    bits: int = 4  # 4 or 8
    bnb_4bit_compute_dtype: str = "bfloat16"  # For 4-bit
    bnb_4bit_use_double_quant: bool = True  # For 4-bit
    bnb_4bit_quant_type: str = "nf4"  # For 4-bit
    llm_int8_threshold: float = 6.0  # For 8-bit


class BitsAndBytesQuantizer(BaseQuantizer):
    """BitsAndBytes quantization implementation."""
    
    def __init__(self, config: BnBConfig):
        """
        Initialize the BitsAndBytes quantizer.
        
        Args:
            config: BitsAndBytes configuration
        """
        super().__init__(config)
        self.config: BnBConfig = config
        
    def quantize(self, 
                 model: BaseModel,
                 calibration_data: Optional[Any] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Quantize a model using BitsAndBytes.
        
        Args:
            model: The model to quantize
            calibration_data: Not used for BnB (quantization happens on-the-fly)
            
        Returns:
            Tuple of (quantized_model, tokenizer)
        """
        # Create BnB quantization config
        bnb_config = self._create_bnb_config()
        
        # Get model and tokenizer if not already loaded
        if model.model is None:
            model.load()
            
        # Get model path for reloading with quantization
        model_path = model._get_model_path() if hasattr(model, '_get_model_path') else model.config.model_name
        
        print(f"Applying BitsAndBytes {self.config.bits}-bit quantization...")
        
        # Follow the pattern from load_bnb_nucleotide_transformer.py
        # 1. Load model configuration
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = model.config.num_labels
        
        # 2. Determine weights location (local or download from HF)
        if os.path.exists(model_path):
            weights_location = model_path
            print(f"Using local model weights from {weights_location}")
        else:
            print(f"Downloading model weights for {model_path}...")
            weights_location = snapshot_download(repo_id=model_path)
        
        # 3. Create empty model from config
        empty_model = AutoModelForSequenceClassification.from_config(config)
        
        # 4. Load and quantize using accelerate
        print(f"Loading and quantizing model with {self.config.bits}-bit precision...")
        quantized_model = load_and_quantize_model(
            empty_model,
            weights_location=weights_location,
            bnb_quantization_config=bnb_config
        )
        
        # Prepare for training if needed
        if hasattr(quantized_model, 'gradient_checkpointing_enable'):
            quantized_model.gradient_checkpointing_enable()
            
        # Make sure model is in training mode for proper gradients
        quantized_model.train()
        
        print(f"✅ Model quantized with BitsAndBytes {self.config.bits}-bit")
        
        # Clean up original model to get accurate memory footprint
        # Delete the original model and empty model references
        if model.model is not None:
            del model.model
            model.model = None
        if 'empty_model' in locals():
            del empty_model
        
        # Clear GPU cache for accurate measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print memory footprint (now only includes quantized model)
        memory_stats = self.get_memory_footprint(quantized_model)
        print(f"Memory footprint: {memory_stats}")
        
        return quantized_model, model.tokenizer
    
    def load_quantized(self, model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pre-quantized BitsAndBytes model.
        
        Args:
            model_path: Path to the quantized model
            
        Returns:
            Tuple of (quantized_model, tokenizer)
        """
        print(f"Loading BitsAndBytes quantized model from: {model_path}")
        
        # Create BnB config for loading
        bnb_config = self._create_bnb_config()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Follow the pattern from load_bnb_nucleotide_transformer.py
        # 1. Load model configuration
        config = AutoConfig.from_pretrained(model_path)
        
        # 2. Determine weights location
        if os.path.exists(model_path):
            weights_location = model_path
        else:
            weights_location = snapshot_download(repo_id=model_path)
        
        # 3. Create empty model from config
        empty_model = AutoModelForSequenceClassification.from_config(config)
        
        # 4. Load and quantize using accelerate
        model = load_and_quantize_model(
            empty_model,
            weights_location=weights_location,
            bnb_quantization_config=bnb_config
        )
        
        # Clean up empty model reference
        if 'empty_model' in locals():
            del empty_model
        
        # Clear GPU cache for accurate memory state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ BitsAndBytes quantized model loaded")
        
        return model, tokenizer
    
    def _create_bnb_config(self) -> BnbQuantizationConfig:
        """Create BitsAndBytes quantization configuration."""
        if self.config.bits == 4:
            # Parse compute dtype
            compute_dtype = torch.bfloat16
            if self.config.bnb_4bit_compute_dtype == "float16":
                compute_dtype = torch.float16
            elif self.config.bnb_4bit_compute_dtype == "float32":
                compute_dtype = torch.float32
                
            return BnbQuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type
            )
        elif self.config.bits == 8:
            return BnbQuantizationConfig(
                load_in_8bit=True,
                llm_int8_threshold=self.config.llm_int8_threshold
            )
        else:
            raise ValueError(f"BitsAndBytes only supports 4-bit or 8-bit quantization, got {self.config.bits}")
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """Get information about the quantization configuration."""
        info = {
            "method": "BitsAndBytes",
            "bits": self.config.bits,
        }
        
        if self.config.bits == 4:
            info.update({
                "compute_dtype": self.config.bnb_4bit_compute_dtype,
                "use_double_quant": self.config.bnb_4bit_use_double_quant,
                "quant_type": self.config.bnb_4bit_quant_type
            })
        else:  # 8-bit
            info.update({
                "llm_int8_threshold": self.config.llm_int8_threshold
            })
            
        return info
