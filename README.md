# GLMAccBench

A modular framework for training and quantizing genomic language models with support for multiple quantization methods and easy extensibility.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for models, quantization, training, and evaluation
- **Multiple Quantization Methods**: Support for BitsAndBytes, W4A4, W4A16, W8A8, GPTQ, and AWQ quantization
- **Easy Extensibility**: Simple to add new models and quantization methods through well-defined interfaces
- **Configuration Management**: YAML-based configuration system for reproducible experiments
- **Comprehensive Evaluation**: Built-in evaluation and benchmarking capabilities
- **Memory Efficient**: Optimized for training and inference with large genomic models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GLMAccBench.git
cd GLMAccBench

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### 1. Download Models and Datasets

```bash
# Download model
python scripts/download.py --model InstaDeepAI/nucleotide-transformer-2.5b-1000g

# Download dataset
python scripts/download.py --dataset InstaDeepAI/nucleotide_transformer_downstream_tasks_revised

### 2. Train a Model
# Basic training
accelerate launch scripts/train_nucleotide_transformer.py
```
### Training hyperparameters can be found: 
./scripts/train_nucleotide_transformer.py (line 15-20)

```bash
DATASET_PATH = "./datasets/nucleotide_transformer_downstream_tasks_revised"
MODEL_PATH = "./models/InstaDeepAI/nucleotide-transformer-2.5b-1000g"
MODEL_ID = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"  # Fallback
QUANTIZATION_TYPE = "none"  # Can be "4bit", "8bit", or "none"
SAVE_PATH = f"./trained_models/"
TRAINING_TASK = [] # Leave empty to train all tasks, or ["promoter_all", "enhancer_all"] for the specific tasks you want to train 

```


```bash
### 3. Quantize a Model
# BitsAndBytes 4-bit quantization
python scripts/quantize.py --method bnb --bits 4 --model-path <path_to_model folder>
e.g. python scripts/quantize.py --method bnb --bits 4 --model-path GLMAccBench/results/checkpoint-1521

# W4A16 quantization with LLMCompressor
python scripts/quantize.py --method w4a16 --output quantized_models/w4a16

# GPTQ quantization with custom settings
python scripts/quantize.py --method gptq --bits 4 --group-size 128

# Load pre-quantized model
python scripts/quantize.py --method w4a16 --quantized-model-path ./quantized_models/w4a16
```

### 4. Evaluate Models

```bash
# Evaluate a trained model
python scripts/evaluate.py --model-path ./results/checkpoint-1000

# Evaluate a quantized model
python scripts/evaluate.py --quantized-model-path ./quantized_models/w4a16

# Compare multiple models
python scripts/evaluate.py --compare model1_path model2_path model3_path

# Benchmark on multiple datasets
python scripts/evaluate.py --benchmark dataset1 dataset2 --model-path ./results/best_model
```

## Adding New Models

The framework uses a modular architecture that makes it easy to add new genomic language models. All models must inherit from the `BaseModel` class and implement its abstract methods.

### Step 1: Create Model Configuration

First, create a configuration class that inherits from `ModelConfig`:

```python
# src/models/my_model.py
from src.models.base import BaseModel, ModelConfig
from dataclasses import dataclass
from typing import Optional

@dataclass
class MyModelConfig(ModelConfig):
    """Configuration for MyModel."""
    
    # Inherit base fields: model_name, model_path, num_labels, max_length, etc.
    model_name: str = "my-organization/my-genomic-model"
    max_length: int = 1024  # Adjust based on your model's capabilities
    
    # Add model-specific configuration parameters
    special_param: str = "default_value"
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Model architecture parameters (if needed)
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None
```

### Step 2: Implement the Model Class

Create your model class by inheriting from `BaseModel`. The key methods you must implement are:

- `load()`: Load model and tokenizer from HuggingFace Hub or local path
- `save()`: Save model and tokenizer to specified directory  
- `download_from_hub()`: Download model from HuggingFace Hub (optional)

```python
import os
from typing import Tuple, Optional
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from huggingface_hub import snapshot_download

class MyModel(BaseModel):
    """Implementation of MyModel for genomic sequence classification."""
    
    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        self.config: MyModelConfig = config
    
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model and tokenizer."""
        model_path = self._get_model_path()
        print(f"Loading MyModel from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir
        )
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.config.num_labels,
            torch_dtype=self._get_torch_dtype(),
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir
        )
        
        # Initialize classifier weights if needed
        if hasattr(self.config, 'reinitialize_classifier') and self.config.reinitialize_classifier:
            print("Reinitializing classifier weights...")
            self.initialize_classifier_weights()
        
        print(f"✅ Model loaded successfully!")
        return self.model, self.tokenizer
    
    def save(self, output_path: str) -> None:
        """Save the model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving model to: {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("✅ Model saved successfully")
    
    def _get_model_path(self) -> str:
        """Determine the model path to use."""
        if self.config.model_path and os.path.exists(self.config.model_path):
            return self.config.model_path
        return self.config.model_name
    
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
```

### Step 3: Register Your Model

Add your model to the module's `__init__.py` file:

```python
# src/models/__init__.py
from .base import BaseModel, ModelConfig
from .nucleotide_transformer import NucleotideTransformer, NucleotideTransformerConfig
from .my_model import MyModel, MyModelConfig  # Add your model

__all__ = [
    'BaseModel',
    'ModelConfig',
    'NucleotideTransformer',
    'NucleotideTransformerConfig',
    'MyModel',           # Add your model
    'MyModelConfig'      # Add your config
]
```

### Step 4: Update Scripts (Optional)

If you want your model to be accessible from the command-line scripts, update the model selection logic:

```python
# In scripts/train.py, scripts/quantize.py, etc.
def get_model(model_type: str, config_dict: dict):
    """Get the appropriate model based on type."""
    if model_type == "nucleotide-transformer":
        config = NucleotideTransformerConfig(**config_dict)
        return NucleotideTransformer(config)
    elif model_type == "my-model":  # Add your model
        config = MyModelConfig(**config_dict)
        return MyModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### Step 5: Create Configuration Files (Optional)

Create YAML configuration files for your model:

```yaml
# config/models/my_model.yaml
model_name: "my-organization/my-genomic-model"
num_labels: 2
max_length: 1024
special_param: "custom_value"
layer_norm_eps: 1e-12
hidden_dropout_prob: 0.1
reinitialize_classifier: true
torch_dtype: "auto"
device_map: "auto"
```

### Usage Example

```python
from src.models import MyModel, MyModelConfig

# Create configuration
config = MyModelConfig(
    model_name="my-organization/my-genomic-model",
    num_labels=3,
    max_length=2048,
    special_param="custom_setting"
)

# Initialize and load model
model = MyModel(config)
model_obj, tokenizer = model.load()

# Use the model for inference or training
inputs = tokenizer("ATCGATCGATCG", return_tensors="pt")
outputs = model_obj(**inputs)
```

```bash
# Command line usage
python scripts/train.py --model my-model --model-name "my-organization/my-genomic-model"
python scripts/quantize.py --model-name "my-organization/my-genomic-model" --method bnb
```

## Supported Quantization Methods

GLMAccBench supports multiple state-of-the-art quantization methods optimized for genomic language models. Each method offers different trade-offs between model size, inference speed, and accuracy.

### BitsAndBytes (BnB)

**Method ID**: `bnb`

BitsAndBytes provides efficient 4-bit and 8-bit quantization with minimal accuracy loss. It's particularly well-suited for memory-constrained environments and supports both training and inference.

**Features:**
- **4-bit quantization**: Uses NF4 (Normal Float 4) format for optimal precision
- **8-bit quantization**: Uses INT8 with outlier detection
- **Dynamic quantization**: No calibration data required
- **Training support**: Supports QLoRA (Quantized Low-Rank Adaptation)

**Configuration Parameters:**
```python
@dataclass
class BnBConfig(QuantizationConfig):
    method: str = "bnb"
    bits: int = 4                           # 4 or 8 bits
    bnb_4bit_compute_dtype: str = "bfloat16" # Compute dtype for 4-bit
    bnb_4bit_use_double_quant: bool = True   # Double quantization for better compression
    bnb_4bit_quant_type: str = "nf4"        # Quantization type: "fp4" or "nf4"
    llm_int8_threshold: float = 6.0         # Threshold for 8-bit outlier detection
```

**Usage Examples:**
```bash
# 4-bit quantization with NF4
python scripts/quantize.py --method bnb --bits 4

# 8-bit quantization
python scripts/quantize.py --method bnb --bits 8

# Custom configuration
python scripts/quantize.py --method bnb --bits 4 --bnb-4bit-quant-type fp4
```

**Best for:** Memory-constrained inference, training with QLoRA, quick quantization without calibration data.

### W4A16 (Weight-4bit, Activation-16bit)

**Method ID**: `w4a16`

W4A16 quantizes model weights to 4-bit while keeping activations in 16-bit precision. This provides excellent inference speedup with minimal accuracy degradation.

**Features:**
- **Weight quantization**: 4-bit weights for significant memory reduction
- **Full precision activations**: 16-bit activations maintain model quality
- **Group-wise quantization**: Supports different group sizes for optimal trade-offs
- **Calibration-based**: Uses calibration data for optimal quantization parameters

**Configuration Parameters:**
```python
@dataclass
class W4A16Config(QuantizationConfig):
    method: str = "w4a16"
    bits: int = 4
    calibration_samples: int = 512
    ignore: list = None  # Layers to skip quantization
```

**Usage Examples:**
```bash
# Basic W4A16 quantization
python scripts/quantize.py --method w4a16

# With custom calibration samples
python scripts/quantize.py --method w4a16 --calibration-samples 1024

# Ignore specific layers
python scripts/quantize.py --method w4a16 --ignore-layers embeddings classifier
```

**Best for:** Inference optimization, balanced memory/accuracy trade-off, deployment scenarios.

### W4A4 (Weight-4bit, Activation-4bit)

**Method ID**: `w4a4`

W4A4 provides maximum compression by quantizing both weights and activations to 4-bit precision. Offers the smallest model size but requires careful calibration.

**Features:**
- **Full 4-bit quantization**: Both weights and activations quantized
- **Maximum compression**: Smallest model size possible
- **Advanced calibration**: Requires high-quality calibration data
- **Hardware acceleration**: Optimized for specialized inference hardware

**Configuration Parameters:**
```python
@dataclass
class W4A4Config(QuantizationConfig):
    method: str = "w4a4"
    bits: int = 4
    calibration_samples: int = 512
    ignore: list = None
```

**Usage Examples:**
```bash
# Full 4-bit quantization
python scripts/quantize.py --method w4a4

# With high-quality calibration data
python scripts/quantize.py --method w4a4 --calibration-samples 2048
```

**Best for:** Extreme compression requirements, specialized hardware deployment, edge computing.

### W8A8 (Weight-8bit, Activation-8bit)

**Method ID**: `w8a8`

W8A8 uses SmoothQuant technique combined with 8-bit quantization for both weights and activations. Provides good compression with better accuracy preservation than 4-bit methods.

**Features:**
- **SmoothQuant integration**: Uses activation smoothing for better quantization
- **8-bit precision**: Better accuracy than 4-bit methods
- **Balanced trade-off**: Good compression with minimal accuracy loss
- **Robust quantization**: Less sensitive to calibration data quality

**Configuration Parameters:**
```python
@dataclass
class W8A8Config(QuantizationConfig):
    method: str = "w8a8"
    bits: int = 8
    smoothquant_alpha: float = 0.8  # SmoothQuant smoothing parameter
    calibration_samples: int = 512
    ignore: list = None
```

**Usage Examples:**
```bash
# W8A8 with SmoothQuant
python scripts/quantize.py --method w8a8

# Custom smoothing parameter
python scripts/quantize.py --method w8a8 --smoothquant-alpha 0.5

# More calibration samples for better quality
python scripts/quantize.py --method w8a8 --calibration-samples 1024
```

**Best for:** High-accuracy requirements, production deployments, when 4-bit accuracy is insufficient.

### GPTQ (Gradient-based Post-Training Quantization)

**Method ID**: `gptq`

GPTQ uses gradient information to optimize quantization parameters, providing excellent accuracy preservation with 4-bit quantization.

**Features:**
- **Gradient-based optimization**: Uses second-order information for optimal quantization
- **Group-wise quantization**: Configurable group sizes for fine-tuned compression
- **High accuracy**: Minimal accuracy loss compared to naive quantization
- **Flexible bit widths**: Supports 2, 3, 4, and 8-bit quantization

**Configuration Parameters:**
```python
@dataclass
class GPTQConfig(QuantizationConfig):
    method: str = "gptq"
    bits: int = 4
    group_size: int = 128        # Group size for quantization
    damp_percent: float = 0.01   # Damping factor for numerical stability
    calibration_samples: int = 512
    ignore: list = None
```

**Usage Examples:**
```bash
# Standard GPTQ 4-bit
python scripts/quantize.py --method gptq --bits 4

# Custom group size and damping
python scripts/quantize.py --method gptq --group-size 64 --damp-percent 0.005

# 3-bit quantization for maximum compression
python scripts/quantize.py --method gptq --bits 3 --group-size 32
```

**Best for:** High-accuracy quantization, research applications, when calibration data is available.

### AWQ (Activation-aware Weight Quantization)

**Method ID**: `awq`

AWQ protects salient weights based on activation statistics, providing excellent accuracy with 4-bit quantization by keeping important weights at higher precision.

**Features:**
- **Activation-aware**: Uses activation statistics to identify important weights
- **Salient weight protection**: Keeps critical weights at higher precision
- **Excellent accuracy**: Superior accuracy compared to naive 4-bit quantization
- **Efficient inference**: Optimized for fast inference on various hardware

**Configuration Parameters:**
```python
@dataclass
class AWQConfig(QuantizationConfig):
    method: str = "awq"
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True      # Use zero-point quantization
    calibration_samples: int = 512
    ignore: list = None
```

**Usage Examples:**
```bash
# Standard AWQ quantization
python scripts/quantize.py --method awq

# Custom group size
python scripts/quantize.py --method awq --group-size 64

# Without zero-point quantization
python scripts/quantize.py --method awq --zero-point false
```

**Best for:** High-accuracy 4-bit quantization, production deployments, when preserving model quality is critical.

### Comparison Table

| Method | Bits | Calibration Required | Training Support | Best Use Case |
|--------|------|---------------------|------------------|---------------|
| **BnB** | 4/8 | No | Yes (QLoRA) | Quick quantization, training |
| **W4A16** | 4 | Yes | No | Inference optimization |
| **W4A4** | 4 | Yes | No | Maximum compression |
| **W8A8** | 8 | Yes | No | High accuracy needs |
| **GPTQ** | 4 | Yes | No | Research, high accuracy |
| **AWQ** | 4 | Yes | No | Production, quality critical |

### Choosing the Right Method

**For quick experimentation:**
```bash
python scripts/quantize.py --method bnb --bits 4
```

**For production deployment:**
```bash
python scripts/quantize.py --method awq --calibration-samples 1024
```

**For maximum compression:**
```bash
python scripts/quantize.py --method w4a4 --calibration-samples 2048
```

**For training/fine-tuning:**
```bash
python scripts/quantize.py --method bnb --bits 4  # Then use for QLoRA training
```

**For high-accuracy requirements:**
```bash
python scripts/quantize.py --method gptq --bits 4 --group-size 128
```

## Adding New Quantization Methods

The framework supports multiple quantization methods through a unified interface. All quantization methods must inherit from the `BaseQuantizer` class and implement its abstract methods.

### Step 1: Create Quantization Configuration

First, create a configuration class that inherits from `QuantizationConfig`:

```python
# src/quantization/my_quantizer.py
from src.quantization.base import BaseQuantizer, QuantizationConfig
from dataclasses import dataclass
from typing import Optional, List, Any

@dataclass
class MyQuantConfig(QuantizationConfig):
    """Configuration for MyQuantizer."""
    
    # Inherit base fields: method, bits, calibration_samples, output_path
    method: str = "my_quant"
    bits: int = 4
    
    # Add quantization-specific parameters
    alpha: float = 0.5
    beta: float = 0.1
    group_size: int = 128
    symmetric: bool = True
    
    # Advanced parameters
    calibration_method: str = "minmax"
    percentile: float = 99.9
    ignore_layers: Optional[List[str]] = None
    
    # Optimization parameters
    max_iterations: int = 1000
    tolerance: float = 1e-6
```

### Step 2: Implement the Quantizer Class

Create your quantizer class by inheriting from `BaseQuantizer`. The key methods you must implement are:

- `quantize()`: Apply quantization to a loaded model
- `load_quantized()`: Load a pre-quantized model from disk

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional, Any, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from ..models.base import BaseModel

class MyQuantizer(BaseQuantizer):
    """Custom quantization implementation."""
    
    def __init__(self, config: MyQuantConfig):
        super().__init__(config)
        self.config: MyQuantConfig = config
        self.calibration_stats = {}
        self.quantization_params = {}
    
    def quantize(self, 
                 model: BaseModel,
                 calibration_data: Optional[Any] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Quantize a model using your custom method."""
        
        # Load model if not already loaded
        if model.model is None:
            model.load()
        
        print(f"Applying {self.config.method.upper()} quantization...")
        
        # Step 1: Collect calibration statistics
        if calibration_data is not None:
            print("Collecting calibration statistics...")
            self._collect_calibration_stats(model.model, calibration_data, model.tokenizer)
        
        # Step 2: Compute quantization parameters
        print("Computing quantization parameters...")
        self._compute_quantization_params(model.model)
        
        # Step 3: Apply quantization
        print("Applying quantization to model layers...")
        quantized_model = self._apply_quantization(model.model)
        
        # Step 4: Save if output path specified
        if self.config.output_path:
            self.save_quantized(quantized_model, model.tokenizer, self.config.output_path)
        
        print(f"✅ Model quantized with {self.config.method.upper()}")
        return quantized_model, model.tokenizer
    
    def load_quantized(self, model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a pre-quantized model."""
        print(f"Loading {self.config.method.upper()} quantized model from: {model_path}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Verify quantization status
        is_quantized = self._verify_quantization(model)
        if is_quantized:
            print(f"✅ {self.config.method.upper()} quantized model loaded successfully")
        else:
            print(f"⚠️  Model loaded but quantization status unclear")
        
        return model, tokenizer
    
    def _collect_calibration_stats(self, model: PreTrainedModel, calibration_data: Any, tokenizer: PreTrainedTokenizer):
        """Collect statistics from calibration data."""
        model.eval()
        calibration_samples = self.prepare_calibration_data(
            calibration_data, tokenizer, self.config.calibration_samples
        )
        
        # Your calibration logic here
        # Example: collect activation statistics, compute sensitivity, etc.
        print(f"Collected statistics for calibration")
    
    def _compute_quantization_params(self, model: PreTrainedModel):
        """Compute quantization parameters for each layer."""
        self.quantization_params = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not self._should_ignore_layer(name):
                # Compute scale and zero-point based on your method
                weight = module.weight.data
                
                if self.config.symmetric:
                    max_val = weight.abs().max()
                    scale = max_val / (2**(self.config.bits-1) - 1)
                    zero_point = 0
                else:
                    min_val, max_val = weight.min(), weight.max()
                    scale = (max_val - min_val) / (2**self.config.bits - 1)
                    zero_point = -min_val / scale
                
                self.quantization_params[name] = {
                    'scale': scale, 'zero_point': zero_point, 'bits': self.config.bits
                }
    
    def _apply_quantization(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply quantization to the model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.quantization_params:
                params = self.quantization_params[name]
                
                # Quantize weights
                quantized_weight = self._quantize_tensor(
                    module.weight.data, params['scale'], params['zero_point'], params['bits']
                )
                module.weight.data = quantized_weight
        
        return model
    
    def _quantize_tensor(self, tensor: torch.Tensor, scale: float, zero_point: float, bits: int) -> torch.Tensor:
        """Quantize a tensor."""
        quantized = torch.round(tensor / scale + zero_point)
        
        if self.config.symmetric:
            quantized = torch.clamp(quantized, -(2**(bits-1)), 2**(bits-1)-1)
        else:
            quantized = torch.clamp(quantized, 0, 2**bits-1)
        
        # Dequantize for fake quantization
        dequantized = (quantized - zero_point) * scale
        return dequantized
    
    def _should_ignore_layer(self, layer_name: str) -> bool:
        """Check if a layer should be ignored during quantization."""
        ignore_patterns = self.config.ignore_layers or ['embeddings', 'classifier', 'lm_head']
        return any(pattern in layer_name for pattern in ignore_patterns)
    
    def _verify_quantization(self, model: PreTrainedModel) -> bool:
        """Verify that the model is properly quantized."""
        # Implementation depends on your quantization method
        return True
```

### Step 3: Register Your Quantizer

Add your quantizer to the module's `__init__.py` file:

```python
# src/quantization/__init__.py
from .base import BaseQuantizer, QuantizationConfig
from .bnb import BitsAndBytesQuantizer, BnBConfig
from .llm_compressor import (
    LLMCompressorQuantizer,
    W4A4Config, W4A16Config, W8A8Config, GPTQConfig, AWQConfig
)
from .my_quantizer import MyQuantizer, MyQuantConfig  # Add your quantizer

__all__ = [
    'BaseQuantizer', 'QuantizationConfig',
    'BitsAndBytesQuantizer', 'BnBConfig',
    'LLMCompressorQuantizer',
    'W4A4Config', 'W4A16Config', 'W8A8Config', 'GPTQConfig', 'AWQConfig',
    'MyQuantizer', 'MyQuantConfig'  # Add your quantizer
]
```

### Step 4: Update Quantization Script

Add your quantizer to the script's selection logic:

```python
# In scripts/quantize.py - update get_quantizer function
def get_quantizer(method: str, bits: int, config_dict: dict):
    """Get the appropriate quantizer based on method."""
    
    # ... existing methods ...
    
    elif method == "my_quant":
        config = MyQuantConfig(
            bits=bits,
            calibration_samples=config_dict.get('calibration_samples', 512),
            output_path=config_dict.get('output_path'),
            alpha=config_dict.get('alpha', 0.5),
            group_size=config_dict.get('group_size', 128),
            symmetric=config_dict.get('symmetric', True),
            ignore_layers=config_dict.get('ignore_layers')
        )
        return MyQuantizer(config)
    
    else:
        raise ValueError(f"Unknown quantization method: {method}")

# Update the argument parser choices
parser.add_argument("--method", type=str, required=True,
                   choices=["bnb", "w4a4", "w4a16", "w8a8", "gptq", "awq", "my_quant"],
                   help="Quantization method")
```

### Step 5: Create Configuration Files (Optional)

Create YAML configuration files for your quantizer:

```yaml
# config/quantization/my_quant.yaml
method: "my_quant"
bits: 4
calibration_samples: 512
alpha: 0.5
beta: 0.1
group_size: 128
symmetric: true
calibration_method: "minmax"
percentile: 99.9
max_iterations: 1000
tolerance: 1e-6
ignore_layers:
  - "embeddings"
  - "classifier" 
  - "lm_head"
```

### Usage Example

```bash
# Command line usage
python scripts/quantize.py --method my_quant --bits 4 --alpha 0.7 --group-size 64

# Using configuration file
python scripts/quantize.py --config config/quantization/my_quant.yaml
```

```python
# Programmatic usage
from src.models import NucleotideTransformer, NucleotideTransformerConfig
from src.quantization import MyQuantizer, MyQuantConfig

# Create model
model_config = NucleotideTransformerConfig(
    model_name="InstaDeepAI/nucleotide-transformer-2.5b-1000g"
)
model = NucleotideTransformer(model_config)

# Create quantizer
quant_config = MyQuantConfig(bits=4, alpha=0.6, group_size=128, symmetric=True)
quantizer = MyQuantizer(quant_config)

# Quantize
quantized_model, tokenizer = quantizer.quantize(model, calibration_data)
```

### Best Practices for Custom Quantization

1. **Handle Different Data Types**: Ensure your quantizer works with different model architectures and data types.

2. **Memory Efficiency**: Consider memory usage during calibration and quantization.

3. **Numerical Stability**: Add checks for numerical stability and edge cases.

4. **Validation**: Implement thorough validation to ensure quantization quality.

5. **Documentation**: Document your quantization method's assumptions and limitations.

6. **Testing**: Create unit tests for your quantization components.

7. **Calibration Data**: Handle different calibration dataset formats gracefully.

8. **Error Handling**: Provide clear error messages for common issues.

## Configuration Files

The framework uses YAML configuration files for reproducible experiments:

```yaml
# config/training/custom.yaml
output_dir: "./results/experiment1"
num_train_epochs: 5
per_device_train_batch_size: 16
learning_rate: 2e-5
freeze_backbone: true
```

Use configurations with any script:

```bash
python scripts/train.py --config config/training/custom.yaml
```

## Advanced Usage

### Multi-GPU Training

The framework automatically uses all available GPUs:

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --model nucleotide-transformer

# Distributed training with accelerate
accelerate launch scripts/train.py --model nucleotide-transformer
```

### Mixed Precision Training

Enable mixed precision for faster training:

```bash
# FP16 training
python scripts/train.py --fp16

# BF16 training (recommended for A100 GPUs)
python scripts/train.py --bf16
```

### Gradient Checkpointing

Reduce memory usage with gradient checkpointing:

```bash
python scripts/train.py --gradient-checkpointing
```

## API Documentation

### Models Module

- `BaseModel`: Abstract base class for all models
- `NucleotideTransformer`: Implementation for nucleotide transformer models
- `ModelConfig`: Configuration dataclass for models

### Quantization Module

- `BaseQuantizer`: Abstract base class for quantization methods
- `BitsAndBytesQuantizer`: BitsAndBytes quantization (4-bit and 8-bit)
- `LLMCompressorQuantizer`: Supports W4A4, W4A16, W8A8, GPTQ, AWQ

### Training Module

- `GenomicTrainer`: Main trainer class for genomic models
- `TrainingConfig`: Configuration for training
- `TensorBoardCallback`: TensorBoard logging callback
- `GradientClippingCallback`: Gradient clipping and monitoring

### Evaluation Module

- `Evaluator`: Model evaluation and benchmarking
- `EvaluationConfig`: Configuration for evaluation

### Data Module

- `DataLoader`: Dataset loading and preprocessing
- `DataConfig`: Configuration for data handling

## Troubleshooting

### Out of Memory Errors

- Reduce batch size: `--batch-size 4`
- Enable gradient checkpointing: `--gradient-checkpointing`
- Use quantization: `--method bnb --bits 4`
- Use gradient accumulation: `--gradient-accumulation 4`

### Slow Training

- Enable mixed precision: `--fp16` or `--bf16`
- Use multiple GPUs
- Increase batch size if memory allows
- Use faster data loading: increase `num_proc` in data config

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{glmaccbench,
  title = {GLMAccBench: A Modular Framework for Genomic Language Models},
  year = {2024},
  url = {https://github.com/yourusername/GLMAccBench}
}
```

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.
