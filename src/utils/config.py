"""Configuration utility functions."""

import json
import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import asdict, is_dataclass


def load_config(config_path: str, format: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        format: File format ('json', 'yaml', or None for auto-detect)
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    # Auto-detect format if not specified
    if format is None:
        _, ext = os.path.splitext(config_path)
        format = ext.lstrip('.').lower()
        
    with open(config_path, 'r') as f:
        if format in ['json']:
            return json.load(f)
        elif format in ['yaml', 'yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {format}")


def save_config(config: Any, config_path: str, format: Optional[str] = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration object (dict or dataclass)
        config_path: Path to save configuration
        format: File format ('json', 'yaml', or None for auto-detect)
    """
    # Convert dataclass to dict if needed
    if is_dataclass(config):
        config = asdict(config)
    elif hasattr(config, 'to_dict'):
        config = config.to_dict()
        
    # Auto-detect format if not specified
    if format is None:
        _, ext = os.path.splitext(config_path)
        format = ext.lstrip('.').lower()
        
    # Create directory if needed
    os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
    
    with open(config_path, 'w') as f:
        if format in ['json']:
            json.dump(config, f, indent=2)
        elif format in ['yaml', 'yml']:
            yaml.safe_dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {format}")
            
    print(f"Configuration saved to: {config_path}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge (later ones override earlier ones)
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        if config:
            result = _deep_merge(result, config)
            
    return result


def _deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge (overrides dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result


def validate_config(config: Dict[str, Any], required_fields: list) -> None:
    """
    Validate that required fields are present in config.
    
    Args:
        config: Configuration dictionary
        required_fields: List of required field paths (e.g., ['model.name', 'training.batch_size'])
        
    Raises:
        ValueError: If required fields are missing
    """
    missing = []
    
    for field_path in required_fields:
        parts = field_path.split('.')
        current = config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                missing.append(field_path)
                break
            current = current[part]
            
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")
