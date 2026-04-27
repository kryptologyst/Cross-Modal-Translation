"""Utility functions for cross-modal translation project."""

import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For PyTorch 2.0+ deterministic behavior
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def get_dtype(precision: str = "fp16") -> torch.dtype:
    """Get the appropriate torch dtype based on precision setting.
    
    Args:
        precision: Precision setting ("fp16", "fp32", "bf16").
        
    Returns:
        torch.dtype: The appropriate dtype.
    """
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else:
        return torch.float32


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes.
    
    Args:
        num: Number to format.
        
    Returns:
        str: Formatted number string.
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


def create_dir_if_not_exists(path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename.
        
    Returns:
        str: Safe filename.
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Dict containing configuration.
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save configuration.
    """
    import yaml
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
