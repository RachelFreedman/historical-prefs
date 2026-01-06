"""
Model utilities for historical preferences project.

Provides functions for loading ProgressGym historical LLMs,
including automatic version discovery from HuggingFace.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig, NAS_BASE, configure_environment


def find_latest_model_version(model_size: str, model_century: str) -> str:
    """
    Find the latest available model version on HuggingFace.
    
    Args:
        model_size: Model size ("8B" or "70B")
        model_century: Century code (e.g., "C013")
    
    Returns:
        Version string (e.g., "v0.2")
    """
    config = ModelConfig(size=model_size, century=model_century)
    api = HfApi()
    base_name = config.base_model_name
    
    try:
        models = api.list_models(author=config.hf_org, search=base_name)
        
        versions: list[tuple[float, str]] = []
        for model in models:
            model_id = getattr(model, 'id', None) or getattr(model, 'modelId', None)
            if model_id and base_name in model_id:
                version_part = model_id.split('-')[-1]
                if version_part.startswith('v'):
                    try:
                        version_num = float(version_part[1:])
                        versions.append((version_num, version_part))
                    except ValueError:
                        pass
        
        if versions:
            versions.sort(reverse=True)
            latest = versions[0][1]
            print(f"Found versions: {[v[1] for v in versions]}, using {latest}")
            return latest
        else:
            print(f"No matching model versions found, using {config.default_version}")
            return config.default_version
            
    except Exception as e:
        print(f"Could not query HuggingFace for model versions: {e}")
        print(f"Using default {config.default_version}")
        return config.default_version


def load_model_and_tokenizer(
    model_size: str, 
    model_century: str,
    device_map: str = "auto"
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        model_size: Model size ("8B" or "70B")
        model_century: Century code (e.g., "C013")
        device_map: Device mapping strategy for model loading
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Configure environment for HuggingFace cache
    configure_environment()
    
    config = ModelConfig(size=model_size, century=model_century)
    cache_dir = NAS_BASE / "hf_cache"
    
    # Find latest version
    version = find_latest_model_version(model_size, model_century)
    model_name = config.full_model_name(version)
    
    print(f"Loading model and tokenizer: {model_name}")
    print("This may take a while on first run as it downloads the model...")
    print(f"Cache directory: {cache_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir)
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=str(cache_dir)
    )
    
    print("Model loaded successfully.")
    return model, tokenizer

