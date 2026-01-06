"""
GPU utilities for historical preferences project.

Provides functions for discovering and selecting available GPUs
based on memory usage thresholds.
"""

import subprocess
from typing import Optional

from config import PROCESSING_CONFIG


def find_available_gpu(threshold: Optional[float] = None) -> int:
    """
    Find the first GPU with low memory usage.
    
    Args:
        threshold: Maximum memory usage ratio (0-1) to consider GPU available.
                  Defaults to PROCESSING_CONFIG.gpu_memory_threshold.
    
    Returns:
        Index of an available GPU, or 0 if none found or on error.
    """
    if threshold is None:
        threshold = PROCESSING_CONFIG.gpu_memory_threshold
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', 
             '--format=csv,nounits,noheader'],
            capture_output=True, 
            text=True, 
            check=True
        )
        
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) == 3:
                gpu_idx = int(parts[0])
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                mem_ratio = mem_used / mem_total
                
                if mem_ratio < threshold:
                    print(f"Found available GPU {gpu_idx} with {mem_ratio*100:.1f}% memory used")
                    return gpu_idx
        
        print("No low-usage GPU found, using GPU 0")
        return 0
        
    except FileNotFoundError:
        print("nvidia-smi not found. Using GPU 0")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi failed: {e}. Using GPU 0")
        return 0
    except Exception as e:
        print(f"Could not query GPU status: {e}. Using GPU 0")
        return 0

