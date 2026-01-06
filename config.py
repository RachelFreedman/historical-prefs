"""
Centralized configuration for historical preferences project.

This module provides path configuration and model parameters used across
all scripts in the project. Paths are configured to use NAS storage for
large files (models, datasets, outputs) with symlinks in the local workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple
import os

# =============================================================================
# Base Paths
# =============================================================================

NAS_BASE = Path("/nas/ucb/rachel/historical-prefs")
LOCAL_BASE = Path(__file__).parent / "data"
PROFILES_PATH = Path(__file__).parent / "data" / "profiles.json"

# =============================================================================
# Environment Configuration
# =============================================================================

def configure_environment() -> None:
    """Configure environment variables for HuggingFace and temp directories."""
    os.environ['HF_HOME'] = str(NAS_BASE / "hf_cache")
    os.environ['TRANSFORMERS_CACHE'] = str(NAS_BASE / "hf_cache")
    os.environ['TMPDIR'] = str(NAS_BASE / "tmp")
    
    # Ensure directories exist
    (NAS_BASE / "hf_cache").mkdir(parents=True, exist_ok=True)
    (NAS_BASE / "tmp").mkdir(parents=True, exist_ok=True)

# =============================================================================
# Dataset Configuration
# =============================================================================

DatasetName = Literal["ca", "prism"]

@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    name: DatasetName
    nas_base: Path = field(default_factory=lambda: NAS_BASE)
    local_base: Path = field(default_factory=lambda: LOCAL_BASE)
    
    @property
    def nas_data_dir(self) -> Path:
        """NAS directory for dataset files."""
        return self.nas_base / "data" / self.name
    
    @property
    def nas_prefs_dir(self) -> Path:
        """NAS directory for preference output files."""
        return self.nas_data_dir / "prefs"
    
    @property
    def local_data_dir(self) -> Path:
        """Local directory for dataset files (symlinks)."""
        return self.local_base / self.name
    
    @property
    def local_prefs_dir(self) -> Path:
        """Local directory for preference output files (symlinks)."""
        return self.local_data_dir / "prefs"
    
    @property
    def questions_path(self) -> Path:
        """Path to questions CSV (on NAS)."""
        return self.nas_data_dir / "questions.csv"
    
    @property
    def questions_pairwise_path(self) -> Path:
        """Path to pairwise questions CSV (on NAS)."""
        return self.nas_data_dir / "questions_pairwise.csv"
    
    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        self.nas_data_dir.mkdir(parents=True, exist_ok=True)
        self.nas_prefs_dir.mkdir(parents=True, exist_ok=True)
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        self.local_prefs_dir.mkdir(parents=True, exist_ok=True)
    
    def prefs_output_path(self, model_size: str, century: str, 
                          start_idx: int, end_idx: int, n_runs: int) -> Path:
        """Generate path for preference output file."""
        filename = f"preferences_model_{model_size}_{century}_pairs{start_idx}-{end_idx}_{n_runs}runs.csv"
        return self.nas_prefs_dir / filename
    
    def checkpoint_path(self, model_size: str, century: str,
                        start_idx: int, end_idx: int, n_runs: int) -> Path:
        """Generate path for checkpoint file."""
        filename = f"preferences_model_{model_size}_{century}_pairs{start_idx}-{end_idx}_{n_runs}runs.csv.checkpoint"
        return self.nas_prefs_dir / filename


def get_dataset_config(dataset: DatasetName) -> DatasetConfig:
    """Get configuration for a specific dataset."""
    return DatasetConfig(name=dataset)

# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for ProgressGym historical models."""
    
    # Model identifiers
    size: str = "8B"  # "8B" or "70B"
    century: str = "C013"  # C013-C021
    
    # Available options
    VALID_SIZES: tuple[str, ...] = ("8B", "70B")
    VALID_CENTURIES: tuple[str, ...] = (
        "C013", "C014", "C015", "C016", "C017", 
        "C018", "C019", "C020", "C021"
    )
    
    # Generation parameters
    max_new_tokens: int = 20
    temperature: float = 0.9
    do_sample: bool = True
    
    # HuggingFace model naming
    hf_org: str = "PKU-Alignment"
    default_version: str = "v0.2"
    
    @property
    def base_model_name(self) -> str:
        """Base model name without version."""
        return f"ProgressGym-HistLlama3-{self.size}-{self.century}-instruct"
    
    def full_model_name(self, version: str | None = None) -> str:
        """Full HuggingFace model path."""
        v = version or self.default_version
        return f"{self.hf_org}/{self.base_model_name}-{v}"

# =============================================================================
# Processing Configuration  
# =============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for preference generation processing."""
    
    # GPU settings
    gpu_memory_threshold: float = 0.10  # Use GPU if < 10% memory used
    
    # Checkpointing
    checkpoint_interval: int = 100  # Save checkpoint every N comparisons
    
    # Default processing
    default_n_samples: int = 10
    default_n_runs: int = 1
    default_start_idx: int = 0

# =============================================================================
# Global instances
# =============================================================================

PROCESSING_CONFIG = ProcessingConfig()

