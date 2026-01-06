"""
File utilities for historical preferences project.

Provides functions for saving files to NAS with local symlinks,
and checkpoint management for long-running preference generation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Set

import pandas as pd

from config import PROCESSING_CONFIG


def save_with_symlink(
    df: pd.DataFrame,
    nas_path: Path,
    local_path: Path,
    sep: str = '\t',
    save_first10: bool = False
) -> None:
    """
    Save DataFrame to NAS and create symlink in local directory.
    
    Args:
        df: DataFrame to save
        nas_path: Path on NAS storage for the actual file
        local_path: Path for the local symlink
        sep: CSV separator (default: tab)
        save_first10: If True, also save first 10 rows locally
    """
    # Ensure parent directories exist
    nas_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to NAS
    df.to_csv(nas_path, sep=sep, index=False)
    print(f"Saved to {nas_path}")
    
    # Create/update symlink
    if local_path.exists() or local_path.is_symlink():
        local_path.unlink()
    local_path.symlink_to(nas_path)
    print(f"Created symlink: {local_path} -> {nas_path}")
    
    # Optionally save first 10 rows locally
    if save_first10:
        first10_path = local_path.parent / local_path.name.replace('.csv', '_first10.csv')
        df.head(10).to_csv(first10_path, sep=sep, index=False)
        print(f"Saved first 10 rows to {first10_path}")


class CheckpointManager:
    """
    Manages checkpointing for long-running preference generation.
    
    Handles saving/loading checkpoints and tracking which comparisons
    have been fully processed to enable resumption. Supports multi-user
    generation by tracking (user_id, pair_index) combinations.
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        n_runs: int,
        checkpoint_interval: Optional[int] = None,
        user_ids: list[int] | None = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_path: Path for checkpoint file
            n_runs: Number of runs per comparison (for determining completion)
            checkpoint_interval: Save checkpoint every N comparisons
            user_ids: List of user IDs being processed (None for single-user mode)
        """
        self.checkpoint_path = checkpoint_path
        self.n_runs = n_runs
        self.checkpoint_interval = checkpoint_interval or PROCESSING_CONFIG.checkpoint_interval
        self.user_ids = user_ids or []
        
        self.preferences: list[dict[str, Any]] = []
        # Track by (user_id, pair_index) for multi-user, or just pair_index for single-user
        self.processed_keys: Set[tuple[int | None, int]] = set()
        self._comparisons_since_checkpoint = 0
    
    def load_checkpoint(self, sample_df: pd.DataFrame) -> bool:
        """
        Load existing checkpoint if available.
        
        Args:
            sample_df: DataFrame of comparisons being processed
        
        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if not self.checkpoint_path.exists():
            return False
        
        print(f"Found checkpoint file: {self.checkpoint_path}")
        print("Loading checkpoint...")
        
        checkpoint_df = pd.read_csv(self.checkpoint_path, sep='\t')
        print(f"Loaded {len(checkpoint_df)} preferences from checkpoint")
        
        self.preferences = checkpoint_df.to_dict('records')
        
        # Determine which (user_id, comparison) pairs are complete
        # Each comparison generates 2 * n_runs preferences (original + reversed)
        # Key: (user_id, question_id, sorted_response_ids)
        comparison_counts: dict[tuple, int] = defaultdict(int)
        
        for pref in self.preferences:
            question_id = pref['question_id']
            resp1 = pref['response_1_id']
            resp2 = pref['response_2_id']
            user_id = pref.get('user_id')  # None for single-user mode
            normalized_key = (user_id, question_id, tuple(sorted([str(resp1), str(resp2)])))
            comparison_counts[normalized_key] += 1
        
        # Find completed comparisons
        for key, count in comparison_counts.items():
            if count >= 2 * self.n_runs:
                user_id, question_id, (resp1, resp2) = key
                for df_idx, df_row in sample_df.iterrows():
                    row_resp1 = str(df_row['response_1_id'])
                    row_resp2 = str(df_row['response_2_id'])
                    if (df_row['question_id'] == question_id and 
                        set([row_resp1, row_resp2]) == set([resp1, resp2])):
                        self.processed_keys.add((user_id, df_idx))
                        break
        
        print(f"Resuming: {len(self.preferences)} preferences, "
              f"{len(self.processed_keys)} (user, comparison) pairs already processed")
        return True
    
    def is_processed(self, idx: int, user_id: int | None = None) -> bool:
        """
        Check if a comparison index has been fully processed.
        
        Args:
            idx: DataFrame index of the comparison
            user_id: User ID (None for single-user mode)
        """
        return (user_id, idx) in self.processed_keys
    
    def add_preference(self, preference: dict[str, Any]) -> None:
        """Add a preference result."""
        self.preferences.append(preference)
    
    def maybe_save_checkpoint(self, force: bool = False) -> None:
        """
        Save checkpoint if interval reached or forced.
        
        Args:
            force: If True, save regardless of interval
        """
        # Calculate entries per comparison based on number of users
        n_users = max(1, len(self.user_ids))
        entries_per_comparison = 2 * self.n_runs * n_users
        total_comparisons = len(self.preferences) // entries_per_comparison if entries_per_comparison > 0 else 0
        
        should_save = force or (
            total_comparisons > 0 and 
            total_comparisons % self.checkpoint_interval == 0 and
            self._comparisons_since_checkpoint >= self.checkpoint_interval
        )
        
        if should_save:
            self._save_checkpoint()
            self._comparisons_since_checkpoint = 0
        else:
            self._comparisons_since_checkpoint += 1
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint to disk."""
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_df = pd.DataFrame(self.preferences)
            checkpoint_df.to_csv(self.checkpoint_path, sep='\t', index=False)
            
            n_users = max(1, len(self.user_ids))
            entries_per_comparison = 2 * self.n_runs * n_users
            comparisons = len(self.preferences) // entries_per_comparison if entries_per_comparison > 0 else 0
            print(f"\n[Checkpoint] Saved {len(self.preferences)} preferences "
                  f"({comparisons} comparisons) to {self.checkpoint_path}")
        except Exception as e:
            print(f"\n[Warning] Failed to save checkpoint: {e}")
    
    def get_preferences_df(self) -> pd.DataFrame:
        """Get preferences as DataFrame."""
        return pd.DataFrame(self.preferences)
    
    def cleanup(self) -> None:
        """Remove checkpoint file after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"Removed checkpoint file: {self.checkpoint_path}")

