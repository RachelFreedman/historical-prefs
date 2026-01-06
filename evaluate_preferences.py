#!/usr/bin/env python3
"""
Evaluate preference consistency for historical preference datasets.

This unified script handles evaluation for both CA and PRISM datasets.
It calculates preference consistency metrics and agreement with human preferences.

Usage:
    python evaluate_preferences.py --dataset prism --input preferences_model_8B_C013_pairs0-22038_3runs.csv
    python evaluate_preferences.py --dataset ca --input preferences_model_8B_C013_pairs0-52122_3runs.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_dataset_config


def calculate_preference_consistency(preferences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate preference consistency for each question.
    
    Preference consistency for a pair of alternatives (A, B) is the percentage
    of times that the more commonly preferred alternative is chosen.
    
    Preference consistency for a question is the average over all pairs.
    
    Args:
        preferences_df: DataFrame with preference data
        
    Returns:
        DataFrame with question_id, num_pairs, and preference_consistency
    """
    consistency_results: list[dict[str, Any]] = []
    unique_question_ids = preferences_df['question_id'].unique()
    
    for question_id in tqdm(unique_question_ids, desc="Calculating consistency", unit="question"):
        question_df = preferences_df[preferences_df['question_id'] == question_id]
        
        # Get unique pairs
        pairs: list[tuple[str, str]] = []
        for _, row in question_df.iterrows():
            pair = tuple(sorted([str(row['response_1_id']), str(row['response_2_id'])]))
            if pair not in pairs:
                pairs.append(pair)
        
        pair_consistencies: list[float] = []
        
        for pair in pairs:
            pair_df = question_df[
                (question_df['response_1_id'].astype(str).isin(pair)) & 
                (question_df['response_2_id'].astype(str).isin(pair))
            ]
            
            # Count preferences, excluding invalid responses
            pref_counts: dict[str, int] = {}
            for _, row in pair_df.iterrows():
                choice = row['model_choice']
                
                # Skip invalid entries
                if pd.isna(choice) or choice in [-1, '-1']:
                    continue
                
                if choice in [1, 2, '1', '2']:
                    if choice in [1, '1']:
                        preferred = str(row['response_1_id'])
                    else:
                        preferred = str(row['response_2_id'])
                    pref_counts[preferred] = pref_counts.get(preferred, 0) + 1
            
            if pref_counts:
                total = sum(pref_counts.values())
                max_pref = max(pref_counts.values())
                consistency = (max_pref / total) * 100
                pair_consistencies.append(consistency)
        
        avg_consistency = sum(pair_consistencies) / len(pair_consistencies) if pair_consistencies else 0.0
        
        consistency_results.append({
            'question_id': question_id,
            'num_pairs': len(pairs),
            'preference_consistency': avg_consistency
        })
    
    return pd.DataFrame(consistency_results)


def calculate_valid_response_frequency(preferences_df: pd.DataFrame) -> float:
    """
    Calculate the percentage of entries with valid responses.
    
    Valid responses are where model_choice is 1, 2, '1', or '2'.
    
    Returns:
        Percentage (0-100) of valid responses
    """
    valid_mask = preferences_df['model_choice'].isin([1, 2, '1', '2'])
    valid_count = valid_mask.sum()
    total_count = len(preferences_df)
    
    return (valid_count / total_count) * 100.0 if total_count > 0 else 0.0


def calculate_agreement_with_human(preferences_df: pd.DataFrame) -> Optional[dict[str, Any]]:
    """
    Calculate agreement between model and human preferences.
    
    Only applicable for datasets with human_preferred column (e.g., PRISM).
    
    Returns:
        Dictionary with agreement metrics, or None if not applicable
    """
    if 'human_preferred' not in preferences_df.columns:
        return None
    
    # Filter to valid model responses
    valid_mask = preferences_df['model_choice'].isin([1, 2, '1', '2'])
    valid_df = preferences_df[valid_mask].copy()
    
    if len(valid_df) == 0:
        return {'agreement_rate': 0.0, 'agreement_count': 0, 'comparison_count': 0}
    
    # Filter to rows with valid human preference (handle float values like 2.0)
    valid_df = valid_df[valid_df['human_preferred'].notna()]
    valid_df = valid_df[valid_df['human_preferred'].apply(
        lambda x: int(float(x)) in [1, 2] if pd.notna(x) else False
    )]
    
    if len(valid_df) == 0:
        return {'agreement_rate': 0.0, 'agreement_count': 0, 'comparison_count': 0}
    
    # Count agreements - normalize both to int for comparison
    def normalize_choice(x):
        """Normalize choice value to int."""
        if pd.isna(x):
            return None
        return int(float(x))
    
    agreement_count = sum(
        normalize_choice(row['human_preferred']) == normalize_choice(row['model_choice'])
        for _, row in valid_df.iterrows()
    )
    
    comparison_count = len(valid_df)
    agreement_rate = (agreement_count / comparison_count) * 100.0 if comparison_count > 0 else 0.0
    
    return {
        'agreement_rate': agreement_rate,
        'agreement_count': agreement_count,
        'comparison_count': comparison_count
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate preference consistency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ca", "prism"],
        help="Dataset being evaluated"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with preferences (filename or full path)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Get dataset configuration
    dataset_config = get_dataset_config(args.dataset)
    
    # Resolve input path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        # Try NAS path first, then local
        nas_path = dataset_config.nas_prefs_dir / args.input
        local_path = dataset_config.local_prefs_dir / args.input
        
        if nas_path.exists():
            input_path = nas_path
        elif local_path.exists():
            input_path = local_path
        else:
            print(f"Error: Could not find {args.input}")
            print(f"Searched: {nas_path}, {local_path}")
            sys.exit(1)
    
    # Load preferences
    start_time = time.time()
    print(f"Loading preferences from {input_path}")
    preferences_df = pd.read_csv(input_path, sep='\t')
    print(f"Loaded {len(preferences_df):,} preferences in {time.time() - start_time:.2f}s")
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print(f"Evaluation Results for {args.dataset.upper()}")
    print("=" * 60)
    
    # Preference consistency
    consistency_df = calculate_preference_consistency(preferences_df)
    avg_consistency = consistency_df['preference_consistency'].mean()
    print(f"\nAverage preference consistency: {avg_consistency:.2f}%")
    
    # Consistency thresholds
    pct_gt_70 = (consistency_df['preference_consistency'] > 70).mean() * 100
    pct_gt_80 = (consistency_df['preference_consistency'] > 80).mean() * 100
    print(f"Questions with consistency > 70%: {pct_gt_70:.2f}%")
    print(f"Questions with consistency > 80%: {pct_gt_80:.2f}%")
    
    # Valid response frequency
    valid_freq = calculate_valid_response_frequency(preferences_df)
    print(f"\nValid response frequency: {valid_freq:.2f}%")
    
    # Agreement with human (if available)
    agreement = calculate_agreement_with_human(preferences_df)
    if agreement:
        print(f"\nAgreement with human preferences: {agreement['agreement_rate']:.2f}%")
        print(f"  ({agreement['agreement_count']} / {agreement['comparison_count']} comparisons)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

