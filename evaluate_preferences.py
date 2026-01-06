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


def calculate_user_agreement(preferences_df: pd.DataFrame) -> Optional[dict[str, Any]]:
    """
    Calculate within-user and between-user pairwise agreement.
    
    This metric compares how consistently users agree with themselves
    versus how much they agree with other users. If user profiles are
    creating meaningful differentiation:
    - within_user_agreement should be HIGH (users are internally consistent)
    - between_user_agreement should be LOWER (users disagree with each other)
    
    Agreement is calculated by comparing pairs of individual runs:
    - Within-user: pairs of runs from the SAME user
    - Between-user: pairs of runs from DIFFERENT users
    
    Args:
        preferences_df: DataFrame with preference data including 'user_id' column
        
    Returns:
        Dictionary with within_user_agreement, between_user_agreement, 
        and the difference (within - between), or None if not applicable
    """
    if 'user_id' not in preferences_df.columns:
        return None
    
    # Filter to valid responses only
    valid_mask = preferences_df['model_choice'].isin([1, 2, '1', '2'])
    valid_df = preferences_df[valid_mask].copy()
    
    if len(valid_df) == 0:
        return None
    
    # Normalize model_choice to string for consistent comparison
    valid_df['choice_normalized'] = valid_df['model_choice'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) else None
    )
    
    # Get unique user IDs
    user_ids = valid_df['user_id'].dropna().unique()
    if len(user_ids) < 2:
        return {
            'within_user_agreement': None,
            'between_user_agreement': None,
            'difference': None,
            'error': 'Need at least 2 users for comparison'
        }
    
    # Get unique question pairs (identified by question_id and response pair)
    valid_df['pair_key'] = valid_df.apply(
        lambda r: (r['question_id'], tuple(sorted([str(r['response_1_id']), str(r['response_2_id'])]))),
        axis=1
    )
    unique_pairs = valid_df['pair_key'].unique()
    
    within_agreements = 0
    within_total = 0
    between_agreements = 0
    between_total = 0
    
    for pair_key in tqdm(unique_pairs, desc="Calculating user agreement", unit="pair"):
        pair_df = valid_df[valid_df['pair_key'] == pair_key]
        
        # Group runs by user
        user_runs: dict[int, list[str]] = {}
        for _, row in pair_df.iterrows():
            uid = row['user_id']
            choice = row['choice_normalized']
            if uid is not None and choice is not None:
                if uid not in user_runs:
                    user_runs[uid] = []
                user_runs[uid].append(choice)
        
        # Calculate within-user agreement (compare runs from same user)
        for uid, runs in user_runs.items():
            if len(runs) < 2:
                continue
            # Compare all pairs of runs from this user
            for i in range(len(runs)):
                for j in range(i + 1, len(runs)):
                    within_total += 1
                    if runs[i] == runs[j]:
                        within_agreements += 1
        
        # Calculate between-user agreement (compare runs from different users)
        user_ids_in_pair = list(user_runs.keys())
        for i, uid1 in enumerate(user_ids_in_pair):
            for uid2 in user_ids_in_pair[i + 1:]:
                # Compare all runs from user1 with all runs from user2
                for run1 in user_runs[uid1]:
                    for run2 in user_runs[uid2]:
                        between_total += 1
                        if run1 == run2:
                            between_agreements += 1
    
    within_agreement_rate = (within_agreements / within_total * 100) if within_total > 0 else None
    between_agreement_rate = (between_agreements / between_total * 100) if between_total > 0 else None
    
    difference = None
    if within_agreement_rate is not None and between_agreement_rate is not None:
        difference = within_agreement_rate - between_agreement_rate
    
    return {
        'within_user_agreement': within_agreement_rate,
        'between_user_agreement': between_agreement_rate,
        'difference': difference,
        'within_comparisons': within_total,
        'between_comparisons': between_total
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
    parser.add_argument(
        "--user_agreement",
        action="store_true",
        help="Calculate within-user vs between-user agreement (requires user_id column)"
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
    
    # User agreement analysis (if requested and data has user_id)
    if args.user_agreement:
        print("\n" + "-" * 60)
        print("User Profile Agreement Analysis")
        print("-" * 60)
        
        user_agreement = calculate_user_agreement(preferences_df)
        if user_agreement is None:
            print("  Not available (no user_id column in data)")
        elif 'error' in user_agreement:
            print(f"  {user_agreement['error']}")
        else:
            within = user_agreement['within_user_agreement']
            between = user_agreement['between_user_agreement']
            diff = user_agreement['difference']
            
            if within is not None:
                print(f"\n  Within-user agreement:  {within:.2f}%")
                print(f"    ({user_agreement['within_comparisons']:,} pairwise comparisons)")
            if between is not None:
                print(f"\n  Between-user agreement: {between:.2f}%")
                print(f"    ({user_agreement['between_comparisons']:,} pairwise comparisons)")
            if diff is not None:
                print(f"\n  Difference (within - between): {diff:+.2f}%")
                if diff > 10:
                    print("  --> User profiles appear to be creating distinct preferences")
                elif diff > 0:
                    print("  --> Some differentiation, but profiles may need strengthening")
                else:
                    print("  --> User profiles are NOT creating meaningful differentiation")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

