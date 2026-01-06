#!/usr/bin/env python3
"""
Evaluate preference consistency for PRISM dataset.

This module provides functions to calculate preference consistency metrics,
measuring how consistently the model chooses the same response when presented
with the same comparison in different orderings.
"""

import pandas as pd
import argparse
from tqdm import tqdm
import time
from pathlib import Path

# Paths configuration
scratch_datapath = Path('/scratch/rachel/historical-prefs/data/prism/prefs/')
local_datapath = Path('data/prism/prefs/')

def calculate_preference_consistency(preferences_df):
    """
    Calculate preference consistency for each question.
    
    Preference consistency for a pair of alternatives (A, B) is the percentage
    of times that the more commonly preferred alternative is chosen when comparing A to B.
    
    Preference consistency for a question is the average over all pairs for that question.
    
    Args:
        preferences_df: DataFrame with preference data
        
    Returns:
        DataFrame with question_id, num_pairs, and preference_consistency
    """
    consistency_results = []
    
    # Get unique question IDs for progress tracking
    unique_question_ids = preferences_df['question_id'].unique()
    
    # Group by question_id with progress bar
    for question_id in tqdm(unique_question_ids, desc="Calculating consistency", unit="question"):
        question_df = preferences_df[preferences_df['question_id'] == question_id]
        
        # Get all unique pairs of response IDs for this question
        pairs = []
        for _, row in question_df.iterrows():
            pair = tuple(sorted([row['response_1_id'], row['response_2_id']]))
            if pair not in pairs:
                pairs.append(pair)
        
        pair_consistencies = []
        
        # Calculate consistency for each pair
        for pair in pairs:
            pair_df = question_df[
                (question_df['response_1_id'].isin(pair)) & 
                (question_df['response_2_id'].isin(pair))
            ]
            
            # Count preferences for each alternative
            # Exclude entries where model_choice is -1 (invalid/unparseable responses)
            pref_counts = {}
            for _, row in pair_df.iterrows():
                # Handle both string and integer model_choice values
                choice = row['model_choice']
                # Explicitly exclude -1 values (invalid responses that couldn't be parsed)
                # These entries should not count as consistent or inconsistent
                if pd.isna(choice) or choice == -1 or choice == '-1':
                    continue  # Skip invalid entries
                # Only process valid choices (1 or 2)
                if choice in [1, 2, '1', '2']:
                    # Determine which response was preferred
                    if choice == 1 or choice == '1':
                        preferred = row['response_1_id']
                    else:  # choice == 2 or '2'
                        preferred = row['response_2_id']
                    
                    pref_counts[preferred] = pref_counts.get(preferred, 0) + 1
            
            # Calculate consistency: percentage of times the more popular choice was selected
            if len(pref_counts) > 0:
                total_comparisons = sum(pref_counts.values())
                max_preferences = max(pref_counts.values())
                consistency = (max_preferences / total_comparisons) * 100
                pair_consistencies.append(consistency)
        
        # Average consistency across all pairs
        if len(pair_consistencies) > 0:
            avg_consistency = sum(pair_consistencies) / len(pair_consistencies)
        else:
            avg_consistency = 0.0
        
        consistency_results.append({
            'question_id': question_id,
            'num_pairs': len(pairs),
            'preference_consistency': avg_consistency
        })
    
    return pd.DataFrame(consistency_results)

def calculate_valid_response_frequency(preferences_df):
    """
    Calculate the percentage of entries that contain valid responses.
    A valid response is one where model_choice is 1, 2, '1', or '2'.
    Invalid responses (where model_choice is -1, '-1', or NaN) are excluded.
    
    Returns:
        float: Percentage (0-100) of valid responses
    """
    # Count valid responses: must be 1, 2, '1', or '2'
    # This matches the logic in calculate_preference_consistency
    valid_mask = preferences_df['model_choice'].isin([1, 2, '1', '2'])
    valid_count = valid_mask.sum()
    total_count = len(preferences_df)
    if total_count > 0:
        return (valid_count / total_count) * 100.0
    else:
        return 0.0

def calculate_agreement_with_human(preferences_df):
    """
    Calculate agreement between model preferences and human preferences.
    
    Uses the human_preferred column to determine human preferences.
    
    Returns:
        dict with agreement metrics
    """
    # Check if human_preferred column exists
    if 'human_preferred' not in preferences_df.columns:
        return None
    
    # Filter to valid model responses
    valid_mask = preferences_df['model_choice'].isin([1, 2, '1', '2'])
    valid_df = preferences_df[valid_mask].copy()
    
    if len(valid_df) == 0:
        return {'agreement_rate': 0.0, 'total_valid': 0}
    
    # Filter to rows with valid human preference
    valid_df = valid_df[valid_df['human_preferred'].isin(['1', '2', 1, 2])]
    
    if len(valid_df) == 0:
        return {'agreement_rate': 0.0, 'total_valid': 0}
    
    # Compare model choice to human preference
    agreement_count = 0
    comparison_count = len(valid_df)
    
    for _, row in valid_df.iterrows():
        human_choice = str(row['human_preferred'])
        model_choice = str(row['model_choice'])
        
        if human_choice == model_choice:
            agreement_count += 1
    
    if comparison_count > 0:
        agreement_rate = (agreement_count / comparison_count) * 100.0
    else:
        agreement_rate = 0.0
    
    return {
        'agreement_rate': agreement_rate,
        'agreement_count': agreement_count,
        'comparison_count': comparison_count
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate preference consistency for PRISM dataset")
    parser.add_argument("--input", type=str, default="preferences_model_8B_C013_pairs0-61467_3runs.csv",
                       help="Input CSV file with preferences")
    args = parser.parse_args()
    
    # Try scratch path first, then local
    input_path = scratch_datapath / args.input
    if not input_path.exists():
        input_path = local_datapath / args.input
    
    # Load preferences with progress indicator
    start_time = time.time()
    print(f"Loading preferences from {input_path}")
    preferences_df = pd.read_csv(input_path, sep='\t')
    print(f"Loaded {len(preferences_df):,} preferences in {(time.time() - start_time):.2f} seconds")

    # Analyze preferences
    consistency_df = calculate_preference_consistency(preferences_df)

    # Average consistency across all questions
    avg_consistency = consistency_df['preference_consistency'].mean()
    print(f"\nAverage preference consistency: {avg_consistency:.2f}%")

    # Percent of questions with consistency > 70%
    percent_questions_with_consistency_gt_70 = len(consistency_df[consistency_df['preference_consistency'] > 70]) / len(consistency_df) * 100.0
    print(f"Percent of questions with consistency > 70%: {percent_questions_with_consistency_gt_70:.2f}%")

    # Percent of questions with consistency > 80%
    percent_questions_with_consistency_gt_80 = len(consistency_df[consistency_df['preference_consistency'] > 80]) / len(consistency_df) * 100.0
    print(f"Percent of questions with consistency > 80%: {percent_questions_with_consistency_gt_80:.2f}%")

    # Calculate valid response frequency
    valid_response_frequency = calculate_valid_response_frequency(preferences_df)
    print(f"Valid response frequency: {valid_response_frequency:.2f}%")
    
    # Calculate agreement with human preferences (if available)
    agreement = calculate_agreement_with_human(preferences_df)
    if agreement:
        print(f"\nAgreement with human preferences: {agreement['agreement_rate']:.2f}%")
        print(f"  ({agreement['agreement_count']} / {agreement['comparison_count']} comparisons)")

if __name__ == "__main__":
    main()

