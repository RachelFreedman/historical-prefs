import pandas as pd
import argparse
from tqdm import tqdm
import time

datapath = 'data/prefs/'

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
            pref_counts = {}
            for _, row in pair_df.iterrows():
                # Handle both string and integer model_choice values
                choice = row['model_choice']
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate preference consistency")
    parser.add_argument("--input", type=str, default="preferences_model_8B_C013_pairs0-52122_3runs.csv",
                       help="Input CSV file with preferences")
    args = parser.parse_args()
    
    # Load preferences with progress indicator
    input_path = datapath + args.input
    start_time = time.time()
    preferences_df = pd.read_csv(input_path, sep='\t')
    print(f"Loaded {len(preferences_df):,} preferences in {(time.time() - start_time):.2f} seconds")

    # analyze preferences
    consistency_df = calculate_preference_consistency(preferences_df)

    # average consistency across all questions
    avg_consistency = consistency_df['preference_consistency'].mean()
    print(f"Average preference consistency: {avg_consistency:.2f}%")

    
if __name__ == "__main__":
    main()