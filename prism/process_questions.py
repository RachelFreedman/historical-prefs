# %%
#!/usr/bin/env python3
"""
Process the PRISM dataset from HuggingFace to create pairwise comparison questions.

The PRISM dataset contains human-LLM conversations with multiple model responses per prompt.
Each response has a score and if_chosen indicator from human annotators.

This script:
1. Loads the utterances config from HannahRoseKirk/prism-alignment
2. Groups responses by interaction (same prompt, multiple model responses)
3. Creates pairwise comparisons between responses
4. Outputs questions.csv and questions_pairwise.csv
"""
from datasets import load_dataset
import pandas as pd
from itertools import combinations
from collections import defaultdict
from pathlib import Path
import os

# Store large files in scratch directory (more space available)
scratch_datapath = Path('/scratch/rachel/historical-prefs/data/prism/')
scratch_datapath.mkdir(parents=True, exist_ok=True)

# Local path for symlinks
local_datapath = Path('data/prism/')
local_datapath.mkdir(parents=True, exist_ok=True)


def save_with_symlink(df, filename, save_first10=True):
    """Save DataFrame to scratch and create symlink in local directory."""
    scratch_path = scratch_datapath / filename
    local_path = local_datapath / filename
    
    # Save to scratch
    df.to_csv(scratch_path, sep='\t', index=False)
    print(f"Saved to {scratch_path}")
    
    # Create symlink in local directory
    if local_path.exists() or local_path.is_symlink():
        local_path.unlink()
    local_path.symlink_to(scratch_path)
    print(f"Created symlink: {local_path} -> {scratch_path}")
    
    # Save first 10 rows locally (small file, no symlink needed)
    if save_first10:
        first10_filename = filename.replace('.csv', '_first10.csv')
        first10_path = local_datapath / first10_filename
        df[:10].to_csv(first10_path, sep='\t', index=False)
        print(f"Saved first 10 rows to {first10_path}")

# %%
# Load the dataset
print("Loading PRISM utterances dataset...")
dataset = load_dataset('HannahRoseKirk/prism-alignment', 'utterances', trust_remote_code=True)
utterances = dataset['train']
print(f"Loaded {len(utterances)} utterances")

# %%
# Group utterances by interaction_id (same prompt, different responses)
print("\nGrouping utterances by interaction...")
interactions = defaultdict(list)

for row in utterances:
    interactions[row['interaction_id']].append({
        'utterance_id': row['utterance_id'],
        'conversation_id': row['conversation_id'],
        'user_id': row['user_id'],
        'turn': row['turn'],
        'within_turn_id': row['within_turn_id'],
        'conversation_type': row['conversation_type'],
        'user_prompt': row['user_prompt'],
        'model_response': row['model_response'],
        'model_name': row['model_name'],
        'model_provider': row['model_provider'],
        'score': row['score'],
        'if_chosen': row['if_chosen'],
    })

print(f"Found {len(interactions)} unique interactions")

# Filter to first-turn interactions only (turn == 0)
# This ensures we only include the opening question of each conversation
interactions_first_turn = {k: v for k, v in interactions.items() if v[0]['turn'] == 0}
print(f"First-turn interactions (turn=0): {len(interactions_first_turn)}")

# Filter to interactions with 2+ responses (needed for pairwise comparisons)
interactions_with_pairs = {k: v for k, v in interactions_first_turn.items() if len(v) >= 2}
print(f"First-turn interactions with 2+ responses: {len(interactions_with_pairs)}")

# %%
# Create questions DataFrame
# Each question is a unique interaction with all its responses
print("\nCreating questions DataFrame...")

questions_data = []
for question_id, (interaction_id, responses) in enumerate(interactions_with_pairs.items()):
    # Sort responses by within_turn_id for consistent ordering
    responses = sorted(responses, key=lambda x: x['within_turn_id'])
    
    # Create response columns dynamically based on number of responses
    question_row = {
        'question_id': question_id,
        'interaction_id': interaction_id,
        'conversation_id': responses[0]['conversation_id'],
        'user_id': responses[0]['user_id'],
        'turn': responses[0]['turn'],
        'conversation_type': responses[0]['conversation_type'],
        'prompt': responses[0]['user_prompt'],
        'num_responses': len(responses),
    }
    
    # Add each response with its metadata
    for i, resp in enumerate(responses):
        letter = chr(ord('A') + i)  # A, B, C, D, ...
        question_row[f'response_{letter}'] = resp['model_response']
        question_row[f'response_{letter}_model'] = resp['model_name']
        question_row[f'response_{letter}_score'] = resp['score']
        question_row[f'response_{letter}_chosen'] = resp['if_chosen']
        question_row[f'response_{letter}_utterance_id'] = resp['utterance_id']
    
    questions_data.append(question_row)

questions_df = pd.DataFrame(questions_data)
print(f"Created questions DataFrame with {len(questions_df)} rows")
print(f"Columns: {list(questions_df.columns)}")

# Save to CSV (in scratch with symlink)
save_with_symlink(questions_df, 'questions.csv')

# %%
# Create pairwise comparisons
print("\nCreating pairwise comparisons...")

# Reload questions to ensure consistency
questions_df = pd.read_csv(scratch_datapath / 'questions.csv', sep='\t')
print(f"Loaded {len(questions_df)} questions")

comparisons_data = []

for _, row in questions_df.iterrows():
    question_id = row['question_id']
    prompt = row['prompt']
    num_responses = row['num_responses']
    
    # Get all available responses for this question
    responses = {}
    chosen_letter = None
    for i in range(num_responses):
        letter = chr(ord('A') + i)
        response_col = f'response_{letter}'
        if response_col in row and pd.notna(row[response_col]):
            is_chosen = row.get(f'response_{letter}_chosen', False)
            # Handle string 'True'/'False' from CSV
            if isinstance(is_chosen, str):
                is_chosen = is_chosen.lower() == 'true'
            responses[letter] = {
                'text': row[response_col],
                'model': row.get(f'response_{letter}_model', ''),
                'score': row.get(f'response_{letter}_score', ''),
                'chosen': is_chosen,
                'utterance_id': row.get(f'response_{letter}_utterance_id', ''),
            }
            if is_chosen:
                chosen_letter = letter
    
    # Only create pairs that include the chosen response
    # This allows direct comparison with human preference ground truth
    if chosen_letter is None:
        # Skip questions where no response was chosen
        continue
    
    chosen_resp = responses[chosen_letter]
    for letter, resp in responses.items():
        if letter == chosen_letter:
            continue  # Skip pairing chosen with itself
        
        # Always put chosen response as response_2 (the "correct" answer)
        # This makes it clear that human_preferred = response_2
        comparisons_data.append({
            'question_id': question_id,
            'response_1_id': f"{question_id}{letter}",
            'response_2_id': f"{question_id}{chosen_letter}",
            'prompt': prompt,
            'response_1': resp['text'],
            'response_2': chosen_resp['text'],
            'response_1_model': resp['model'],
            'response_2_model': chosen_resp['model'],
            'response_1_score': resp['score'],
            'response_2_score': chosen_resp['score'],
            'response_1_chosen': False,
            'response_2_chosen': True,
            'human_preferred': '2',  # Human always prefers response_2 (the chosen one)
            'interaction_id': row['interaction_id'],
            'conversation_id': row['conversation_id'],
        })

comparisons_df = pd.DataFrame(comparisons_data)
print(f"Created {len(comparisons_df)} pairwise comparisons")
print(f"Columns: {list(comparisons_df.columns)}")

# With chosen-only pairs, each question with n responses generates (n-1) pairs
# (pairing the chosen response with each non-chosen response)
questions_with_comparisons = comparisons_df['question_id'].nunique()
print(f"Questions with valid comparisons: {questions_with_comparisons}")
print(f"Questions skipped (no chosen response): {len(questions_df) - questions_with_comparisons}")

# Save to CSV (in scratch with symlink)
save_with_symlink(comparisons_df, 'questions_pairwise.csv')

# %%
# Summary statistics
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)
print(f"Total questions: {len(questions_df)}")
print(f"Total pairwise comparisons: {len(comparisons_df)}")
print(f"\nResponses per question distribution:")
print(questions_df['num_responses'].value_counts().sort_index())
print(f"\nConversation types:")
print(questions_df['conversation_type'].value_counts())

