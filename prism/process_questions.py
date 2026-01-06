#!/usr/bin/env python3
"""
Process the PRISM dataset to create pairwise comparison questions.

The PRISM dataset contains human-LLM conversations with multiple model responses.
Each response has a score and if_chosen indicator from human annotators.

This script:
1. Loads the utterances config from HannahRoseKirk/prism-alignment
2. Filters to first-turn interactions only
3. Groups responses by interaction (same prompt, multiple model responses)
4. Creates pairwise comparisons only between chosen and non-chosen responses
5. Outputs questions.csv and questions_pairwise.csv

Usage:
    python prism/process_questions.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_dataset_config
from utils.file_utils import save_with_symlink


def main() -> None:
    """Main entry point."""
    # Get dataset configuration
    config = get_dataset_config("prism")
    config.ensure_dirs()
    
    # Load dataset
    print("Loading PRISM utterances dataset...")
    dataset = load_dataset('HannahRoseKirk/prism-alignment', 'utterances', trust_remote_code=True)
    utterances = dataset['train']
    print(f"Loaded {len(utterances)} utterances")
    
    # Group utterances by interaction_id
    print("\nGrouping utterances by interaction...")
    interactions: dict[str, list[dict]] = defaultdict(list)
    
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
    interactions_first = {
        k: v for k, v in interactions.items() 
        if v[0]['turn'] == 0
    }
    print(f"First-turn interactions (turn=0): {len(interactions_first)}")
    
    # Filter to interactions with 2+ responses
    interactions_valid = {
        k: v for k, v in interactions_first.items() 
        if len(v) >= 2
    }
    print(f"First-turn interactions with 2+ responses: {len(interactions_valid)}")
    
    # Create questions DataFrame
    print("\nCreating questions DataFrame...")
    questions_data: list[dict] = []
    
    for question_id, (interaction_id, responses) in enumerate(interactions_valid.items()):
        # Sort by within_turn_id for consistent ordering
        responses = sorted(responses, key=lambda x: x['within_turn_id'])
        
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
        
        # Add each response with metadata
        for i, resp in enumerate(responses):
            letter = chr(ord('A') + i)
            question_row[f'response_{letter}'] = resp['model_response']
            question_row[f'response_{letter}_model'] = resp['model_name']
            question_row[f'response_{letter}_score'] = resp['score']
            question_row[f'response_{letter}_chosen'] = resp['if_chosen']
            question_row[f'response_{letter}_utterance_id'] = resp['utterance_id']
        
        questions_data.append(question_row)
    
    questions_df = pd.DataFrame(questions_data)
    print(f"Created questions DataFrame with {len(questions_df)} rows")
    print(f"Columns: {list(questions_df.columns)}")
    
    # Save questions
    save_with_symlink(
        questions_df,
        config.questions_path,
        config.local_data_dir / "questions.csv",
        save_first10=True
    )
    
    # Create pairwise comparisons
    print("\nCreating pairwise comparisons...")
    comparisons_data: list[dict] = []
    
    for _, row in questions_df.iterrows():
        question_id = row['question_id']
        prompt = row['prompt']
        num_responses = row['num_responses']
        
        # Collect responses and find chosen one
        responses: dict[str, dict] = {}
        chosen_letter: str | None = None
        
        for i in range(num_responses):
            letter = chr(ord('A') + i)
            response_col = f'response_{letter}'
            
            if response_col in row and pd.notna(row[response_col]):
                is_chosen = row.get(f'response_{letter}_chosen', False)
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
        if chosen_letter is None:
            continue
        
        chosen_resp = responses[chosen_letter]
        
        for letter, resp in responses.items():
            if letter == chosen_letter:
                continue
            
            # Always put chosen response as response_2 (human prefers 2)
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
                'human_preferred': '2',  # Human always prefers response_2
                'interaction_id': row['interaction_id'],
                'conversation_id': row['conversation_id'],
            })
    
    comparisons_df = pd.DataFrame(comparisons_data)
    print(f"Created {len(comparisons_df)} pairwise comparisons")
    print(f"Columns: {list(comparisons_df.columns)}")
    
    questions_with_comparisons = comparisons_df['question_id'].nunique()
    print(f"Questions with valid comparisons: {questions_with_comparisons}")
    print(f"Questions skipped (no chosen response): {len(questions_df) - questions_with_comparisons}")
    
    # Save pairwise comparisons
    save_with_symlink(
        comparisons_df,
        config.questions_pairwise_path,
        config.local_data_dir / "questions_pairwise.csv",
        save_first10=True
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total questions: {len(questions_df)}")
    print(f"Total pairwise comparisons: {len(comparisons_df)}")
    print(f"\nResponses per question distribution:")
    print(questions_df['num_responses'].value_counts().sort_index())
    print(f"\nConversation types:")
    print(questions_df['conversation_type'].value_counts())


if __name__ == "__main__":
    main()
