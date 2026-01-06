#!/usr/bin/env python3
"""
Process the Community Alignment dataset to create pairwise comparison questions.

The CA dataset contains human-LLM conversations with 4 model responses per prompt.
This script extracts unique questions and creates all pairwise comparisons.

Usage:
    python ca/process_questions.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_dataset_config
from utils.file_utils import save_with_symlink


def parse_responses(response_text: str) -> dict[str, str]:
    """
    Parse the standard Community Alignment response text into individual responses.
    
    The CA format has responses labeled as:
    # Response A:
    <response text>
    # Response B:
    ...
    
    Args:
        response_text: Raw response text from the dataset
    
    Returns:
        Dictionary mapping response labels (A, B, C, D) to response text
    """
    responses: dict[str, str] = {}
    current_response: str | None = None
    current_text: list[str] = []
    
    for line in response_text.split('\n'):
        if line.startswith('# Response A:'):
            if current_response:
                responses[current_response] = '\n'.join(current_text).strip()
            current_response = 'A'
            current_text = []
        elif line.startswith('# Response B:'):
            if current_response:
                responses[current_response] = '\n'.join(current_text).strip()
            current_response = 'B'
            current_text = []
        elif line.startswith('# Response C:'):
            if current_response:
                responses[current_response] = '\n'.join(current_text).strip()
            current_response = 'C'
            current_text = []
        elif line.startswith('# Response D:'):
            if current_response:
                responses[current_response] = '\n'.join(current_text).strip()
            current_response = 'D'
            current_text = []
        elif current_response:
            current_text.append(line)
    
    # Save last response
    if current_response:
        responses[current_response] = '\n'.join(current_text).strip()
    
    return responses


def main() -> None:
    """Main entry point."""
    # Get dataset configuration
    config = get_dataset_config("ca")
    config.ensure_dirs()
    
    # Load dataset
    print("Loading Community Alignment dataset...")
    dataset = load_dataset("facebook/community-alignment-dataset", split='filtered')
    dataset_en = dataset.filter(lambda x: x['assigned_lang'] == 'en')
    print(f"Loaded {dataset_en.num_rows} conversations in English")
    
    # Group by unique (prompt, responses) combinations
    print(f"\nDataset columns: {dataset_en.column_names}")
    print(f"Unique prompts: {len(set(dataset_en['first_turn_prompt']))}")
    
    questions_dict: dict[tuple[str, str], list[int]] = defaultdict(list)
    
    for conv_id, prompt, responses_text in zip(
        dataset_en['conversation_id'],
        dataset_en['first_turn_prompt'],
        dataset_en['first_turn_responses']
    ):
        questions_dict[(prompt, responses_text)].append(conv_id)
    
    print(f"\nUnique questions (first turn prompt-response combos): {len(questions_dict)}")
    
    # Create questions DataFrame
    questions_data: list[dict] = []
    for question_id, ((prompt, responses_text), conversation_ids) in enumerate(questions_dict.items()):
        parsed = parse_responses(responses_text)
        questions_data.append({
            'question_id': question_id,
            'prompt': prompt,
            'response_A': parsed.get('A', ''),
            'response_B': parsed.get('B', ''),
            'response_C': parsed.get('C', ''),
            'response_D': parsed.get('D', ''),
            'conversation_ids': '|'.join(map(str, conversation_ids))
        })
    
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
        
        responses = {
            'A': row['response_A'],
            'B': row['response_B'],
            'C': row['response_C'],
            'D': row['response_D']
        }
        
        # Create all 6 pairwise combinations (AB, AC, AD, BC, BD, CD)
        for (letter1, text1), (letter2, text2) in combinations(responses.items(), 2):
            comparisons_data.append({
                'question_id': question_id,
                'response_1_id': f"{question_id}{letter1}",
                'response_2_id': f"{question_id}{letter2}",
                'prompt': prompt,
                'response_1': text1,
                'response_2': text2,
                'conversation_ids': row['conversation_ids']
            })
    
    comparisons_df = pd.DataFrame(comparisons_data)
    assert len(comparisons_df) == len(questions_df) * 6, \
        "Number of pairwise comparisons should be questions * 6"
    
    print(f"Created {len(comparisons_df)} pairwise comparisons")
    print(f"Columns: {list(comparisons_df.columns)}")
    
    # Save pairwise comparisons
    save_with_symlink(
        comparisons_df,
        config.questions_pairwise_path,
        config.local_data_dir / "questions_pairwise.csv",
        save_first10=True
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total questions: {len(questions_df)}")
    print(f"Total pairwise comparisons: {len(comparisons_df)}")


if __name__ == "__main__":
    main()
