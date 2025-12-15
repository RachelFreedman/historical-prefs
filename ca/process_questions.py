# %%
#!/usr/bin/env python3
from datasets import load_dataset
import pandas as pd
from utils import parse_responses

datapath = 'data/'
|# %%
# Load the dataset
print("Loading Community Alignment dataset...")
dataset = load_dataset("facebook/community-alignment-dataset", split='filtered')
dataset_en = dataset.filter(lambda x: x['assigned_lang'] == 'en')
print(f"Loaded {dataset_en.num_rows} conversations in English")

# %%
# Explore the dataset
print(f"Dataset columns: {dataset_en.column_names}")
print(f"Unique prompts: {len(set(dataset_en['first_turn_prompt']))}")

# Group by unique (prompt, responses) combinations and collect conversation_ids
from collections import defaultdict

# Dictionary to track conversation_ids for each unique question
questions_dict = defaultdict(list)

for conv_id, prompt, responses_text in zip(
    dataset_en['conversation_id'], 
    dataset_en['first_turn_prompt'], 
    dataset_en['first_turn_responses']
):
    questions_dict[(prompt, responses_text)].append(conv_id)

print(f"\nNumber of unique questions (first turn prompt-response combos): {len(questions_dict)}")
# 8687 unique questions

# %%

# Parse responses and create DataFrame with 7 columns 
questions_data = []
for question_id, ((prompt, responses_text), conversation_ids) in enumerate(questions_dict.items()):
    parsed_responses = parse_responses(responses_text)
    questions_data.append({
        'question_id': question_id,
        'prompt': prompt,
        'response_A': parsed_responses.get('A', ''),
        'response_B': parsed_responses.get('B', ''),
        'response_C': parsed_responses.get('C', ''),
        'response_D': parsed_responses.get('D', ''),
        'conversation_ids': '|'.join(map(str, conversation_ids))  # Join multiple IDs with pipe
    })

questions_df = pd.DataFrame(questions_data)
print(f"Created questions DataFrame with {len(questions_df)} rows and columns: {list(questions_df.columns)}")

# Save to CSV (tab-separated values) to| avoid issues with commas/semicolons in text
questions_df.to_csv(datapath + 'questions.csv', sep='\t', index=False)
questions_df[:10].to_csv(datapath + 'questions_first10.csv', sep='\t', index=False)
print("Saved questions to questions.csv")

# %%
# Create dataset of pairwise comparison questions
from itertools import combinations

# Read the questions data
questions_df = pd.read_csv(datapath + 'questions.csv', sep='\t')
print(f"Loaded {len(questions_df)} questions from questions.csv")

# Create all pairwise comparisons
comparisons_data = []

for _, row in questions_df.iterrows():
    question_id = row['question_id']
    prompt = row['prompt']
    
    # Get all 4 responses
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
assert len(comparisons_df) == len(questions_df) * 6, "Number of pairwise comparisons does not match number of questions * 6"
print(f"\nCreated pairwise comparisons DataFrame with {len(comparisons_df)} rows")
print(f"\nColumns: {list(comparisons_df.columns)}")

# Save to CSV
comparisons_df.to_csv(datapath + 'questions_pairwise.csv', sep='\t', index=False)
comparisons_df[:10].to_csv(datapath + 'questions_pairwise_first10.csv', sep='\t', index=False)
print(f"\nSaved pairwise comparisons to questions_pairwise.csv")
