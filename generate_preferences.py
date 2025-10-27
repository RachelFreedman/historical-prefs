#!/usr/bin/env python3
"""
Generate preferences using the ProgressGym historical model.
This script loads the model and generates responses to pairwise comparison questions.

The ProgressGym model is a 13th-century historical LLM based on Llama-3-8B,
trained on historical text data to reflect moral perspectives from that era.
It can be used to generate comparative assessments between responses.
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import argparse
from pathlib import Path
import os

# Model configuration
MODEL_NAME = "PKU-Alignment/ProgressGym-HistLlama3-8B-C013-instruct-v0.2"
datapath = 'data/'

# Set HuggingFace cache to /scratch to avoid disk space issues
os.environ['HF_HOME'] = '/scratch/rachel/ProgressGym'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/rachel/ProgressGym'

def load_model_and_tokenizer():
    """Load the model and tokenizer from HuggingFace."""
    print(f"Loading model and tokenizer: {MODEL_NAME}")
    print("This may take a while on first run as it downloads the model...")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir='/scratch/rachel/ProgressGym'
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir='/scratch/rachel/ProgressGym'
    )
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_preference(model, tokenizer, prompt, response_1, response_2):
    """
    Generate a preference between two responses using the model.
    
    Returns:
        str: The model's choice ('1' or '2')
    """
    # Try using chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [
            {"role": "user", "content": f"Question: {prompt}\n\nOption 1: {response_1}\n\nOption 2: {response_2}\n\nWhich option is better? Answer with only the number 1 or 2."}
        ]
        comparison_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback to simple prompt
        comparison_prompt = f"""Question: {prompt}

Option 1: {response_1}

Option 2: {response_2}

Which option is better? Answer with the number 1 or 2, nothing else."""
    
    # Tokenize
    inputs = tokenizer(comparison_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new generated part (after the prompt)
    response = generated_text[len(comparison_prompt):].strip()
    
    # Extract just the number from the response - check for option mentions
    response_lower = response.lower().strip()
        
    # Check if response includes "option 1" or "option 2"
    has_option_1 = 'option 1' in response_lower
    has_option_2 = 'option 2' in response_lower
    
    if has_option_1 and not has_option_2:
        return '1'
    elif has_option_2 and not has_option_1:
        return '2'
    elif has_option_1 and has_option_2:
        return '-1'  # Both mentioned - ambiguous
    else:
        # Fallback to checking if response starts with '1' or '2'
        if response_lower.startswith('1'):
            return '1'
        elif response_lower.startswith('2'):
            return '2'
        else:
            return '-1'  # Invalid response

def main():
    parser = argparse.ArgumentParser(description="Generate preferences using historical LLM")
    parser.add_argument("--input", type=str, default=datapath + "questions_pairwise_first10.csv",
                       help="Input CSV file with comparisons")
    parser.add_argument("--output", type=str, default=datapath + "preferences_first10.csv",
                       help="Output CSV file")
    parser.add_argument("--n_samples", type=int, default=10,
                       help="Number of comparisons to process (default: 10)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index for processing (default: 0)")
    parser.add_argument("-n", "--n_runs", type=int, default=1,
                       help="Number of times to run each question in each order (default: 1)")
    args = parser.parse_args()
    
    # Load comparisons
    print(f"Loading comparisons from {args.input}")
    comparisons_df = pd.read_csv(args.input, sep='\t')
    print(f"Loaded {len(comparisons_df)} total comparisons")
    
    # Select sample to process
    end_idx = args.start_idx + args.n_samples
    sample_df = comparisons_df.iloc[args.start_idx:end_idx].copy()
    print(f"Processing comparisons {args.start_idx} to {end_idx}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Generate preferences - ask each question n_runs times in each order
    preferences = []
    total_runs = len(sample_df) * 2 * args.n_runs
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Generating preferences"):
        # Run original order n_runs times
        for run_num in range(args.n_runs):
            try:
                choice = generate_preference(
                    model, tokenizer,
                    row['prompt'],
                    row['response_1'],
                    row['response_2']
                )
                
                # Parse choice into preferred/dispreferred IDs
                if choice == '1':
                    preferred_response_id = row['response_1_id']
                    dispreferred_response_id = row['response_2_id']
                elif choice == '2':
                    preferred_response_id = row['response_2_id']
                    dispreferred_response_id = row['response_1_id']
                else:  # choice == '-1' or invalid
                    preferred_response_id = -1
                    dispreferred_response_id = -1
                
                preferences.append({
                    'question_id': row['question_id'],
                    'preferred_response_id': preferred_response_id,
                    'dispreferred_response_id': dispreferred_response_id,
                    'response_1_id': row['response_1_id'],
                    'response_2_id': row['response_2_id'],
                    'model_choice': choice,
                    'prompt': row['prompt'],
                    'response_1': row['response_1'],
                    'response_2': row['response_2'],
                    'order': 'original',
                    'run': run_num + 1
                })
                
            except Exception as e:
                print(f"\nError processing row {idx} (original order, run {run_num + 1}): {e}")
                preferences.append({
                    'question_id': row['question_id'],
                    'model_choice': '-1',
                    'preferred_response_id': -1,
                    'dispreferred_response_id': -1,
                    'response_1_id': row['response_1_id'],
                    'response_2_id': row['response_2_id'],
                    'prompt': row['prompt'],
                    'response_1': row['response_1'],
                    'response_2': row['response_2'],
                    'order': 'original',
                    'run': run_num + 1
                })
        
        # Run reversed order n_runs times
        for run_num in range(args.n_runs):
            try:
                choice_reversed = generate_preference(
                    model, tokenizer,
                    row['prompt'],
                    row['response_2'],  # response_2 becomes option 1
                    row['response_1']   # response_1 becomes option 2
                )
                
                # Parse choice into preferred/dispreferred IDs (accounting for reversed order)
                if choice_reversed == '1':
                    preferred_response_id = row['response_2_id']  # response_2 was option 1
                    dispreferred_response_id = row['response_1_id']  # response_1 was option 2
                elif choice_reversed == '2':
                    preferred_response_id = row['response_1_id']  # response_1 was option 2
                    dispreferred_response_id = row['response_2_id']  # response_2 was option 1
                else:  # choice == '-1' or invalid
                    preferred_response_id = -1
                    dispreferred_response_id = -1
                
                preferences.append({
                    'question_id': row['question_id'],
                    'preferred_response_id': preferred_response_id,
                    'dispreferred_response_id': dispreferred_response_id,
                    'response_1_id': row['response_2_id'],  # Swap: response_2_id becomes response_1_id
                    'response_2_id': row['response_1_id'],  # Swap: response_1_id becomes response_2_id
                    'model_choice': choice_reversed,
                    'prompt': row['prompt'],
                    'response_1': row['response_2'],  # Swap: response_2 becomes response_1
                    'response_2': row['response_1'],  # Swap: response_1 becomes response_2
                    'order': 'reversed',
                    'run': run_num + 1
                })
                
            except Exception as e:
                print(f"\nError processing row {idx} (reversed order, run {run_num + 1}): {e}")
                preferences.append({
                    'question_id': row['question_id'],
                    'model_choice': '-1',
                    'preferred_response_id': -1,
                    'dispreferred_response_id': -1,
                    'response_1_id': row['response_2_id'],  # Swap: response_2_id becomes response_1_id
                    'response_2_id': row['response_1_id'],  # Swap: response_1_id becomes response_2_id
                    'prompt': row['prompt'],
                    'response_1': row['response_2'],  # Swap: response_2 becomes response_1
                    'response_2': row['response_1'],  # Swap: response_1 becomes response_2
                    'order': 'reversed',
                    'run': run_num + 1
                })
    
    # Save results
    preferences_df = pd.DataFrame(preferences)
    preferences_df.to_csv(args.output, sep='\t', index=False)
    print(f"\nSaved preferences to {args.output}")
    
if __name__ == "__main__":
    main()

