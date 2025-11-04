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
import subprocess
from huggingface_hub import HfApi
from evaluate_preferences import calculate_preference_consistency

# Model configuration
datapath = 'data/'

# Set HuggingFace cache to /scratch to avoid disk space issues
os.environ['HF_HOME'] = '/scratch/rachel/ProgressGym'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/rachel/ProgressGym'

def find_available_gpu():
    """Find the first GPU with low memory usage (< 10% used)."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, check=True)
        
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) == 3:
                gpu_idx, mem_used, mem_total = int(parts[0]), int(parts[1]), int(parts[2])
                mem_percent = (mem_used / mem_total) * 100
                if mem_percent < 10:  # Less than 10% memory used
                    print(f"Found available GPU {gpu_idx} with {mem_percent:.1f}% memory used")
                    return gpu_idx
        
        # If no low-usage GPU found, just return 0
        print("No low-usage GPU found, using GPU 0")
        return 0
    except Exception as e:
        print(f"Could not query GPU status: {e}. Using GPU 0")
        return 0

def find_latest_model_version(model_size, model_century):
    """Find the latest available model version."""
    api = HfApi()
    base_name = f"ProgressGym-HistLlama3-{model_size}-{model_century}-instruct"
    
    try:
        # Search for models matching the pattern
        models = api.list_models(author="PKU-Alignment", search=base_name)
        
        # Extract versions from model IDs
        versions = []
        for model in models:
            model_id = getattr(model, 'id', None) or getattr(model, 'modelId', None)
            if model_id and base_name in model_id:
                # Extract version from model ID (e.g., "v0.2" from "ProgressGym-HistLlama3-8B-C013-instruct-v0.2")
                version_part = model_id.split('-')[-1]
                if version_part.startswith('v'):
                    try:
                        # Parse version as float for comparison (e.g., "v0.2" -> 0.2)
                        version_num = float(version_part[1:])
                        versions.append((version_num, version_part))
                    except ValueError:
                        pass
        
        if versions:
            # Sort by version number and return the latest
            versions.sort(reverse=True)
            latest_version = versions[0][1]
            print(f"Found versions: {[v[1] for v in versions]}, using {latest_version}")
            return latest_version
        else:
            # Fallback to v0.2 if no versions found
            print("No matching model versions found, using v0.2")
            return "v0.2"
            
    except Exception as e:
        print(f"Could not query HuggingFace for model versions: {e}")
        print("Using default v0.2")
        return "v0.2"

def load_model_and_tokenizer(model_size, model_century):
    """Load the model and tokenizer from HuggingFace."""
    # Find latest version
    version = find_latest_model_version(model_size, model_century)
    MODEL_NAME = f"PKU-Alignment/ProgressGym-HistLlama3-{model_size}-{model_century}-instruct-{version}"
    
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

Which option is better? Answer with just the number "1" or just the number "2". Do not include any other text."""
    
    # Tokenize
    inputs = tokenizer(comparison_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.9,
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
    parser.add_argument("--model_size", type=str, default="8B",
                       help="Model size (default: 8B, options: 8B, 70B)")
    parser.add_argument("--model_century", type=str, default="C013",
                       help="Model century (default: C013, options: C013, C014, C015, C016, C017, C018, C019, C020, C021")
    parser.add_argument("--input", type=str, default=datapath + "questions_pairwise.csv",
                       help="Input CSV file with comparisons")
    parser.add_argument("--n_samples", type=int, default=10,
                       help="Number of comparisons to process (default: 10)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index for processing (default: 0)")
    parser.add_argument("-n", "--n_runs", type=int, default=1,
                       help="Number of times to run each question in each order (default: 1)")
    args = parser.parse_args()
    
    # Find available GPU and set CUDA_VISIBLE_DEVICES
    available_gpu = find_available_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpu)
    print(f"Using GPU {available_gpu}")
    
    # Load comparisons
    print(f"Loading comparisons from {args.input}")
    comparisons_df = pd.read_csv(args.input, sep='\t')
    print(f"Loaded {len(comparisons_df)} total comparisons")
    
    # Select sample to process
    end_idx = min(args.start_idx + args.n_samples, len(comparisons_df))
    sample_df = comparisons_df.iloc[args.start_idx:end_idx].copy()
    print(f"Processing comparisons {args.start_idx} to {end_idx}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_size, args.model_century)
    
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
    filename = f"preferences_model_{args.model_size}_{args.model_century}_pairs{args.start_idx}-{end_idx}_{args.n_runs}runs.csv"
    
    # Create output directory if it doesn't exist
    output_dir = Path(datapath) / 'prefs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    preferences_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved preferences to {output_path}")
    
    # Calculate and report preference consistency
    consistency_df = calculate_preference_consistency(preferences_df)
    print("\n" + "="*80)
    overall_consistency = consistency_df['preference_consistency'].mean()
    print(f"Overall average consistency: {overall_consistency:.2f}%")
        
    print("="*80 + "\n")

if __name__ == "__main__":
    main()