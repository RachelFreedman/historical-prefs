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
from collections import defaultdict
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
    
    # Setup checkpointing - save every 1000 comparisons to avoid data loss
    checkpoint_interval = 1000
    scratch_output_dir = Path('/scratch/rachel/historical-prefs/data/prefs')
    scratch_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create local output dir for symlink
    local_output_dir = Path(datapath) / 'prefs'
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"preferences_model_{args.model_size}_{args.model_century}_pairs{args.start_idx}-{end_idx}_{args.n_runs}runs.csv"
    scratch_output_path = scratch_output_dir / filename
    
    # Check if there's a checkpoint to resume from
    checkpoint_path = scratch_output_dir / f"{filename}.checkpoint"
    processed_indices = set()
    if checkpoint_path.exists():
        print(f"Found checkpoint file: {checkpoint_path}")
        print("Loading checkpoint...")
        checkpoint_df = pd.read_csv(checkpoint_path, sep='\t')
        print(f"Loaded {len(checkpoint_df)} preferences from checkpoint")
        # Convert checkpoint back to list of dicts
        preferences = checkpoint_df.to_dict('records')
        # Determine which comparisons have been fully processed
        # Each comparison generates 2 * n_runs preferences (original + reversed, each n_runs times)
        # We need to normalize the key since reversed order swaps response_1_id and response_2_id
        comparison_counts = defaultdict(int)
        for pref in preferences:
            # Normalize key: sort response IDs to handle both original and reversed orders
            question_id = pref['question_id']
            resp1 = pref['response_1_id']
            resp2 = pref['response_2_id']
            # Create normalized key with sorted response IDs
            normalized_key = (question_id, tuple(sorted([resp1, resp2])))
            comparison_counts[normalized_key] += 1
        
        # Find which comparisons are complete (have 2 * n_runs entries)
        for key, count in comparison_counts.items():
            if count >= 2 * args.n_runs:
                question_id, (resp1, resp2) = key[0], key[1]
                # Find the row index in sample_df that matches this comparison
                # (order of response_1_id and response_2_id doesn't matter)
                for df_idx, df_row in sample_df.iterrows():
                    if (df_row['question_id'] == question_id and 
                        set([df_row['response_1_id'], df_row['response_2_id']]) == set([resp1, resp2])):
                        processed_indices.add(df_idx)
                        break
        
        print(f"Resuming from {len(preferences)} preferences ({len(processed_indices)} comparisons already processed)")
    
    processed_count = len(processed_indices)
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Generating preferences", initial=processed_count):
        # Skip if this comparison was already processed
        if idx in processed_indices:
            continue
            
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
        
        # Periodic checkpointing to avoid data loss (after each comparison is complete)
        # Checkpoint every N comparisons to avoid losing too much work
        comparisons_completed = (len(preferences) - (len(preferences) % (2 * args.n_runs))) // (2 * args.n_runs)
        checkpoint_every_n_comparisons = max(100, checkpoint_interval // (2 * args.n_runs))  # Checkpoint every ~100 comparisons
        if comparisons_completed > 0 and comparisons_completed % checkpoint_every_n_comparisons == 0:
            try:
                checkpoint_df = pd.DataFrame(preferences)
                checkpoint_df.to_csv(checkpoint_path, sep='\t', index=False)
                print(f"\n[Checkpoint] Saved {len(preferences)} preferences ({comparisons_completed} comparisons) to {checkpoint_path}")
            except Exception as e:
                print(f"\n[Warning] Failed to save checkpoint: {e}")
                # If checkpoint fails due to disk space, try saving to a different location
                if "Disk quota exceeded" in str(e) or (hasattr(e, 'errno') and e.errno == 122):
                    alt_checkpoint = Path('/scratch/rachel') / f"{filename}.checkpoint"
                    try:
                        checkpoint_df = pd.DataFrame(preferences)
                        checkpoint_df.to_csv(alt_checkpoint, sep='\t', index=False)
                        print(f"[Checkpoint] Saved to alternative location: {alt_checkpoint}")
                    except:
                        pass
    
    # Save final results to scratch (has more space)
    preferences_df = pd.DataFrame(preferences)
    
    try:
        # Save to scratch first (more space available)
        preferences_df.to_csv(scratch_output_path, sep='\t', index=False)
        print(f"\nSaved preferences to {scratch_output_path}")
        
        # Create symlink in local directory for easy access
        local_output_path = local_output_dir / filename
        if local_output_path.exists() or local_output_path.is_symlink():
            local_output_path.unlink()
        local_output_path.symlink_to(scratch_output_path)
        print(f"Created symlink: {local_output_path} -> {scratch_output_path}")
        
        # Clean up checkpoint file if final save succeeded
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"Removed checkpoint file: {checkpoint_path}")
            
    except OSError as e:
        if "Disk quota exceeded" in str(e) or e.errno == 122:
            print(f"\n[ERROR] Disk quota exceeded when saving to {scratch_output_path}")
            print(f"[INFO] Data is still in memory. Attempting to save to alternative location...")
            # Try saving to a different location in scratch
            alt_path = Path('/scratch/rachel') / filename
            try:
                preferences_df.to_csv(alt_path, sep='\t', index=False)
                print(f"[SUCCESS] Saved to alternative location: {alt_path}")
                print(f"[INFO] You can move this file later or create a symlink")
            except Exception as e2:
                print(f"[CRITICAL] Failed to save to alternative location: {e2}")
                print(f"[INFO] {len(preferences)} preferences are still in memory")
                print(f"[INFO] Checkpoint file may exist at: {checkpoint_path}")
                raise
        else:
            raise
    
    # Calculate and report preference consistency
    consistency_df = calculate_preference_consistency(preferences_df)
    print("\n" + "="*80)
    overall_consistency = consistency_df['preference_consistency'].mean()
    print(f"Overall average consistency: {overall_consistency:.2f}%")
        
    print("="*80 + "\n")

if __name__ == "__main__":
    main()