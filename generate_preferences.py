#!/usr/bin/env python3
"""
Generate preferences using ProgressGym historical LLMs.

This unified script handles preference generation for both CA and PRISM datasets.
It loads historical models and generates comparative assessments between response pairs.

Usage:
    python generate_preferences.py --dataset prism --model_size 8B --model_century C013
    python generate_preferences.py --dataset ca --model_size 8B --model_century C013 --n_samples 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DatasetName,
    ModelConfig,
    configure_environment,
    get_dataset_config,
)
from utils import (
    CheckpointManager,
    find_available_gpu,
    generate_preference,
    load_model_and_tokenizer,
    save_with_symlink,
)
from utils.preference_utils import build_preference_record


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate preferences using historical LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["ca", "prism"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--model_size", 
        type=str, 
        default="8B",
        choices=["8B", "70B"],
        help="Model size"
    )
    parser.add_argument(
        "--model_century", 
        type=str, 
        default="C013",
        choices=ModelConfig.VALID_CENTURIES,
        help="Model century"
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=10,
        help="Number of comparisons to process"
    )
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=0,
        help="Starting index for processing"
    )
    parser.add_argument(
        "-n", "--n_runs", 
        type=int, 
        default=1,
        help="Number of times to run each comparison in each order"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default=None,
        help="Override input CSV path"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Configure environment
    configure_environment()
    
    # Get dataset configuration
    dataset_config = get_dataset_config(args.dataset)
    dataset_config.ensure_dirs()
    
    # Find available GPU
    available_gpu = find_available_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpu)
    print(f"Using GPU {available_gpu}")
    
    # Determine input path
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = dataset_config.questions_pairwise_path
    
    # Load comparisons
    print(f"Loading comparisons from {input_path}")
    comparisons_df = pd.read_csv(input_path, sep='\t')
    print(f"Loaded {len(comparisons_df)} total comparisons")
    
    # Select sample to process
    end_idx = min(args.start_idx + args.n_samples, len(comparisons_df))
    sample_df = comparisons_df.iloc[args.start_idx:end_idx].copy()
    print(f"Processing comparisons {args.start_idx} to {end_idx}")
    
    # Setup output paths
    output_path = dataset_config.prefs_output_path(
        args.model_size, args.model_century,
        args.start_idx, end_idx, args.n_runs
    )
    checkpoint_path = dataset_config.checkpoint_path(
        args.model_size, args.model_century,
        args.start_idx, end_idx, args.n_runs
    )
    local_output_path = dataset_config.local_prefs_dir / output_path.name
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_path=checkpoint_path,
        n_runs=args.n_runs
    )
    checkpoint_mgr.load_checkpoint(sample_df)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_size, args.model_century)
    model_config = ModelConfig(size=args.model_size, century=args.model_century)
    
    # Check if human_preferred column exists (PRISM has it, CA doesn't)
    include_human_preferred = 'human_preferred' in comparisons_df.columns
    
    # Generate preferences
    processed_count = len(checkpoint_mgr.processed_indices)
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), 
                         desc="Generating preferences", initial=processed_count):
        # Skip already processed
        if checkpoint_mgr.is_processed(idx):
            continue
        
        row_dict = row.to_dict()
        
        # Run original order n_runs times
        for run_num in range(args.n_runs):
            try:
                choice = generate_preference(
                    model, tokenizer,
                    row['prompt'],
                    row['response_1'],
                    row['response_2'],
                    config=model_config
                )
            except Exception as e:
                print(f"\nError processing row {idx} (original, run {run_num + 1}): {e}")
                choice = '-1'
            
            record = build_preference_record(
                row_dict, choice, 'original', run_num + 1,
                include_human_preferred=include_human_preferred
            )
            checkpoint_mgr.add_preference(record)
        
        # Run reversed order n_runs times
        for run_num in range(args.n_runs):
            try:
                choice = generate_preference(
                    model, tokenizer,
                    row['prompt'],
                    row['response_2'],  # Swap
                    row['response_1'],  # Swap
                    config=model_config
                )
            except Exception as e:
                print(f"\nError processing row {idx} (reversed, run {run_num + 1}): {e}")
                choice = '-1'
            
            record = build_preference_record(
                row_dict, choice, 'reversed', run_num + 1,
                include_human_preferred=include_human_preferred
            )
            checkpoint_mgr.add_preference(record)
        
        # Maybe save checkpoint
        checkpoint_mgr.maybe_save_checkpoint()
    
    # Save final results
    preferences_df = checkpoint_mgr.get_preferences_df()
    
    try:
        save_with_symlink(preferences_df, output_path, local_output_path)
        checkpoint_mgr.cleanup()
    except OSError as e:
        if e.errno == 122:  # Disk quota exceeded
            print(f"\n[ERROR] Disk quota exceeded when saving to {output_path}")
            alt_path = output_path.parent.parent / output_path.name
            preferences_df.to_csv(alt_path, sep='\t', index=False)
            print(f"[SUCCESS] Saved to alternative location: {alt_path}")
        else:
            raise
    
    # Report consistency
    try:
        from evaluate_preferences import calculate_preference_consistency
        consistency_df = calculate_preference_consistency(preferences_df)
        print("\n" + "=" * 80)
        print(f"Overall average consistency: {consistency_df['preference_consistency'].mean():.2f}%")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"Could not calculate consistency: {e}")


if __name__ == "__main__":
    main()

