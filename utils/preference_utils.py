"""
Preference generation utilities for historical preferences project.

Provides functions for generating and parsing preferences using LLMs.
"""

from __future__ import annotations

from typing import Any

import torch

from config import ModelConfig


def parse_model_response(response: str) -> str:
    """
    Parse model response to extract preference choice.
    
    Args:
        response: Raw model response text
    
    Returns:
        '1' if option 1 preferred, '2' if option 2 preferred, '-1' if ambiguous/invalid
    """
    import re
    
    response_lower = response.lower().strip()
    
    # Empty response
    if not response_lower:
        return '-1'
    
    # Patterns that indicate choice 1 or 2
    # Handle various phrasings: "option 1", "ption 1" (truncated), "the 1 option", etc.
    patterns_1 = [
        r'\boption\s*1\b',           # "option 1"
        r'\bption\s*1\b',            # "ption 1" (truncated)
        r'\btion\s*1\b',             # "tion 1" (more truncated)
        r'\bthe\s+1\s+option\b',     # "the 1 option"
        r'\b1\s*(?:st)?\s*option\b', # "1 option" or "1st option"
        r'\bchoice\s*1\b',           # "choice 1"
        r'\bresponse\s*1\b',         # "response 1"
        r'\banswer\s*1\b',           # "answer 1"
        r'\boption\s+is\s+1\b',      # "option is 1"
        r'\bfirst\s+option\b',       # "first option"
        r'\bfirst\s+response\b',     # "first response"
        r'\bfirst\s+choice\b',       # "first choice"
        r'\b(?:is|be)\s+1\b',        # "is 1" or "be 1"
        r'\bnumber\s*1\b',           # "number 1"
        r'#1\b',                     # "#1"
    ]
    
    patterns_2 = [
        r'\boption\s*2\b',           # "option 2"
        r'\bption\s*2\b',            # "ption 2" (truncated)
        r'\btion\s*2\b',             # "tion 2" (more truncated)
        r'\bthe\s+2\s+option\b',     # "the 2 option"
        r'\b2\s*(?:nd)?\s*option\b', # "2 option" or "2nd option"
        r'\bchoice\s*2\b',           # "choice 2"
        r'\bresponse\s*2\b',         # "response 2"
        r'\banswer\s*2\b',           # "answer 2"
        r'\boption\s+is\s+2\b',      # "option is 2"
        r'\bsecond\s+option\b',      # "second option"
        r'\bsecond\s+response\b',    # "second response"
        r'\bsecond\s+choice\b',      # "second choice"
        r'\b(?:is|be)\s+2\b',        # "is 2" or "be 2"
        r'\bnumber\s*2\b',           # "number 2"
        r'#2\b',                     # "#2"
    ]
    
    has_1 = any(re.search(p, response_lower) for p in patterns_1)
    has_2 = any(re.search(p, response_lower) for p in patterns_2)
    
    if has_1 and not has_2:
        return '1'
    elif has_2 and not has_1:
        return '2'
    elif has_1 and has_2:
        return '-1'  # Both mentioned - ambiguous
    
    # Fallback: check if response starts with '1' or '2'
    if response_lower.startswith('1'):
        return '1'
    elif response_lower.startswith('2'):
        return '2'
    
    # Secondary fallback: look for standalone '1' or '2' in very short responses
    # (only for responses under 15 chars to avoid false positives like "There are 2 ways")
    if len(response_lower) < 15:
        has_digit_1 = bool(re.search(r'\b1\b', response_lower))
        has_digit_2 = bool(re.search(r'\b2\b', response_lower))
        
        if has_digit_1 and not has_digit_2:
            return '1'
        elif has_digit_2 and not has_digit_1:
            return '2'
    
    return '-1'  # Invalid/unparseable response


def generate_preference(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response_1: str,
    response_2: str,
    config: ModelConfig | None = None,
    user_profile: str | None = None
) -> str:
    """
    Generate a preference between two responses using the model.
    
    Args:
        model: The loaded LLM
        tokenizer: The model's tokenizer
        prompt: The original question/prompt
        response_1: First response option
        response_2: Second response option
        config: Model configuration (optional, uses defaults if not provided)
        user_profile: Optional user profile description to condition the response
    
    Returns:
        '1' if option 1 preferred, '2' if option 2 preferred, '-1' if invalid
    """
    if config is None:
        config = ModelConfig()
    
    # Build comparison prompt with optional user profile
    if user_profile:
        comparison_content = (
            f"You are: {user_profile}\n\n"
            f"Given your perspective, which response do you prefer?\n\n"
            f"Question: {prompt}\n\n"
            f"Option 1: {response_1}\n\n"
            f"Option 2: {response_2}\n\n"
            f"Answer with only the number 1 or 2."
        )
    else:
        comparison_content = (
            f"Question: {prompt}\n\n"
            f"Option 1: {response_1}\n\n"
            f"Option 2: {response_2}\n\n"
            f"Which option is better? Answer with only the number 1 or 2."
        )
    
    # Use chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": comparison_content}]
        comparison_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        comparison_prompt = (
            f"{comparison_content}\n\n"
            "Answer with just the number \"1\" or just the number \"2\". "
            "Do not include any other text."
        )
    
    # Tokenize
    inputs = tokenizer(comparison_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(comparison_prompt):].strip()
    
    return parse_model_response(response)


def build_preference_record(
    row: dict[str, Any],
    choice: str,
    order: str,
    run_num: int,
    include_human_preferred: bool = True,
    user_id: int | None = None
) -> dict[str, Any]:
    """
    Build a preference record dictionary.
    
    Args:
        row: Original comparison row data
        choice: Model's choice ('1', '2', or '-1')
        order: 'original' or 'reversed'
        run_num: Run number (1-indexed)
        include_human_preferred: Whether to include human_preferred field
        user_id: Optional user profile ID for multi-user generation
    
    Returns:
        Dictionary with preference record fields
    """
    is_reversed = order == 'reversed'
    
    # Determine response IDs based on order
    if is_reversed:
        resp_1_id = row['response_2_id']
        resp_2_id = row['response_1_id']
        resp_1 = row['response_2']
        resp_2 = row['response_1']
    else:
        resp_1_id = row['response_1_id']
        resp_2_id = row['response_2_id']
        resp_1 = row['response_1']
        resp_2 = row['response_2']
    
    # Determine preferred/dispreferred based on choice
    if choice == '1':
        preferred_id = resp_1_id
        dispreferred_id = resp_2_id
    elif choice == '2':
        preferred_id = resp_2_id
        dispreferred_id = resp_1_id
    else:
        preferred_id = -1
        dispreferred_id = -1
    
    record = {
        'question_id': row['question_id'],
        'preferred_response_id': preferred_id,
        'dispreferred_response_id': dispreferred_id,
        'response_1_id': resp_1_id,
        'response_2_id': resp_2_id,
        'model_choice': choice,
        'prompt': row['prompt'],
        'response_1': resp_1,
        'response_2': resp_2,
        'order': order,
        'run': run_num,
    }
    
    # Add user_id if provided
    if user_id is not None:
        record['user_id'] = user_id
    
    # Handle human_preferred for PRISM dataset
    if include_human_preferred and 'human_preferred' in row:
        original_pref = row.get('human_preferred', '')
        if is_reversed and original_pref in ['1', '2']:
            record['human_preferred'] = '1' if original_pref == '2' else '2'
        else:
            record['human_preferred'] = original_pref
    
    return record

