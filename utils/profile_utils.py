"""
User profile utilities for historical preferences project.

Provides functions for loading and filtering user profiles from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_profiles(path: Path | str) -> list[dict[str, Any]]:
    """
    Load all user profiles from JSON file.
    
    Args:
        path: Path to profiles JSON file
        
    Returns:
        List of profile dictionaries
        
    Raises:
        FileNotFoundError: If profiles file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Profiles file not found: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data.get('profiles', [])


def get_profiles_for_century(
    profiles: list[dict[str, Any]], 
    century: str
) -> list[dict[str, Any]]:
    """
    Filter profiles to those belonging to a specific century.
    
    Args:
        profiles: List of all profile dictionaries
        century: Century code (e.g., "C013")
        
    Returns:
        List of profiles for the specified century
    """
    return [p for p in profiles if p.get('century') == century]


def get_profile_by_id(
    profiles: list[dict[str, Any]], 
    user_id: int,
    century: str | None = None
) -> dict[str, Any] | None:
    """
    Get a single profile by user ID, optionally filtered by century.
    
    Args:
        profiles: List of all profile dictionaries
        user_id: User ID to find
        century: Optional century code to filter by
        
    Returns:
        Profile dictionary if found, None otherwise
    """
    for p in profiles:
        if p.get('user_id') == user_id:
            if century is None or p.get('century') == century:
                return p
    return None


def get_user_ids_for_century(
    profiles: list[dict[str, Any]], 
    century: str
) -> list[int]:
    """
    Get list of user IDs available for a specific century.
    
    Args:
        profiles: List of all profile dictionaries
        century: Century code (e.g., "C013")
        
    Returns:
        List of user IDs for that century
    """
    century_profiles = get_profiles_for_century(profiles, century)
    return sorted(set(p.get('user_id') for p in century_profiles if p.get('user_id') is not None))


def parse_user_ids(
    user_ids_arg: str,
    profiles: list[dict[str, Any]],
    century: str
) -> list[int]:
    """
    Parse user_ids argument string into list of user IDs.
    
    Args:
        user_ids_arg: Either "all" or comma-separated IDs (e.g., "1,2,3")
        profiles: List of all profile dictionaries
        century: Century code to filter profiles
        
    Returns:
        List of user IDs to process
        
    Raises:
        ValueError: If user_ids_arg contains invalid IDs for the century
    """
    available_ids = get_user_ids_for_century(profiles, century)
    
    if not available_ids:
        raise ValueError(f"No user profiles found for century {century}")
    
    if user_ids_arg.lower() == 'all':
        return available_ids
    
    # Parse comma-separated IDs
    try:
        requested_ids = [int(x.strip()) for x in user_ids_arg.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid user_ids format: {user_ids_arg}. Use 'all' or comma-separated integers.") from e
    
    # Validate all requested IDs exist
    invalid_ids = set(requested_ids) - set(available_ids)
    if invalid_ids:
        raise ValueError(
            f"User IDs {invalid_ids} not found for century {century}. "
            f"Available IDs: {available_ids}"
        )
    
    return requested_ids

