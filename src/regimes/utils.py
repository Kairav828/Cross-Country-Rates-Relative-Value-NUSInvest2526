import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

def get_continuous_regime_periods(
    regime_series: pd.Series, 
    target_state: int, 
    min_duration: int = 5
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Identify continuous time spells where the regime stays in target_state.
    Returns list of (start_date, end_date) tuples.
    """
    # Create boolean mask
    is_state = (regime_series == target_state)
    
    # Identify run starts and ends
    # A run starts where is_state is True and previous was False
    # A run ends where is_state is True and next is False
    # Using integer diffs
    
    int_series = is_state.astype(int)
    diffs = int_series.diff()
    
    starts = int_series.index[diffs == 1]
    # diffs == -1 marks the first index *after* leaving target_state.
    # We want the last index that is still in target_state, i.e., the
    # previous index before the -1 transition.
    transition_end_pos = np.where(diffs == -1)[0]
    ends = int_series.index[transition_end_pos - 1]
    
    # Handle edge cases (starts at index 0, ends at index -1)
    if is_state.iloc[0]:
        starts = starts.insert(0, int_series.index[0])
    if is_state.iloc[-1]:
        ends = ends.insert(len(ends), int_series.index[-1])
        
    spells = []
    for start, end in zip(starts, ends):
        # Calculate duration in business days (approx via len)
        segment = regime_series.loc[start:end]
        if len(segment) >= min_duration:
            spells.append((start, end))
            
    return spells

def get_regime_durations(regime_series: pd.Series) -> Dict[int, List[int]]:
    """
    Returns a dictionary of all duration lengths for each regime state.
    Used for calculating persistence probabilities.
    """
    durations = {}
    unique_states = sorted(regime_series.unique())
    
    # Simple run-length encoding
    values = regime_series.values
    # Locations of switches
    switches = np.where(np.diff(values) != 0)[0] + 1
    split_indices = np.concatenate([[0], switches, [len(values)]])
    
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i+1]
        state = values[start]
        duration = end - start
        
        if state not in durations:
            durations[state] = []
        durations[state].append(duration)
        
    return durations