"""
Match melodies using Dynamic Time Warping (DTW)
"""

import numpy as np
from typing import Tuple, List
import logging
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

class MelodyMatcher:
    def __init__(self, window_size: float = 0.5):
        """Initialize melody matcher
        
        Args:
            window_size: Size of the sliding window in seconds
        """
        self.window_size = window_size
        
    def match(self, reference_melody: np.ndarray, input_melody: np.ndarray) -> Tuple[List[float], List[float]]:
        """Match input melody against reference melody using DTW
        
        Args:
            reference_melody: Reference melody pitch contour
            input_melody: Input melody pitch contour
            
        Returns:
            Tuple of (scores, times) where:
                scores: List of similarity scores (0-100) for each window
                times: List of time points corresponding to the scores
        """
        try:
            # Ensure arrays are 1D
            reference_melody = reference_melody.flatten()
            input_melody = input_melody.flatten()
            
            # Handle empty arrays
            if len(reference_melody) == 0 or len(input_melody) == 0:
                return [0.0], [0.0]
            
            # Normalize pitch values to 0-1 range
            ref_min, ref_max = np.min(reference_melody), np.max(reference_melody)
            input_min, input_max = np.min(input_melody), np.max(input_melody)
            
            # Handle constant values
            if ref_max == ref_min:
                ref_norm = np.zeros_like(reference_melody)
            else:
                ref_norm = (reference_melody - ref_min) / (ref_max - ref_min)
                
            if input_max == input_min:
                input_norm = np.zeros_like(input_melody)
            else:
                input_norm = (input_melody - input_min) / (input_max - input_min)
            
            # Calculate window size in samples
            window_samples = max(int(len(input_melody) * self.window_size), 1)
            step_size = max(window_samples // 2, 1)
            
            # Initialize lists for scores and times
            scores = []
            times = []
            
            # Slide window over input melody
            for i in range(0, len(input_norm) - window_samples, step_size):
                window = input_norm[i:i + window_samples]
                
                # Calculate DTW distance
                distance = dtw.distance(ref_norm, window)
                
                # Convert distance to similarity score (0-100)
                max_distance = np.sqrt(len(ref_norm))  # Maximum possible DTW distance
                score = max(0, 100 * (1 - distance / max_distance))
                
                scores.append(score)
                times.append(i / len(input_norm))  # Normalize time to 0-1 range
            
            # If no scores were calculated, return default
            if not scores:
                return [0.0], [0.0]
                
            return scores, times
            
        except Exception as e:
            logging.error(f"Error matching melodies: {e}")
            logging.exception("Full traceback:")
            return [0.0], [0.0]
