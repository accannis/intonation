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
            
            # Normalize pitch values to 0-1 range
            ref_min, ref_max = np.min(reference_melody), np.max(reference_melody)
            input_min, input_max = np.min(input_melody), np.max(input_melody)
            
            ref_norm = (reference_melody - ref_min) / (ref_max - ref_min)
            input_norm = (input_melody - input_min) / (input_max - input_min)
            
            # Calculate window size in samples
            samples_per_second = len(input_melody) / len(reference_melody)
            window_samples = int(self.window_size * samples_per_second)
            
            # Initialize lists for scores and times
            scores = []
            times = []
            
            # Slide window over input melody
            for i in range(0, len(input_norm) - window_samples, window_samples // 2):
                # Get current window
                window = input_norm[i:i + window_samples]
                
                # Calculate DTW distance
                distance = dtw.distance(
                    ref_norm.astype(np.float64),
                    window.astype(np.float64)
                )
                
                # Convert distance to similarity score (0-100)
                # DTW distance is unbounded, so we use an exponential transform
                score = 100 * np.exp(-distance)
                
                # Calculate time point for this window
                time = i / samples_per_second
                
                scores.append(score)
                times.append(time)
            
            # Convert to numpy arrays
            scores = np.array(scores)
            times = np.array(times)
            
            # Ensure we have at least one score
            if len(scores) == 0:
                scores = np.array([0.0])
                times = np.array([0.0])
            
            logging.info(
                f"Melody matching complete - "
                f"Windows: {len(scores)}, "
                f"Duration: {times[-1]:.1f}s, "
                f"Mean score: {np.mean(scores):.1f}"
            )
            
            return scores.tolist(), times.tolist()
            
        except Exception as e:
            logging.error(f"Error matching melodies: {e}")
            logging.exception("Full traceback:")
            return [0.0], [0.0]
