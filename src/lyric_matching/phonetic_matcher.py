"""
Match phonetic features using MFCC similarity
"""

import numpy as np
from typing import Tuple, List
import logging
from scipy.spatial.distance import cdist

class PhoneticMatcher:
    def __init__(self, window_size: float = 1.0):
        """Initialize phonetic matcher
        
        Args:
            window_size: Size of the sliding window in seconds
        """
        self.window_size = window_size
        
    def match(self, reference_mfcc: np.ndarray, input_mfcc: np.ndarray) -> Tuple[List[float], List[float]]:
        """Match input MFCCs against reference MFCCs
        
        Args:
            reference_mfcc: Reference MFCC features (n_mfcc x time)
            input_mfcc: Input MFCC features to compare (n_mfcc x time)
            
        Returns:
            Tuple of (scores, times) where:
                scores: List of similarity scores (0-100) for each window
                times: List of time points corresponding to the scores
        """
        try:
            # Get dimensions
            n_mfcc, ref_frames = reference_mfcc.shape
            _, input_frames = input_mfcc.shape
            
            # Calculate window size in frames
            frames_per_second = input_frames / ref_frames
            window_frames = max(int(self.window_size * frames_per_second), 2)  # Ensure minimum window size
            step_size = max(window_frames // 2, 1)  # Ensure minimum step size
            
            # Initialize lists for scores and times
            scores = []
            times = []
            
            # Slide window over input MFCCs
            for i in range(0, input_frames - window_frames, step_size):
                # Get current window
                window = input_mfcc[:, i:i + window_frames]
                
                # Calculate cosine distance between window and reference
                distances = cdist(window.T, reference_mfcc.T, metric='cosine')
                
                # Get minimum distance for this window
                min_distance = np.min(distances)
                
                # Convert distance to similarity score (0-100)
                # Cosine distance ranges from 0 (identical) to 2 (opposite)
                score = 100 * (1 - min_distance / 2)
                
                scores.append(score)
                times.append(i / frames_per_second)
                
            return scores, times
            
        except Exception as e:
            logging.error(f"Error matching phonetic features: {e}")
            logging.exception("Full traceback:")
            return [], []
