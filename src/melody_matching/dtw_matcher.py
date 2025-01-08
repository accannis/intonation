import numpy as np
from dtaidistance import dtw
from typing import Tuple, List, Dict, Optional
import logging
import librosa

class MelodyMatcher:
    def __init__(self, window_size: int = None):
        self.window_size = window_size
        self.reference_features = None
        self.reference_pitch = None
        
    def set_reference(self, features: Dict[str, np.ndarray]):
        """Set reference features for comparison"""
        try:
            self.reference_features = features
            if 'pitch' in features:
                self.reference_pitch = self.normalize_pitch_sequence(features['pitch'])
            logging.info("Reference features set for melody matching")
        except Exception as e:
            logging.error(f"Error setting reference features: {e}")
            logging.exception("Full traceback:")
    
    def normalize_pitch_sequence(self, pitch_sequence: np.ndarray, reference_pitch: float = None) -> np.ndarray:
        """
        Normalize pitch sequence to semitones relative to reference pitch
        """
        if reference_pitch is None or reference_pitch <= 0:
            # Use median of non-zero values as reference if not provided
            non_zero = pitch_sequence[pitch_sequence > 0]
            if len(non_zero) == 0:
                return np.zeros_like(pitch_sequence)  # Return zeros if no valid pitches
            reference_pitch = np.median(non_zero)
            
        # Replace zeros with small value to avoid log(0)
        safe_pitch = np.where(pitch_sequence > 0, pitch_sequence, reference_pitch * 1e-6)
        normalized = 12 * np.log2(safe_pitch / reference_pitch)
        return normalized
    
    def compute_melody_similarity(self, 
                                reference_pitch: np.ndarray,
                                performance_pitch: np.ndarray) -> float:
        """
        Compute similarity between two pitch sequences using DTW
        
        Args:
            reference_pitch: Reference pitch sequence (already normalized)
            performance_pitch: Performance pitch sequence (already normalized)
            
        Returns:
            Similarity score between 0 and 1
        """
        # Handle edge cases
        if len(reference_pitch) == 0 or len(performance_pitch) == 0:
            return 0.0
            
        # Reshape sequences for DTW
        ref_seq = reference_pitch.reshape(-1, 1)
        perf_seq = performance_pitch.reshape(-1, 1)
        
        try:
            # Compute DTW distance
            distance = dtw.distance(ref_seq, perf_seq)
            
            # Convert distance to similarity score (0 to 1)
            max_distance = np.sqrt(len(ref_seq) * len(perf_seq))  # Theoretical maximum distance
            similarity = 1.0 - (distance / max_distance)
            
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logging.error(f"Error computing DTW distance: {str(e)}")
            return 0.0

    def calculate_score(self, features: Optional[Dict[str, np.ndarray]]) -> float:
        """Calculate melody score by comparing with reference"""
        try:
            if features is None or self.reference_pitch is None:
                return 0.0
                
            if 'pitch' not in features:
                return 0.0
                
            # Normalize performance pitch
            performance_pitch = self.normalize_pitch_sequence(features['pitch'])
            
            # Compute similarity
            similarity = self.compute_melody_similarity(self.reference_pitch, performance_pitch)
            
            # Convert to score (0-100)
            score = similarity * 100.0
            
            return score
            
        except Exception as e:
            logging.error(f"Error calculating melody score: {e}")
            return 0.0
