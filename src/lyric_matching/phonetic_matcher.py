import numpy as np
from typing import Dict, Optional, List
import logging
import librosa

class PhoneticMatcher:
    def __init__(self):
        self.reference_features = None
        self.reference_mfccs = None
        
    def set_reference(self, features: Dict[str, np.ndarray]):
        """Set reference features for comparison"""
        try:
            self.reference_features = features
            if 'mfccs' in features:
                self.reference_mfccs = features['mfccs']
            logging.info("Reference features set for phonetic matching")
        except Exception as e:
            logging.error(f"Error setting reference features: {e}")
            logging.exception("Full traceback:")
            
    def compute_phonetic_similarity(self, ref_mfccs: np.ndarray, perf_mfccs: np.ndarray) -> float:
        """
        Compute similarity between two MFCC sequences
        
        Args:
            ref_mfccs: Reference MFCC features
            perf_mfccs: Performance MFCC features
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Ensure same number of time steps by truncating or padding
            min_length = min(ref_mfccs.shape[1], perf_mfccs.shape[1])
            ref_mfccs = ref_mfccs[:, :min_length]
            perf_mfccs = perf_mfccs[:, :min_length]
            
            # Compute cosine similarity for each time step
            similarities = []
            for t in range(min_length):
                ref_vec = ref_mfccs[:, t]
                perf_vec = perf_mfccs[:, t]
                
                # Normalize vectors
                ref_norm = np.linalg.norm(ref_vec)
                perf_norm = np.linalg.norm(perf_vec)
                
                if ref_norm > 0 and perf_norm > 0:
                    similarity = np.dot(ref_vec, perf_vec) / (ref_norm * perf_norm)
                    similarities.append(max(0, similarity))  # Only keep positive similarities
                    
            if not similarities:
                return 0.0
                
            # Return average similarity
            return np.mean(similarities)
            
        except Exception as e:
            logging.error(f"Error computing phonetic similarity: {e}")
            return 0.0

    def calculate_score(self, features: Optional[Dict[str, np.ndarray]]) -> float:
        """Calculate phonetic score by comparing with reference"""
        try:
            if features is None or self.reference_mfccs is None:
                return 0.0
                
            if 'mfccs' not in features:
                return 0.0
                
            # Compute similarity
            similarity = self.compute_phonetic_similarity(self.reference_mfccs, features['mfccs'])
            
            # Convert to score (0-100)
            score = similarity * 100.0
            
            return score
            
        except Exception as e:
            logging.error(f"Error calculating phonetic score: {e}")
            return 0.0
