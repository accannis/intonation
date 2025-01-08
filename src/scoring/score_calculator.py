"""
Calculate overall score based on melody and phonetic matching
"""

import numpy as np
from typing import Dict, Optional
import logging

class ScoreCalculator:
    def __init__(self, melody_weight: float = 0.6, phonetic_weight: float = 0.4):
        """Initialize score calculator with weights"""
        self.set_weights(melody_weight, phonetic_weight)
        
    def set_weights(self, melody_weight: float, phonetic_weight: float):
        """Set weights for score components"""
        # Normalize weights to sum to 1
        total = melody_weight + phonetic_weight
        self.melody_weight = melody_weight / total
        self.phonetic_weight = phonetic_weight / total
        logging.info(f"Score weights set - Melody: {self.melody_weight:.2f}, Phonetic: {self.phonetic_weight:.2f}")
        
    def calculate_total_score(self, melody_score: float, phonetic_score: float) -> float:
        """Calculate total score from component scores"""
        try:
            # Ensure scores are in valid range
            melody_score = max(0.0, min(100.0, melody_score))
            phonetic_score = max(0.0, min(100.0, phonetic_score))
            
            # Calculate weighted sum
            total_score = (
                self.melody_weight * melody_score +
                self.phonetic_weight * phonetic_score
            )
            
            return total_score
            
        except Exception as e:
            logging.error(f"Error calculating total score: {e}")
            return 0.0
