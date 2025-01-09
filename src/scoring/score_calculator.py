"""
Calculate overall score based on melody and phonetic matching
"""

import numpy as np
from typing import Dict, Optional
import logging

class ScoreCalculator:
    def __init__(self, melody_weight: float = 0.6, phonetic_weight: float = 0.4):
        """Initialize score calculator with weights
        
        Args:
            melody_weight: Weight for melody score (0-1)
            phonetic_weight: Weight for phonetic score (0-1)
        """
        self.set_weights(melody_weight, phonetic_weight)
        
    def set_weights(self, melody_weight: float, phonetic_weight: float):
        """Set weights for score components
        
        Args:
            melody_weight: Weight for melody score (0-1)
            phonetic_weight: Weight for phonetic score (0-1)
        """
        # Normalize weights to sum to 1
        total = melody_weight + phonetic_weight
        self.melody_weight = melody_weight / total
        self.phonetic_weight = phonetic_weight / total
        logging.info(f"Score weights set - Melody: {self.melody_weight:.2f}, Phonetic: {self.phonetic_weight:.2f}")
        
    def calculate(self, melody_score: float, phonetic_score: float) -> float:
        """Calculate total score from component scores
        
        Args:
            melody_score: Melody matching score (0-100)
            phonetic_score: Phonetic matching score (0-100)
            
        Returns:
            Total score (0-100)
        """
        try:
            # Ensure scores are in valid range
            melody_score = max(0.0, min(100.0, melody_score))
            phonetic_score = max(0.0, min(100.0, phonetic_score))
            
            # Calculate weighted sum
            total_score = (
                self.melody_weight * melody_score +
                self.phonetic_weight * phonetic_score
            )
            
            # Log scores
            logging.info(
                f"Scores - Melody: {melody_score:.1f}, "
                f"Phonetic: {phonetic_score:.1f}, "
                f"Total: {total_score:.1f}"
            )
            
            return total_score
            
        except Exception as e:
            logging.error(f"Error calculating total score: {e}")
            logging.exception("Full traceback:")
            return 0.0
