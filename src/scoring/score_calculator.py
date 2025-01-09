"""
Calculate overall score from melody and phonetic scores
"""

import numpy as np
from typing import Optional

class ScoreCalculator:
    def __init__(self, melody_weight: float = 0.6, phonetic_weight: float = 0.4):
        """Initialize score calculator
        
        Args:
            melody_weight: Weight for melody score (0-1)
            phonetic_weight: Weight for phonetic score (0-1)
        """
        self.melody_weight = melody_weight
        self.phonetic_weight = phonetic_weight
        
    def calculate_total_score(self, melody_score: float, phonetic_score: float) -> float:
        """Calculate overall score from melody and phonetic scores
        
        Args:
            melody_score: Melody matching score (0-100)
            phonetic_score: Phonetic matching score (0-100)
            
        Returns:
            Overall score (0-100)
        """
        # Ensure scores are in valid range
        melody_score = np.clip(melody_score, 0, 100)
        phonetic_score = np.clip(phonetic_score, 0, 100)
        
        # Calculate weighted average
        total_score = (
            self.melody_weight * melody_score +
            self.phonetic_weight * phonetic_score
        )
        
        return float(total_score)
