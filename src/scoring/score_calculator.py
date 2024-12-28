from typing import Dict, List, Tuple
import numpy as np

class ScoreCalculator:
    def __init__(self):
        # Scoring weights
        self.melody_weight = 0.6
        self.lyric_weight = 0.4
        
        # Thresholds for feedback
        self.excellent_threshold = 0.85
        self.good_threshold = 0.7
        self.fair_threshold = 0.5
        
        # Store historical scores for trend analysis
        self.score_history: List[Dict[str, float]] = []
        self.history_size = 10
        
    def calculate_score(self, 
                       melody_similarity: float,
                       lyric_similarity: float) -> Dict[str, float]:
        """
        Calculate the total score and generate feedback
        
        Args:
            melody_similarity: Similarity score for melody (0-1)
            lyric_similarity: Similarity score for lyrics (0-1)
            
        Returns:
            Dictionary containing scores and feedback
        """
        # Calculate weighted scores
        melody_score = self._calculate_melody_score(melody_similarity)
        lyric_score = self._calculate_lyric_score(lyric_similarity)
        
        # Calculate total score (0-100)
        total_score = (
            self.melody_weight * melody_score +
            self.lyric_weight * lyric_score
        ) * 100
        
        # Store scores
        scores = {
            'total_score': total_score,
            'melody_score': melody_score,
            'lyric_score': lyric_score
        }
        
        self._update_history(scores)
        
        return scores
        
    def _calculate_melody_score(self, similarity: float) -> float:
        """
        Calculate melody score with emphasis on pitch accuracy
        
        Args:
            similarity: Raw similarity score (0-1)
            
        Returns:
            Processed melody score (0-1)
        """
        # Apply non-linear scaling to emphasize accuracy
        return np.power(similarity, 1.2)
        
    def _calculate_lyric_score(self, similarity: float) -> float:
        """
        Calculate lyrics score with tolerance for minor differences
        
        Args:
            similarity: Raw similarity score (0-1)
            
        Returns:
            Processed lyrics score (0-1)
        """
        # Apply more forgiving curve for lyrics
        return np.power(similarity, 0.8)
        
    def _update_history(self, scores: Dict[str, float]):
        """Update score history for trend analysis"""
        self.score_history.append(scores)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)
            
    def generate_feedback(self) -> Tuple[str, List[str]]:
        """
        Generate detailed feedback based on recent performance
        
        Returns:
            Tuple of (overall_feedback, list of specific_tips)
        """
        if not self.score_history:
            return "Start singing to get feedback!", []
            
        recent_scores = self.score_history[-1]
        total_score = recent_scores['total_score']
        melody_score = recent_scores['melody_score']
        lyric_score = recent_scores['lyric_score']
        
        feedback = []
        tips = []
        
        # Overall performance feedback
        if total_score >= self.excellent_threshold * 100:
            feedback.append("Excellent performance! ")
        elif total_score >= self.good_threshold * 100:
            feedback.append("Good job! Keep it up! ")
        elif total_score >= self.fair_threshold * 100:
            feedback.append("Fair performance. Room for improvement. ")
        else:
            feedback.append("Keep practicing! You'll get better! ")
            
        # Specific feedback on melody
        if melody_score < self.good_threshold:
            if melody_score < self.fair_threshold:
                tips.append("Try to match the pitch more closely")
            else:
                tips.append("Focus on maintaining consistent pitch")
                
        # Specific feedback on lyrics
        if lyric_score < self.good_threshold:
            if lyric_score < self.fair_threshold:
                tips.append("Work on pronunciation clarity")
            else:
                tips.append("Pay attention to word timing")
                
        # Trend analysis
        if len(self.score_history) >= 3:
            trend = self._analyze_trend()
            if trend > 0.05:
                feedback.append("You're improving! ")
            elif trend < -0.05:
                tips.append("Take a breath and try to focus")
                
        return " ".join(feedback), tips
        
    def _analyze_trend(self) -> float:
        """Calculate the trend in recent scores"""
        if len(self.score_history) < 2:
            return 0.0
            
        recent_scores = [s['total_score'] for s in self.score_history]
        x = np.arange(len(recent_scores))
        slope, _ = np.polyfit(x, recent_scores, 1)
        
        return slope / 100  # Normalize trend
