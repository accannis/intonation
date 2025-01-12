"""Melody matching module"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

class MelodyMatcher:
    """Match melodies using pitch contours"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize melody matcher with config"""
        self.min_note_duration = config.get("min_note_duration", 0.1)  # 100ms
        self.min_pitch_confidence = config.get("min_pitch_confidence", 0.8)  # Reduced from 0.95
        self.pitch_tolerance = config.get("pitch_tolerance", 1.0)  # Reduced from 2.0 semitones
        self.timing_tolerance = config.get("timing_tolerance", 10.0)  # Seconds
        self.min_overlap = config.get("min_overlap", 0.5)
        self.min_voiced_frames = 100  # Reduced from 200 - at least 5 seconds of voiced frames
        self.min_voiced_ratio = 0.2  # Reduced from 0.3 - at least 20% of frames should be voiced
        self.detections = []
        
    def normalize_pitch(self, pitch: float) -> float:
        """Normalize pitch to be in a single octave (between 0 and 12 semitones)"""
        if pitch <= 0:
            return 0
        return np.log2(pitch) * 12 % 12

    def dynamic_time_warp(self, ref_pitch: np.ndarray, input_pitch: np.ndarray) -> float:
        """Apply dynamic time warping to align and score pitch sequences
        
        Args:
            ref_pitch: Reference pitch sequence
            input_pitch: Input pitch sequence
            
        Returns:
            Score between 0 and 100 indicating similarity
        """
        # Get sequence lengths
        N, M = len(ref_pitch), len(input_pitch)
        
        # Initialize cost matrix
        cost = np.zeros((N + 1, M + 1))
        cost[0, 1:] = np.inf
        cost[1:, 0] = np.inf
        
        # Fill cost matrix
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # Calculate pitch difference cost
                diff = abs(ref_pitch[i-1] - input_pitch[j-1])
                diff_cost = min(100, diff * 4)  # Cap at 100, 4x penalty
                
                # Combine costs with previous minimum
                cost[i, j] = diff_cost + min(
                    cost[i-1, j],     # Deletion
                    cost[i, j-1],     # Insertion
                    cost[i-1, j-1]    # Match
                )
                
        # Get final cost and normalize
        final_cost = cost[N, M]
        path_length = N + M  # Conservative estimate of path length
        avg_cost = final_cost / path_length
        
        # Convert cost to score (0-100)
        score = max(0, 100 - avg_cost)
        
        return score

    def check_pitch_contour(self, ref_pitch: np.ndarray, input_pitch: np.ndarray) -> float:
        """Check if pitch contours are similar by comparing pitch changes
        
        Args:
            ref_pitch: Reference pitch sequence
            input_pitch: Input pitch sequence
            
        Returns:
            Score between 0 and 1 indicating contour similarity
        """
        # Get pitch changes (derivatives) 
        ref_changes = np.diff(ref_pitch)
        input_changes = np.diff(input_pitch)
        
        # Resample longer sequence to match shorter one
        if len(ref_changes) > len(input_changes):
            indices = np.linspace(0, len(ref_changes)-1, len(input_changes)).astype(int)
            ref_changes = ref_changes[indices]
        else:
            indices = np.linspace(0, len(input_changes)-1, len(ref_changes)).astype(int)
            input_changes = input_changes[indices]
            
        # Normalize changes to [-1, 1] range
        ref_max = np.max(np.abs(ref_changes))
        input_max = np.max(np.abs(input_changes))
        
        if ref_max > 0:
            ref_changes = ref_changes / ref_max
        if input_max > 0:
            input_changes = input_changes / input_max
            
        # Calculate correlation between changes
        correlation = np.corrcoef(ref_changes, input_changes)[0,1]
        
        # Handle NaN correlation (happens if one sequence is constant)
        if np.isnan(correlation):
            return 0.5
            
        # Convert correlation to [0,1] range
        score = (correlation + 1) / 2
        
        return score
        
    def check_pitch_distribution(self, ref_pitch: np.ndarray, input_pitch: np.ndarray) -> float:
        """Check if pitch distributions are similar
        
        Args:
            ref_pitch: Reference pitch sequence
            input_pitch: Input pitch sequence
            
        Returns:
            Score between 0 and 1 indicating distribution similarity
        """
        # Create histograms
        bins = 50
        ref_hist, _ = np.histogram(ref_pitch, bins=bins, density=True)
        input_hist, _ = np.histogram(input_pitch, bins=bins, density=True)
        
        # Calculate histogram intersection
        intersection = np.minimum(ref_hist, input_hist).sum()
        total = np.maximum(ref_hist, input_hist).sum()
        
        if total == 0:
            return 0.5
            
        score = intersection / total
        return score

    def check_pitch_range(self, ref_pitch: np.ndarray, input_pitch: np.ndarray) -> float:
        """Check if pitch ranges are similar
        
        Args:
            ref_pitch: Reference pitch sequence
            input_pitch: Input pitch sequence
            
        Returns:
            Score between 0 and 1 indicating similarity
        """
        # Get pitch ranges
        ref_range = np.ptp(ref_pitch)  # Peak-to-peak
        input_range = np.ptp(input_pitch)
        
        # Compare ranges
        if ref_range == 0 or input_range == 0:
            return 0.0
            
        range_ratio = min(ref_range, input_range) / max(ref_range, input_range)
        
        # Penalize very different ranges
        if range_ratio < 0.5:  # More than 2x difference
            range_ratio *= 0.5
            
        return range_ratio

    def check_pitch_correlation(self, ref_pitch: np.ndarray, input_pitch: np.ndarray) -> float:
        """Check correlation between pitch sequences
        
        Args:
            ref_pitch: Reference pitch sequence
            input_pitch: Input pitch sequence
            
        Returns:
            Score between 0 and 1 indicating similarity
        """
        # Resample to same length
        min_len = min(len(ref_pitch), len(input_pitch))
        ref_resampled = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(ref_pitch)),
            ref_pitch
        )
        input_resampled = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(input_pitch)),
            input_pitch
        )
        
        # Calculate correlation
        corr = np.corrcoef(ref_resampled, input_resampled)[0, 1]
        
        # Convert to positive score
        score = (corr + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Penalize low correlation more heavily
        if score < 0.5:
            score *= 0.5
            
        return score

    def check_pitch_jumps(self, ref_pitch: np.ndarray, input_pitch: np.ndarray) -> float:
        """Check if number and size of pitch jumps are similar
        
        Args:
            ref_pitch: Reference pitch sequence
            input_pitch: Input pitch sequence
            
        Returns:
            Score between 0 and 1 indicating similarity
        """
        # Calculate pitch jumps
        ref_jumps = np.diff(ref_pitch)
        input_jumps = np.diff(input_pitch)
        
        # Count significant jumps (more than 2 semitones)
        ref_sig_jumps = np.sum(np.abs(ref_jumps) > 2)
        input_sig_jumps = np.sum(np.abs(input_jumps) > 2)
        
        # Compare number of jumps
        if ref_sig_jumps == 0 and input_sig_jumps == 0:
            return 1.0
        elif ref_sig_jumps == 0 or input_sig_jumps == 0:
            return 0.0
            
        jump_ratio = min(ref_sig_jumps, input_sig_jumps) / max(ref_sig_jumps, input_sig_jumps)
        
        # Penalize very different numbers of jumps
        if jump_ratio < 0.5:  # More than 2x difference
            jump_ratio *= 0.5
            
        return jump_ratio

    def check_length_ratio(self, ref_pitch: np.ndarray, input_pitch: np.ndarray) -> float:
        """Check ratio of sequence lengths
        
        Args:
            ref_pitch: Reference pitch sequence
            input_pitch: Input pitch sequence
            
        Returns:
            Score between 0 and 1 indicating length similarity
        """
        # Calculate ratio of shorter to longer length
        ratio = min(len(ref_pitch), len(input_pitch)) / max(len(ref_pitch), len(input_pitch))
        
        # Apply progressive penalty
        if ratio > 0.7:  # Allow 30% difference with no penalty
            return 1.0
        elif ratio > 0.4:  # Progressive penalty between 40-70%
            return 0.5 + (ratio - 0.4) * 1.67  # Scale to 0.5-1.0
        else:  # Severe penalty below 40%
            return ratio * 1.25  # Scale to 0-0.5
            
        return ratio

    def analyze_windows(self, ref_pitch: np.ndarray, input_pitch: np.ndarray, window_size: int = 50) -> float:
        """Analyze pitch sequences in windows to catch local similarities
        
        Args:
            ref_pitch: Reference pitch sequence
            input_pitch: Input pitch sequence
            window_size: Size of analysis window
            
        Returns:
            Score between 0 and 1 indicating local similarity
        """
        # Get sequence lengths
        N, M = len(ref_pitch), len(input_pitch)
        
        # Initialize scores
        window_scores = []
        
        # Analyze reference windows
        for i in range(0, N - window_size + 1, window_size // 2):  # 50% overlap
            ref_window = ref_pitch[i:i + window_size]
            
            # Find best matching window in input
            best_score = 0
            for j in range(0, M - window_size + 1, window_size // 4):  # 75% overlap for input
                input_window = input_pitch[j:j + window_size]
                
                # Calculate correlation
                try:
                    corr = np.corrcoef(ref_window, input_window)[0, 1]
                    if np.isnan(corr):
                        corr = 0
                except:
                    corr = 0
                    
                # Calculate pitch difference
                pitch_diff = np.mean(np.abs(ref_window - input_window))
                pitch_sim = max(0, 1 - pitch_diff / 6)  # Normalize by half octave
                
                # Combine metrics
                score = 0.7 * corr + 0.3 * pitch_sim
                best_score = max(best_score, score)
            
            window_scores.append(best_score)
            
        # Get overall score
        if window_scores:
            # Weight recent windows more heavily
            weights = np.linspace(0.5, 1.0, len(window_scores))
            weighted_scores = np.array(window_scores) * weights
            return float(np.mean(weighted_scores))
        else:
            return 0.0

    def analyze_chunk(self, ref_chunk: np.ndarray, input_chunk: np.ndarray) -> float:
        """Analyze similarity between two chunks of melody
        
        Args:
            ref_chunk: Reference chunk pitch sequence
            input_chunk: Input chunk pitch sequence
            
        Returns:
            Score between 0 and 100 indicating similarity
        """
        # Normalize chunks to zero mean and unit variance
        ref_mean = np.mean(ref_chunk)
        ref_std = np.std(ref_chunk)
        input_mean = np.mean(input_chunk)
        input_std = np.std(input_chunk)
        
        if ref_std > 0:
            ref_chunk = (ref_chunk - ref_mean) / ref_std
        if input_std > 0:
            input_chunk = (input_chunk - input_mean) / input_std
            
        # Calculate DTW with strict penalty
        N, M = len(ref_chunk), len(input_chunk)
        cost = np.zeros((N + 1, M + 1))
        cost[0, 1:] = np.inf
        cost[1:, 0] = np.inf
        
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                diff = abs(ref_chunk[i-1] - input_chunk[j-1])
                diff_cost = min(100, diff * 8)  # Strict penalty
                
                cost[i, j] = diff_cost + min(
                    cost[i-1, j],
                    cost[i, j-1],
                    cost[i-1, j-1]
                )
                
        # Get final cost and normalize
        final_cost = cost[N, M]
        path_length = N + M
        avg_cost = final_cost / path_length
        dtw_score = max(0, 100 - avg_cost)
        
        # Calculate correlation
        try:
            corr = np.corrcoef(ref_chunk, input_chunk)[0, 1]
            if np.isnan(corr):
                corr = 0
        except:
            corr = 0
            
        # Calculate contour similarity
        ref_contour = np.diff(ref_chunk)
        input_contour = np.diff(input_chunk)
        ref_contour = np.sign(ref_contour)
        input_contour = np.sign(input_contour)
        
        if len(ref_contour) > 0 and len(input_contour) > 0:
            min_len = min(len(ref_contour), len(input_contour))
            contour_match = np.mean(ref_contour[:min_len] == input_contour[:min_len])
        else:
            contour_match = 0
            
        # Combine scores
        chunk_score = (
            0.6 * dtw_score +  # DTW is most important
            0.2 * max(0, corr * 100) +  # Correlation
            0.2 * contour_match * 100  # Contour
        )
        
        return chunk_score

    def match_melody(self, reference: np.ndarray, input_melody: np.ndarray) -> float:
        """Match input melody against reference melody using chunked analysis
        
        Args:
            reference: Reference melody features (N x 2 array of pitch and confidence)
            input_melody: Input melody features (M x 2 array of pitch and confidence)
            
        Returns:
            Score between 0 and 100 indicating how well the melodies match
        """
        # Convert to 2D arrays if needed
        if len(reference.shape) == 1:
            reference = reference.reshape(-1, 1)
        if len(input_melody.shape) == 1:
            input_melody = input_melody.reshape(-1, 1)
            
        # Extract pitch and confidence
        ref_pitch = reference[:, 0]
        ref_conf = np.ones_like(ref_pitch) if reference.shape[1] == 1 else reference[:, 1]
        input_pitch = input_melody[:, 0]
        input_conf = np.ones_like(input_pitch) if input_melody.shape[1] == 1 else input_melody[:, 1]
        
        # Find voiced segments with high confidence
        ref_voiced = (ref_conf >= self.min_pitch_confidence) & (ref_pitch > 0)
        input_voiced = (input_conf >= self.min_pitch_confidence) & (input_pitch > 0)
        
        logging.info(f"Found {np.sum(ref_voiced)} voiced frames in reference and {np.sum(input_voiced)} in input")
        
        # Check minimum voiced frames requirement
        if np.sum(input_voiced) < self.min_voiced_frames:
            logging.warning(f"Not enough voiced frames in input: {np.sum(input_voiced)} < {self.min_voiced_frames}")
            return 0.0
            
        # Check minimum voiced ratio requirement
        voiced_ratio = np.sum(input_voiced) / len(input_voiced)
        if voiced_ratio < self.min_voiced_ratio:
            logging.warning(f"Voiced ratio too low: {voiced_ratio:.3f} < {self.min_voiced_ratio}")
            return 0.0
            
        if not np.any(ref_voiced) or not np.any(input_voiced):
            logging.warning("No voiced frames found in one or both signals")
            return 0.0
            
        # Get voiced pitches
        ref_pitch = ref_pitch[ref_voiced]
        input_pitch = input_pitch[input_voiced]
        
        # Define chunk parameters
        chunk_size = 50  # frames
        chunk_overlap = 25  # frames
        
        # Initialize chunk scores
        chunk_scores = []
        chunk_weights = []
        
        # Analyze reference chunks
        for i in range(0, len(ref_pitch) - chunk_size + 1, chunk_overlap):
            ref_chunk = ref_pitch[i:i + chunk_size]
            
            # Find best matching chunk in input
            best_score = 0
            for j in range(0, len(input_pitch) - chunk_size + 1, chunk_overlap // 2):
                input_chunk = input_pitch[j:j + chunk_size]
                score = self.analyze_chunk(ref_chunk, input_chunk)
                best_score = max(best_score, score)
                
            # Weight by chunk position (later chunks weighted more)
            weight = 0.5 + 0.5 * (i / len(ref_pitch))
            
            chunk_scores.append(best_score)
            chunk_weights.append(weight)
            
        if not chunk_scores:
            return 0.0
            
        # Calculate weighted average score
        chunk_scores = np.array(chunk_scores)
        chunk_weights = np.array(chunk_weights)
        
        # Get overall statistics
        mean_score = np.average(chunk_scores, weights=chunk_weights)
        top_score = np.max(chunk_scores)
        
        # Calculate final score with emphasis on best matches
        score = 0.7 * top_score + 0.3 * mean_score
        
        # Boost high scores
        if score > 60:  # Start boosting earlier for good matches
            boost = min(2.0, 1.0 + (score - 60) / 20)  # Progressive boost
            score = 60 + (score - 60) * boost
            
        score = min(100, score)  # Cap at 100
        
        logging.info(f"Chunk analysis - Mean: {mean_score:.2f}, Top: {top_score:.2f}, Final: {score:.2f}")
        
        # Store detection if score is good
        if score > 50:
            self.detections = [{
                "reference_time": 0.0,
                "input_time": 0.0,
                "score": score
            }]
        else:
            self.detections = []
            
        return score

    def get_detections(self) -> List[Dict[str, float]]:
        """Get list of detected note matches"""
        return self.detections
