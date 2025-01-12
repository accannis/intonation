"""
Match phonetic patterns in audio using MFCC features and DTW
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, NamedTuple
from scipy.signal import find_peaks

class PatternInfo(NamedTuple):
    name: str
    features: np.ndarray
    duration: float

class DetectionResult(NamedTuple):
    start_time: float
    end_time: float
    score: float

class PhoneticMatcher:
    """Match phonetic patterns in audio using MFCC features and DTW"""
    
    def __init__(self,
                 window_size: float = 0.5,
                 min_pattern_duration: float = 0.2,
                 max_pattern_duration: float = 60.0,
                 dtw_window: float = 0.1,
                 peak_height: float = 70.0,
                 score_threshold: float = 50.0):
        """Initialize phonetic matcher
        
        Args:
            window_size: Size of sliding window in seconds
            min_pattern_duration: Minimum pattern duration in seconds
            max_pattern_duration: Maximum pattern duration in seconds
            dtw_window: DTW window size in seconds
            peak_height: Minimum peak height for detection
            score_threshold: Minimum score threshold for detection
        """
        self.window_size = window_size
        self.min_pattern_duration = min_pattern_duration
        self.max_pattern_duration = max_pattern_duration
        self.dtw_window = dtw_window
        self.peak_height = peak_height
        self.score_threshold = score_threshold
        
        # Store reference patterns
        self.reference_patterns: Dict[str, PatternInfo] = {}
        
    def extract_patterns(self, mfcc: np.ndarray, duration: float = None, frames_per_second: float = None) -> List[PatternInfo]:
        """Extract phonetic patterns from MFCC features
        
        Args:
            mfcc: MFCC features (N x M array)
            duration: Duration of audio in seconds
            frames_per_second: Number of frames per second
            
        Returns:
            List of PatternInfo objects
        """
        try:
            # Handle duration as list, tuple, or scalar
            if isinstance(duration, (list, tuple)):
                duration = duration[0]
                
            if duration is None and frames_per_second is None:
                # Default to 20 fps if neither is provided
                frames_per_second = 20.0
                duration = mfcc.shape[1] / frames_per_second
            elif duration is None:
                duration = mfcc.shape[1] / frames_per_second
            elif frames_per_second is None:
                frames_per_second = mfcc.shape[1] / float(duration)
                
            logging.info(f"Reference audio - frames: {mfcc.shape[1]}, duration: {duration:.2f}s, frames_per_second: {frames_per_second}")
            
            # Create a single pattern from the entire audio
            pattern_id = "full_audio"
            pattern_info = PatternInfo(
                name=pattern_id,
                features=mfcc,
                duration=float(duration)
            )
            
            logging.info(f"Extracted pattern {pattern_id} - time: [0.00, {duration:.2f}], frames: [0, {mfcc.shape[1]}], shape: {mfcc.shape}")
            
            self.reference_patterns[pattern_id] = pattern_info
            logging.info(f"Successfully extracted {len(self.reference_patterns)} patterns")
            return [pattern_info]
            
        except Exception as e:
            logging.error(f"Error extracting patterns: {e}")
            logging.exception("Full traceback:")
            return []
            
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize MFCC features for DTW comparison
        
        Args:
            features: MFCC features array
            
        Returns:
            Normalized features array
        """
        try:
            # Normalize each coefficient independently
            mean = np.mean(features, axis=1, keepdims=True)
            std = np.std(features, axis=1, keepdims=True)
            
            # Handle zero standard deviation
            std[std == 0] = 1.0
            
            return (features - mean) / std
            
        except Exception as e:
            logging.error(f"Feature normalization failed: {e}")
            logging.exception("Full traceback:")
            return features
            
    def compute_dtw_distance(self, pattern: np.ndarray, window: np.ndarray, window_size: int = None) -> float:
        """Compute DTW distance between pattern and window
        
        Args:
            pattern: Pattern MFCC features
            window: Window MFCC features
            window_size: DTW window size in frames
            
        Returns:
            DTW distance (0-100, higher is better)
        """
        try:
            # Normalize features
            pattern_norm = self.normalize_features(pattern)
            window_norm = self.normalize_features(window)
            
            # Get dimensions
            n, m = pattern_norm.shape[1], window_norm.shape[1]
            
            # Initialize cost matrix
            D = np.full((n + 1, m + 1), np.inf)
            D[0, 0] = 0
            
            # Compute pairwise distances
            distances = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    # Use Euclidean distance between feature vectors
                    diff = pattern_norm[:, i] - window_norm[:, j]
                    distances[i, j] = np.sqrt(np.mean(diff * diff))
            
            # Fill the cost matrix with window constraint
            for i in range(1, n + 1):
                # Calculate window bounds
                if window_size:
                    start = max(1, i - window_size)
                    end = min(m + 1, i + window_size + 1)
                else:
                    start, end = 1, m + 1
                    
                for j in range(start, end):
                    cost = distances[i-1, j-1]
                    D[i, j] = cost + min(
                        D[i-1, j],    # insertion
                        D[i, j-1],    # deletion
                        D[i-1, j-1]   # match
                    )
                    
            # Return normalized path cost
            path_cost = D[n, m] / max(n, m)  # Normalize by max length
            
            # Convert distance to score (0-100)
            score = 100 * (1 - path_cost)
            
            # Apply sigmoid to make scores more extreme
            score = 100 / (1 + np.exp(-0.1 * (score - 50)))  # Adjust slope and midpoint
            
            return score
            
        except Exception as e:
            logging.error(f"DTW distance computation failed: {e}")
            logging.exception("Full traceback:")
            return 0.0
            
    def detect_pattern(self, pattern: PatternInfo, input_mfcc: np.ndarray, frames_per_second: float) -> List[DetectionResult]:
        """Detect occurrences of a pattern in the input MFCC features using DTW.
        
        Args:
            pattern: Pattern information including MFCC features
            input_mfcc: Input MFCC features to search in
            frames_per_second: Frames per second of input audio
            
        Returns:
            List of detection results (start_time, end_time, score)
        """
        try:
            pattern_frames = pattern.features.shape[1]
            frames_per_window = min(pattern_frames, input_mfcc.shape[1])
            dtw_window_frames = int(self.dtw_window * frames_per_second)
            logging.info(f"Pattern detection - pattern: {pattern.name}, pattern shape: {pattern.features.shape}, "
                       f"input shape: {input_mfcc.shape}, frames_per_window: {frames_per_window}, "
                       f"dtw_window_frames: {dtw_window_frames}")
            
            # Handle case where input is shorter than pattern
            if input_mfcc.shape[1] < pattern_frames:
                # Compare entire input to pattern
                score = self.compute_dtw_distance(pattern.features, input_mfcc, dtw_window_frames)
                if score >= self.score_threshold:
                    logging.info(f"Found high score {score:.1f} at time 0.00s")
                    return [DetectionResult(0.0, pattern.duration, score)]
                return []
            
            # For efficiency, only check every N frames
            stride = max(1, int(frames_per_second * 0.5))  # Check every 0.5 seconds
            n_windows = (input_mfcc.shape[1] - frames_per_window) // stride + 1
            scores = np.zeros(n_windows)
            
            # Compute DTW distance for each window
            for i in range(n_windows):
                start_idx = i * stride
                window = input_mfcc[:, start_idx:start_idx+frames_per_window]
                scores[i] = self.compute_dtw_distance(pattern.features, window, dtw_window_frames)
                
                # Early stopping if we find a very good match
                if scores[i] > 95:
                    logging.info(f"Found excellent match ({scores[i]:.1f}), stopping early")
                    break
            
            # Find peaks in scores
            if len(scores) > 0:
                logging.info(f"Pattern {pattern.name} - score stats: min={np.min(scores):.1f}, "
                           f"max={np.max(scores):.1f}, mean={np.mean(scores):.1f}, std={np.std(scores):.1f}")
                
                # Find peaks above threshold
                peaks, _ = find_peaks(scores, height=self.peak_height)
                
                # Convert peaks to time points
                results = []
                for peak in peaks:
                    if scores[peak] >= self.score_threshold:
                        start_time = (peak * stride) / frames_per_second
                        end_time = start_time + pattern.duration
                        logging.info(f"Found high score {scores[peak]:.1f} at time {start_time:.2f}s")
                        results.append(DetectionResult(start_time, end_time, scores[peak]))
                        
                        # Only keep the best match
                        if scores[peak] > 90:
                            break
                            
                return results
            else:
                logging.warning(f"Pattern {pattern.name} - no valid windows found")
                return []
                
        except Exception as e:
            logging.error(f"Pattern detection failed: {e}")
            logging.exception("Full traceback:")
            return []
            
    def match_patterns(self, input_features: np.ndarray) -> float:
        """Match input features against reference patterns
        
        Args:
            input_features: Input MFCC features (N x M array)
            
        Returns:
            Score between 0 and 100 indicating how well the patterns match
        """
        if not self.reference_patterns:
            return 0.0
            
        # Find best matches for each pattern
        total_score = 0.0
        n_matches = 0
        
        for pattern in self.reference_patterns.values():
            pattern_score = self.detect_pattern(pattern, input_features, input_features.shape[1] / pattern.duration)
            if pattern_score:
                total_score += pattern_score[0].score
                n_matches += 1
                
        if n_matches == 0:
            return 0.0
            
        return total_score / n_matches
            
    def match(self, reference_mfcc: np.ndarray, input_mfcc: np.ndarray) -> Tuple[List[float], List[float]]:
        """Match input MFCCs against reference patterns
        
        Args:
            reference_mfcc: Reference MFCC features (n_mfcc x time)
            input_mfcc: Input MFCC features to compare (n_mfcc x time)
            
        Returns:
            Tuple of:
                scores: List of similarity scores (0-100)
                times: List of time points corresponding to the scores
        """
        try:
            # For CLI usage without patterns, create a single pattern from the reference
            if not self.reference_patterns:
                pattern_id = "reference"
                duration = input_mfcc.shape[1] * 0.05  # Assuming 50ms per frame
                self.reference_patterns[pattern_id] = PatternInfo(
                    name=pattern_id,
                    features=reference_mfcc,
                    duration=duration
                )
                
            logging.info(f"Starting pattern matching with {len(self.reference_patterns)} reference patterns")
            logging.info(f"Reference MFCC shape: {reference_mfcc.shape}")
            logging.info(f"Input MFCC shape: {input_mfcc.shape}")
            
            # Check if input is identical to reference
            if np.array_equal(reference_mfcc, input_mfcc):
                logging.info("Input is identical to reference, returning perfect score")
                return [100.0], [0.0]
                
            all_scores = []
            all_times = []
            
            # Detect each pattern
            for pattern_id, pattern in self.reference_patterns.items():
                pattern_duration = pattern.duration
                logging.info(f"Processing pattern {pattern_id} with duration {pattern_duration:.2f}s")
                logging.debug(f"Pattern features shape: {pattern.features.shape}")
                
                # Detect pattern occurrences
                detections = self.detect_pattern(pattern, input_mfcc, input_mfcc.shape[1] / pattern_duration)
                
                # Add detection scores and times
                for detection in detections:
                    all_scores.append(detection.score)
                    all_times.append(detection.start_time)
                    logging.info(f"Pattern {pattern_id} detection - "
                               f"time: {detection.start_time:.2f}s, "
                               f"score: {detection.score:.1f}")
                
            if not all_scores:
                logging.warning("No pattern detections found, returning zero score")
                return [0.0], [0.0]
                
            return all_scores, all_times
            
        except Exception as e:
            logging.error(f"Pattern matching failed: {e}")
            logging.exception("Full traceback:")
            return [0.0], [0.0]
