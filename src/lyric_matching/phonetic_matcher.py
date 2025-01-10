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
                 max_pattern_duration: float = 2.0,
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
        
    def extract_patterns(self, reference_mfcc: np.ndarray, pattern_times: List[Tuple[str, float, float]]):
        """Extract and store reference patterns
        
        Args:
            reference_mfcc: Reference MFCC features (n_mfcc x time)
            pattern_times: List of (pattern_id, start_time, end_time) tuples
        """
        try:
            logging.info(f"Extracting patterns from reference MFCC shape: {reference_mfcc.shape}")
            
            # Calculate frames per second
            total_frames = reference_mfcc.shape[1]
            total_duration = pattern_times[-1][2]  # Use last end time as total duration
            frames_per_second = total_frames / total_duration
            
            logging.info(f"Reference audio - frames: {total_frames}, "
                        f"duration: {total_duration:.2f}s, "
                        f"frames_per_second: {frames_per_second:.1f}")
            
            # Extract each pattern
            for pattern_id, start_time, end_time in pattern_times:
                try:
                    # Convert times to frame indices
                    start_frame = int(start_time * frames_per_second)
                    end_frame = int(end_time * frames_per_second)
                    duration = end_time - start_time
                    
                    # Skip if pattern is too short or too long
                    if duration < self.min_pattern_duration or duration > self.max_pattern_duration:
                        logging.warning(f"Pattern {pattern_id} duration {duration:.2f}s outside range "
                                     f"[{self.min_pattern_duration}, {self.max_pattern_duration}], skipping")
                        continue
                    
                    # Extract pattern features
                    pattern_features = reference_mfcc[:, start_frame:end_frame]
                    
                    logging.info(f"Extracted pattern {pattern_id} - "
                               f"time: [{start_time:.2f}, {end_time:.2f}], "
                               f"frames: [{start_frame}, {end_frame}], "
                               f"shape: {pattern_features.shape}")
                    
                    # Store pattern
                    self.reference_patterns[pattern_id] = PatternInfo(
                        name=pattern_id,
                        features=pattern_features,
                        duration=duration
                    )
                    
                except Exception as e:
                    logging.error(f"Failed to extract pattern {pattern_id}: {e}")
                    logging.exception("Full traceback:")
                    continue
            
            logging.info(f"Successfully extracted {len(self.reference_patterns)} patterns")
            
        except Exception as e:
            logging.error(f"Pattern extraction failed: {e}")
            logging.exception("Full traceback:")
            
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
            
    def compute_dtw_distance(self, pattern: np.ndarray, window: np.ndarray, window_size: int) -> float:
        """Compute DTW distance between pattern and window
        
        Args:
            pattern: Pattern MFCC features
            window: Window MFCC features
            window_size: DTW window size in frames
            
        Returns:
            DTW distance (0-100, higher is better)
        """
        try:
            # Compute frame-wise Euclidean distances
            distances = []
            for i in range(pattern.shape[1]):
                frame_dists = []
                for j in range(window.shape[1]):
                    dist = np.sqrt(np.sum((pattern[:, i] - window[:, j]) ** 2))
                    frame_dists.append(dist)
                distances.append(frame_dists)
                
            distances = np.array(distances)
            
            # Normalize distances to [0, 1]
            if np.max(distances) > 0:
                distances = distances / np.max(distances)
                
            # Initialize cost matrix
            n, m = distances.shape
            D = np.full((n + 1, m + 1), np.inf)
            D[0, 0] = 0
            
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
            path_cost = D[n, m] / n
            
            # Convert distance to score (0-100)
            score = 100 * (1 - path_cost)
            
            # Apply sigmoid to make scores more extreme
            score = 100 / (1 + np.exp(-0.2 * (score - 30)))  # Adjust slope and midpoint
            
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
            # Calculate frames per window and pattern length in frames
            frames_per_window = int(self.window_size * frames_per_second)
            pattern_frames = pattern.features.shape[1]
            
            # Calculate DTW window size in frames
            dtw_window_frames = int(self.dtw_window * frames_per_second)
            
            logging.info(f"Pattern detection - pattern: {pattern.name}, "
                        f"pattern shape: {pattern.features.shape}, "
                        f"input shape: {input_mfcc.shape}, "
                        f"frames_per_window: {frames_per_window}, "
                        f"dtw_window_frames: {dtw_window_frames}")
            
            # Ensure both sequences have the same number of features
            min_features = min(pattern.features.shape[0], input_mfcc.shape[0])
            pattern_features = pattern.features[:min_features]
            input_features = input_mfcc[:min_features]
            
            # Normalize pattern features
            pattern_norm = self.normalize_features(pattern_features)
            
            # Initialize arrays for DTW distances and timestamps
            distances = []
            timestamps = []
            
            # Slide window over input
            for start_frame in range(0, input_features.shape[1] - pattern_frames + 1, frames_per_window):
                try:
                    # Extract window
                    end_frame = start_frame + pattern_frames
                    window = input_features[:, start_frame:end_frame]
                    
                    # Skip if window is too short
                    if window.shape[1] < pattern_frames:
                        continue
                        
                    # Normalize window features
                    window_norm = self.normalize_features(window)
                    
                    # Compute DTW distance
                    distance = self.compute_dtw_distance(
                        pattern_norm,
                        window_norm,
                        dtw_window_frames
                    )
                    
                    # Store distance and timestamp
                    distances.append(distance)
                    timestamps.append(start_frame / frames_per_second)
                    
                except Exception as e:
                    logging.error(f"Window processing failed at {start_frame}: {e}")
                    logging.exception("Full traceback:")
                    continue
                    
            if distances:
                # Convert distances to scores (0-100)
                scores = np.array(distances)
                
                logging.info(f"Pattern {pattern.name} - score stats: "
                           f"min={np.min(scores):.1f}, max={np.max(scores):.1f}, "
                           f"mean={np.mean(scores):.1f}, std={np.std(scores):.1f}")
                
                # Find highest scoring detection
                max_idx = np.argmax(scores)
                max_score = scores[max_idx]
                
                if max_score >= self.peak_height:
                    logging.info(f"Found high score {max_score:.1f} at time {timestamps[max_idx]:.2f}s")
                    return [DetectionResult(
                        start_time=timestamps[max_idx],
                        end_time=timestamps[max_idx] + pattern.duration,
                        score=max_score
                    )]
                else:
                    logging.warning(f"Pattern {pattern.name} - no scores above threshold {self.peak_height}")
                    return []
            else:
                logging.warning(f"Pattern {pattern.name} - no valid windows found")
                return []
                
        except Exception as e:
            logging.error(f"Pattern detection failed: {e}")
            logging.exception("Full traceback:")
            return []
            
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
