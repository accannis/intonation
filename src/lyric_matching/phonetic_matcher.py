"""
Match phonetic features using pattern detection approach
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks

@dataclass
class PatternInfo:
    name: str
    features: np.ndarray
    duration: float

@dataclass
class DetectionResult:
    start_time: float
    end_time: float
    score: float

class PhoneticMatcher:
    def __init__(self, 
                 window_size: float = 1.0,
                 min_pattern_duration: float = 0.1,  # Minimum duration for a pattern in seconds
                 max_pattern_duration: float = 2.0,  # Maximum duration for a pattern in seconds
                 dtw_window: float = 0.2,  # DTW window constraint in seconds
                 peak_height: float = 20,  # Minimum peak height for pattern detection
                 peak_distance: float = 0.5,  # Minimum distance between peaks in seconds
                 score_threshold: float = 20):  # Minimum score for a detection
        """Initialize phonetic matcher
        
        Args:
            window_size: Size of the sliding window in seconds
            min_pattern_duration: Minimum duration for a pattern in seconds
            max_pattern_duration: Maximum duration for a pattern in seconds
            dtw_window: DTW window constraint in seconds (allows for timing variations)
            peak_height: Minimum peak height for pattern detection (0-100)
            peak_distance: Minimum distance between peaks in seconds
            score_threshold: Minimum score for a detection
        """
        self.window_size = window_size
        self.min_pattern_duration = 0.1  # Override to 100ms
        self.max_pattern_duration = max_pattern_duration
        self.dtw_window = dtw_window
        self.peak_height = peak_height
        self.peak_distance = peak_distance
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
            
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization
        
        Args:
            features: Input features (n_features x n_frames)
            
        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True) + 1e-8
        return (features - mean) / std

    def _dtw_distance(self, x: np.ndarray, y: np.ndarray, window: int) -> float:
        """Calculate DTW distance between two sequences with window constraint
        
        Args:
            x: First sequence (n_features x n_frames)
            y: Second sequence (n_features x n_frames)
            window: Maximum allowed deviation in frames
            
        Returns:
            DTW distance between sequences (0 = perfect match, 1 = no match)
        """
        try:
            logging.info(f"DTW input shapes - x: {x.shape}, y: {y.shape}")
            
            # Ensure both sequences have the same number of features
            min_features = min(x.shape[0], y.shape[0])
            x = x[:min_features]
            y = y[:min_features]
            
            n, m = x.shape[1], y.shape[1]
            
            # Initialize cost matrix
            D = np.full((n + 1, m + 1), np.inf)
            D[0, 0] = 0
            
            # Precompute frame-wise distances
            # Using Euclidean distance between normalized frames
            distances = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    diff = x[:, i] - y[:, j]
                    distances[i, j] = np.sqrt(np.sum(diff * diff))
            
            # Normalize distances to [0, 1]
            if np.max(distances) > 0:
                distances = distances / np.max(distances)
            
            logging.info(f"Frame distance stats - min: {np.min(distances):.3f}, "
                       f"max: {np.max(distances):.3f}, mean: {np.mean(distances):.3f}, "
                       f"std: {np.std(distances):.3f}")
            
            # Fill the cost matrix
            for i in range(1, n + 1):
                # Calculate window bounds
                if window:
                    start = max(1, i - window)
                    end = min(m + 1, i + window + 1)
                else:
                    start, end = 1, m + 1
                
                for j in range(start, end):
                    # Get frame-wise distance
                    cost = distances[i-1, j-1]
                    
                    # Calculate possible previous steps
                    candidates = []
                    if i > 1 and j > 1:
                        candidates.append(D[i-1, j-1])  # diagonal
                    if i > 1:
                        candidates.append(D[i-1, j])    # vertical
                    if j > 1:
                        candidates.append(D[i, j-1])    # horizontal
                    
                    # Update cost matrix
                    if candidates:
                        D[i, j] = cost + min(candidates)
                    else:
                        D[i, j] = cost
            
            # Extract the optimal warping path
            i, j = n, m
            path_length = 0
            path_cost = 0
            path_distances = []
            
            while i > 0 and j > 0:
                path_length += 1
                current_cost = distances[i-1, j-1]
                path_cost += current_cost
                path_distances.append(current_cost)
                
                # Find best previous step
                if i > 1 and j > 1:
                    step = np.argmin([D[i-1, j-1], D[i-1, j], D[i, j-1]])
                    if step == 0:
                        i, j = i-1, j-1
                    elif step == 1:
                        i -= 1
                    else:
                        j -= 1
                elif i > 1:
                    i -= 1
                else:
                    j -= 1
            
            if path_length == 0:
                logging.warning("Zero-length path in DTW")
                return 1.0
            
            # Calculate final score components
            avg_path_cost = path_cost / path_length
            length_ratio = min(n, m) / max(n, m)
            
            # Combine scores with weights
            # Lower cost = better match, so we subtract from 1
            path_score = 1.0 - avg_path_cost
            final_score = 0.7 * path_score + 0.3 * length_ratio
            
            logging.info(f"DTW path stats - length: {path_length}, "
                       f"avg_cost: {avg_path_cost:.3f}, "
                       f"path_score: {path_score:.3f}, "
                       f"length_ratio: {length_ratio:.3f}, "
                       f"final_score: {final_score:.3f}")
            
            logging.debug(f"Path distances - min: {min(path_distances):.3f}, "
                       f"max: {max(path_distances):.3f}, "
                       f"mean: {np.mean(path_distances):.3f}, "
                       f"std: {np.std(path_distances):.3f}")
            
            # For identical sequences, we should get a perfect score
            if np.array_equal(x, y):
                logging.info("Sequences are identical, returning perfect score")
                return 0.0
            
            return 1.0 - final_score
            
        except Exception as e:
            logging.error(f"DTW calculation failed: {e}")
            logging.exception("Full traceback:")
            return 1.0
            
    def detect_pattern(self, pattern: PatternInfo, input_mfcc: np.ndarray, frames_per_second: float) -> List[DetectionResult]:
        """Detect occurrences of a pattern in the input MFCC features using DTW.
        
        Args:
            pattern: Pattern information including MFCC features
            input_mfcc: Input MFCC features to search in
            frames_per_second: Frames per second of input audio
            
        Returns:
            List of DetectionResult objects containing match locations and scores
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
            pattern_features = self._normalize_features(pattern_features)
            
            # Calculate DTW distances for each window
            distances = []
            timestamps = []
            
            # Debug first few windows
            debug_windows = 3
            window_count = 0
            
            for start_frame in range(0, input_features.shape[1] - pattern_frames + 1, frames_per_window):
                end_frame = start_frame + pattern_frames
                window = input_features[:, start_frame:end_frame]
                
                # Skip if window is too small
                if window.shape[1] < pattern_frames // 2:
                    continue
                    
                # Normalize window features
                window = self._normalize_features(window)
                
                # Debug first few windows
                if window_count < debug_windows:
                    logging.debug(f"Window {window_count} - "
                               f"start: {start_frame}, end: {end_frame}, "
                               f"shape: {window.shape}, "
                               f"range: [{np.min(window):.3f}, {np.max(window):.3f}], "
                               f"mean: {np.mean(window):.3f}")
                    window_count += 1
                
                # Calculate DTW distance
                distance = self._dtw_distance(pattern_features, window, dtw_window_frames)
                distances.append(distance)
                timestamps.append(start_frame / frames_per_second)
                
                logging.debug(f"Window {start_frame}-{end_frame} DTW distance: {distance:.3f}")
                
            if distances:
                # Convert distances to scores (0-100)
                scores = np.array([(1.0 - d) * 100 for d in distances])
                
                logging.info(f"Pattern {pattern.name} - score stats: "
                           f"min={np.min(scores):.1f}, max={np.max(scores):.1f}, "
                           f"mean={np.mean(scores):.1f}, std={np.std(scores):.1f}")
                
                # If max score is high enough, just use that window
                max_score = np.max(scores)
                if max_score >= self.peak_height:
                    max_idx = np.argmax(scores)
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
            Tuple of (scores, times) where:
                scores: List of similarity scores (0-100) for each window
                times: List of time points corresponding to the scores
        """
        try:
            if not self.reference_patterns:
                logging.error("No reference patterns extracted. Call extract_patterns() first.")
                return [0.0], [0.0]
                
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
                
            # Sort by time
            times_scores = sorted(zip(all_times, all_scores))
            all_times, all_scores = zip(*times_scores)
            
            logging.info(f"Final match results - {len(all_scores)} total detections, "
                        f"score range: [{min(all_scores):.1f}, {max(all_scores):.1f}], "
                        f"mean: {np.mean(all_scores):.1f}")
            
            return list(all_scores), list(all_times)
            
        except Exception as e:
            logging.error(f"Pattern matching failed: {e}")
            logging.exception("Full traceback:")
            return [0.0], [0.0]
