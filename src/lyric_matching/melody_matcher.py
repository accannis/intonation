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
        self.min_segment_score = 0.4  # Minimum score for a segment match
        
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
                    corr = np.corrcoef(ref_window, input_window)[0,1]
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

    def analyze_subsequence(self, ref_seq: np.ndarray, input_seq: np.ndarray) -> float:
        """Analyze similarity between two melody subsequences
        
        Args:
            ref_seq: Reference subsequence pitch sequence
            input_seq: Input subsequence pitch sequence
            
        Returns:
            Score between 0 and 100 indicating similarity
        """
        # Normalize sequences
        ref_mean = np.mean(ref_seq)
        ref_std = np.std(ref_seq)
        input_mean = np.mean(input_seq)
        input_std = np.std(input_seq)
        
        if ref_std > 0:
            ref_seq = (ref_seq - ref_mean) / ref_std
        if input_std > 0:
            input_seq = (input_seq - input_mean) / input_std
            
        # Calculate DTW score
        N, M = len(ref_seq), len(input_seq)
        cost = np.zeros((N + 1, M + 1))
        cost[0, 1:] = np.inf
        cost[1:, 0] = np.inf
        
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                diff = abs(ref_seq[i-1] - input_seq[j-1])
                diff_cost = min(100, diff * 12)  # Higher penalty for pitch differences
                
                cost[i, j] = diff_cost + min(
                    cost[i-1, j-1],     # Match
                    cost[i-1, j] + 3,     # Gap penalties
                    cost[i, j-1] + 3
                )
                
        final_cost = cost[N, M]
        path_length = N + M
        avg_cost = final_cost / path_length
        dtw_score = max(0, 100 - avg_cost)
        
        # Calculate contour similarity
        ref_contour = np.diff(ref_seq)
        input_contour = np.diff(input_seq)
        ref_contour = np.sign(ref_contour)
        input_contour = np.sign(input_contour)
        
        if len(ref_contour) > 0 and len(input_contour) > 0:
            min_len = min(len(ref_contour), len(input_contour))
            contour_match = np.mean(ref_contour[:min_len] == input_contour[:min_len])
            
            # Weight direction changes more heavily
            ref_changes = np.where(np.diff(ref_contour[:min_len]) != 0)[0]
            input_changes = np.where(np.diff(input_contour[:min_len]) != 0)[0]
            if len(ref_changes) > 0 and len(input_changes) > 0:
                changes_match = np.mean(np.abs(ref_changes - input_changes[:, None]) <= 2)
                contour_match = 0.7 * contour_match + 0.3 * changes_match
        else:
            contour_match = 0
            
        # Calculate correlation
        try:
            corr = np.corrcoef(ref_seq, input_seq)[0, 1]
            if np.isnan(corr):
                corr = 0
        except:
            corr = 0
            
        # Combine scores
        return (
            0.5 * dtw_score +
            0.3 * contour_match * 100 +
            0.2 * max(0, corr * 100)
        )

    def align_sequences(self, ref_seq: np.ndarray, input_seq: np.ndarray) -> tuple:
        """Align two sequences using DTW and return aligned subsequences
        
        Args:
            ref_seq: Reference sequence
            input_seq: Input sequence
            
        Returns:
            Tuple of (aligned reference, aligned input, alignment score)
        """
        # Normalize sequences
        ref_mean = np.mean(ref_seq)
        ref_std = np.std(ref_seq)
        input_mean = np.mean(input_seq)
        input_std = np.std(input_seq)
        
        if ref_std > 0:
            ref_seq = (ref_seq - ref_mean) / ref_std
        if input_std > 0:
            input_seq = (input_seq - input_mean) / input_std
            
        # Calculate DTW matrix
        N, M = len(ref_seq), len(input_seq)
        cost = np.zeros((N + 1, M + 1))
        cost[0, 1:] = np.inf
        cost[1:, 0] = np.inf
        
        # Track path
        path = np.zeros((N + 1, M + 1, 2), dtype=int)
        
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                diff = abs(ref_seq[i-1] - input_seq[j-1])
                diff_cost = min(100, diff * 10)
                
                choices = [
                    (cost[i-1, j-1], (-1, -1)),  # Match
                    (cost[i-1, j] + 3, (-1, 0)),  # Gap penalties
                    (cost[i, j-1] + 3, (0, -1))   # Gap penalties
                ]
                
                best_cost, best_step = min(choices, key=lambda x: x[0])
                cost[i, j] = diff_cost + best_cost
                path[i, j] = best_step
                
        # Backtrack to find alignment
        i, j = N, M
        ref_aligned = []
        input_aligned = []
        path_cost = 0
        path_length = 0
        
        while i > 0 and j > 0:
            di, dj = path[i, j]
            if di == -1 and dj == -1:  # Match
                ref_aligned.append(ref_seq[i-1])
                input_aligned.append(input_seq[j-1])
                diff = abs(ref_seq[i-1] - input_seq[j-1])
                path_cost += min(100, diff * 10)
                path_length += 1
            elif di == -1:  # Gap in input
                ref_aligned.append(ref_seq[i-1])
                input_aligned.append(0)  # Gap
                path_cost += 3
                path_length += 1
            else:  # Gap in reference
                ref_aligned.append(0)  # Gap
                input_aligned.append(input_seq[j-1])
                path_cost += 3
                path_length += 1
            i += di
            j += dj
            
        # Reverse sequences
        ref_aligned = np.array(ref_aligned[::-1])
        input_aligned = np.array(input_aligned[::-1])
        
        # Calculate alignment score
        avg_cost = path_cost / path_length if path_length > 0 else 100
        alignment_score = max(0, 100 - avg_cost)
        
        return ref_aligned, input_aligned, alignment_score

    def score_aligned_sequences(self, ref_aligned: np.ndarray, input_aligned: np.ndarray) -> float:
        """Score aligned sequences based on various metrics
        
        Args:
            ref_aligned: Aligned reference sequence
            input_aligned: Aligned input sequence
            
        Returns:
            Score between 0 and 100
        """
        # Remove gaps for contour analysis
        ref_valid = ref_aligned != 0
        input_valid = input_aligned != 0
        valid_mask = ref_valid & input_valid
        
        if not np.any(valid_mask):
            return 0.0
            
        ref_seq = ref_aligned[valid_mask]
        input_seq = input_aligned[valid_mask]
        
        # Calculate contour similarity
        ref_contour = np.diff(ref_seq)
        input_contour = np.diff(input_seq)
        ref_contour = np.sign(ref_contour)
        input_contour = np.sign(input_contour)
        
        if len(ref_contour) > 0 and len(input_contour) > 0:
            contour_match = np.mean(ref_contour == input_contour)
            
            # Weight direction changes more heavily
            ref_changes = np.where(np.diff(ref_contour) != 0)[0]
            input_changes = np.where(np.diff(input_contour) != 0)[0]
            if len(ref_changes) > 0 and len(input_changes) > 0:
                changes_match = np.mean(np.abs(ref_changes - input_changes[:, None]) <= 2)
                contour_match = 0.7 * contour_match + 0.3 * changes_match
        else:
            contour_match = 0
            
        # Calculate correlation
        try:
            corr = np.corrcoef(ref_seq, input_seq)[0, 1]
            if np.isnan(corr):
                corr = 0
        except:
            corr = 0
            
        # Calculate gap ratio
        gap_ratio = np.mean(valid_mask)
        
        return (
            0.4 * contour_match * 100 +
            0.3 * max(0, corr * 100) +
            0.3 * gap_ratio * 100
        )

    def find_matching_segments(self, ref_pitch: np.ndarray, input_pitch: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Find matching segments between reference and input pitch sequences
        
        Args:
            ref_pitch: Reference pitch sequence
            input_pitch: Input pitch sequence
            
        Returns:
            List of tuples (ref_start, ref_end, input_start, input_end, match_score)
        """
        matches = []
        min_segment_len = 10  # Minimum segment length
        max_segment_len = 100  # Maximum segment length
        step_size = 5  # Step size for sliding window
        
        # Calculate overall pitch statistics
        ref_mean = np.mean(ref_pitch)
        ref_std = np.std(ref_pitch)
        input_mean = np.mean(input_pitch)
        input_std = np.std(input_pitch)
        
        # Calculate overall pitch difference
        overall_pitch_diff = abs(ref_mean - input_mean) / ref_mean if ref_mean > 0 else float('inf')
        
        # If overall pitch difference is too large, return no matches
        if overall_pitch_diff > 1.0:  # More than 100% difference in average pitch
            return []
            
        for segment_len in range(min_segment_len, max_segment_len + 1, step_size):
            # Slide window over reference sequence
            for ref_start in range(0, len(ref_pitch) - segment_len + 1, step_size):
                ref_end = ref_start + segment_len
                ref_segment = ref_pitch[ref_start:ref_end]
                
                # Normalize reference segment
                ref_mean = np.mean(ref_segment)
                ref_std = np.std(ref_segment)
                if ref_std == 0:
                    continue
                ref_norm = (ref_segment - ref_mean) / ref_std
                
                # Calculate reference contour
                ref_contour = np.diff(ref_norm)
                ref_contour_pattern = np.sign(ref_contour)
                
                # Slide window over input sequence
                best_match = None
                best_score = -1
                
                for input_start in range(0, len(input_pitch) - segment_len + 1, step_size):
                    input_end = input_start + segment_len
                    input_segment = input_pitch[input_start:input_end]
                    
                    # Normalize input segment
                    input_mean = np.mean(input_segment)
                    input_std = np.std(input_segment)
                    if input_std == 0:
                        continue
                    input_norm = (input_segment - input_mean) / input_std
                    
                    # Calculate correlation
                    corr = np.corrcoef(ref_norm, input_norm)[0, 1]
                    if np.isnan(corr):
                        continue
                        
                    # Calculate pitch difference penalty
                    pitch_diff = abs(ref_mean - input_mean) / ref_mean
                    pitch_penalty = max(0, 1 - pitch_diff * 2)
                    
                    # Calculate contour match
                    input_contour = np.diff(input_norm)
                    input_contour_pattern = np.sign(input_contour)
                    contour_match = np.mean(ref_contour_pattern == input_contour_pattern)
                    
                    # Calculate contour correlation
                    contour_corr = np.corrcoef(ref_contour, input_contour)[0, 1]
                    if np.isnan(contour_corr):
                        contour_corr = 0
                        
                    # Calculate rhythm match
                    rhythm_match = 1.0 - abs(ref_std - input_std) / max(ref_std, input_std)
                    
                    # Combine scores with stricter criteria
                    match_score = (
                        0.3 * max(0, corr) +  # Base correlation
                        0.3 * contour_match +  # Contour direction match
                        0.2 * max(0, contour_corr) +  # Contour shape correlation
                        0.1 * pitch_penalty +  # Pitch difference penalty
                        0.1 * rhythm_match  # Rhythm similarity
                    )
                    
                    # Apply minimum thresholds
                    if (corr < 0.2 or  # Minimum correlation
                        contour_match < 0.3 or  # Minimum contour match
                        pitch_penalty < 0.2):  # Maximum pitch difference
                        continue
                        
                    # Update best match
                    if match_score > best_score and match_score > self.min_segment_score:
                        best_score = match_score
                        best_match = (ref_start, ref_end, input_start, input_end, match_score)
                
                if best_match is not None:
                    matches.append(best_match)
        
        # Filter overlapping matches
        filtered_matches = []
        for match in sorted(matches, key=lambda x: x[4], reverse=True):  # Sort by score
            ref_start, ref_end, input_start, input_end, score = match
            
            # Check for significant overlap with existing matches
            overlap = False
            for existing in filtered_matches:
                ref_s, ref_e, in_s, in_e, _ = existing
                
                # Calculate overlap ratios
                ref_overlap = min(ref_end, ref_e) - max(ref_start, ref_s)
                input_overlap = min(input_end, in_e) - max(input_start, in_s)
                
                if ref_overlap > 0 and input_overlap > 0:
                    ref_ratio = ref_overlap / (ref_end - ref_start)
                    input_ratio = input_overlap / (input_end - input_start)
                    if ref_ratio > 0.3 or input_ratio > 0.3:  # Allow some overlap
                        overlap = True
                        break
            
            if not overlap:
                filtered_matches.append(match)
        
        return filtered_matches

    def align_segment(self, ref_seq: np.ndarray, input_seq: np.ndarray) -> tuple:
        """Align two segments using DTW with strict penalties
        
        Args:
            ref_seq: Reference segment
            input_seq: Input segment
            
        Returns:
            Tuple of (aligned reference, aligned input, alignment score)
        """
        if len(ref_seq) < 2 or len(input_seq) < 2:
            return np.array([]), np.array([]), 0.0
            
        # Convert to relative pitches for key invariance
        ref_relative = np.diff(ref_seq)
        input_relative = np.diff(input_seq)
        
        # Calculate DTW matrix
        N, M = len(ref_relative), len(input_relative)
        cost = np.zeros((N + 1, M + 1))
        cost[0, 1:] = np.inf
        cost[1:, 0] = np.inf
        
        # Track path
        path = np.zeros((N + 1, M + 1, 2), dtype=int)
        
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # Use relative pitch difference
                diff = abs(ref_relative[i-1] - input_relative[j-1])
                diff_cost = min(100, diff * 8)  # Lower multiplier for pitch differences
                
                choices = [
                    (cost[i-1, j-1], (-1, -1)),  # Match
                    (cost[i-1, j] + 4, (-1, 0)),  # Lower gap penalty
                    (cost[i, j-1] + 4, (0, -1))   # Lower gap penalty
                ]
                
                best_cost, best_step = min(choices, key=lambda x: x[0])
                cost[i, j] = diff_cost + best_cost
                path[i, j] = best_step
                
        # Backtrack to find alignment
        i, j = N, M
        ref_aligned = []
        input_aligned = []
        path_cost = 0
        path_length = 0
        
        while i > 0 and j > 0:
            di, dj = path[i, j]
            if di == -1 and dj == -1:  # Match
                ref_aligned.append(ref_seq[i])
                input_aligned.append(input_seq[j])
                diff = abs(ref_relative[i-1] - input_relative[j-1])
                path_cost += min(100, diff * 8)
                path_length += 1
            elif di == -1:  # Gap in input
                ref_aligned.append(ref_seq[i])
                input_aligned.append(0)
                path_cost += 4
                path_length += 1
            else:  # Gap in reference
                ref_aligned.append(0)
                input_aligned.append(input_seq[j])
                path_cost += 4
                path_length += 1
            i += di
            j += dj
            
        # Add first elements which were skipped due to diff
        if i > 0:
            ref_aligned.append(ref_seq[0])
            input_aligned.append(0)
        if j > 0:
            ref_aligned.append(0)
            input_aligned.append(input_seq[0])
            
        # Reverse sequences
        ref_aligned = np.array(ref_aligned[::-1])
        input_aligned = np.array(input_aligned[::-1])
        
        # Calculate alignment score with boost for good alignments
        avg_cost = path_cost / max(1, path_length)
        base_score = max(0, 100 - avg_cost)
        
        # Add boost for good alignments
        if base_score > 60:
            boost = (base_score - 60) * 0.25  # 25% boost for scores above 60
            alignment_score = min(100, base_score + boost)
        else:
            alignment_score = base_score
        
        return ref_aligned, input_aligned, alignment_score

    def score_aligned_segment(self, ref_aligned: np.ndarray, input_aligned: np.ndarray) -> float:
        """Score aligned segment based on various metrics
        
        Args:
            ref_aligned: Aligned reference segment
            input_aligned: Aligned input segment
            
        Returns:
            Score between 0 and 100
        """
        # Remove gaps for contour analysis
        ref_valid = ref_aligned != 0
        input_valid = input_aligned != 0
        valid_mask = ref_valid & input_valid
        
        if not np.any(valid_mask) or np.sum(valid_mask) < 2:
            return 0.0
            
        ref_seq = ref_aligned[valid_mask]
        input_seq = input_aligned[valid_mask]
        
        # Convert to relative pitches for key invariance
        ref_relative = np.diff(ref_seq)
        input_relative = np.diff(input_seq)
        
        # Calculate contour similarity using relative pitches
        ref_contour = np.sign(ref_relative)
        input_contour = np.sign(input_relative)
        
        if len(ref_contour) > 0 and len(input_contour) > 0:
            contour_match = np.mean(ref_contour == input_contour)
            
            # Weight direction changes more heavily
            ref_changes = np.where(np.diff(ref_contour) != 0)[0]
            input_changes = np.where(np.diff(input_contour) != 0)[0]
            if len(ref_changes) > 0 and len(input_changes) > 0:
                changes_match = np.mean(np.abs(ref_changes - input_changes[:, None]) <= 3)
                contour_match = 0.7 * contour_match + 0.3 * changes_match
                
                # Add boost for good contour matches
                if contour_match > 0.6:
                    contour_match = min(1.0, contour_match * 1.2)
        else:
            contour_match = 0
            
        # Calculate correlation with more emphasis on local patterns
        try:
            # Get local correlations in windows
            window_size = min(20, len(ref_relative) // 2)
            if window_size > 5:  # Only if we have enough data
                correlations = []
                for i in range(0, len(ref_relative) - window_size, window_size // 2):
                    window_ref = ref_relative[i:i+window_size]
                    window_input = input_relative[i:i+window_size]
                    try:
                        corr = np.corrcoef(window_ref, window_input)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                    except:
                        continue
                if correlations:
                    # Weight higher correlations more
                    correlations = np.array(correlations)
                    weights = np.exp(correlations) / np.sum(np.exp(correlations))
                    local_corr = np.sum(correlations * weights)
                    
                    # Add boost for good correlations
                    if local_corr > 0.6:
                        local_corr = min(1.0, local_corr * 1.2)
                else:
                    local_corr = 0
            else:
                local_corr = 0
                
            # Also get global correlation
            global_corr = np.corrcoef(ref_relative, input_relative)[0, 1]
            if np.isnan(global_corr):
                global_corr = 0
                
            # Combine local and global with boost for good correlations
            corr = 0.7 * max(local_corr, 0) + 0.3 * max(global_corr, 0)
            if corr > 0.6:
                corr = min(1.0, corr * 1.2)
        except:
            corr = 0
            
        # Calculate gap ratio with less penalty
        gap_ratio = np.mean(valid_mask)
        gap_score = 0.85 + 0.15 * gap_ratio  # Higher base score
        
        # Weight the components with more emphasis on contour
        base_score = (
            0.5 * contour_match * 100 +  # More weight on contour for key invariance
            0.3 * max(0, corr * 100) +   # Correlation of relative pitches
            0.2 * gap_score * 100        # Less weight on gaps
        )
        
        # Add final boost for good scores
        if base_score > 65:  # Lower threshold for boost
            boost = (base_score - 65) * 0.25  # 25% boost for scores above 65
            return min(100, base_score + boost)
        else:
            return base_score

    def match_melody(self, reference: np.ndarray, input_melody: np.ndarray) -> float:
        """Match input melody against reference melody using segment-based DTW alignment
        
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
        
        # Calculate length ratio
        len_ratio = min(len(ref_pitch), len(input_pitch)) / max(len(ref_pitch), len(input_pitch))
        
        # Find matching segments
        matches = self.find_matching_segments(ref_pitch, input_pitch)
        
        if not matches:
            return 0.0
            
        # Score each matching segment
        segment_scores = []
        covered_ref_frames = set()  # Track which reference frames are covered
        
        for ref_start, ref_end, input_start, input_end, match_score in matches:
            # Get segments
            ref_segment = ref_pitch[ref_start:ref_end + 1]
            input_segment = input_pitch[input_start:input_end + 1]
            
            # Add reference frames to covered set
            covered_ref_frames.update(range(ref_start, ref_end + 1))
            
            # Align segments
            ref_aligned, input_aligned, alignment_score = self.align_segment(ref_segment, input_segment)
            
            # Score aligned segments
            sequence_score = self.score_aligned_segment(ref_aligned, input_aligned)
            
            # Combine scores with adjusted weights
            segment_score = (
                0.5 * match_score * 100 +  # Increased weight for match quality
                0.3 * alignment_score +
                0.2 * sequence_score
            )
            
            # Weight by segment length
            weight = len(ref_segment) / len(ref_pitch)
            segment_scores.append((segment_score, weight))
            
        if not segment_scores:
            return 0.0
            
        # Calculate weighted average score
        total_weight = sum(weight for _, weight in segment_scores)
        if total_weight > 0:
            raw_score = sum(score * weight for score, weight in segment_scores) / total_weight
        else:
            raw_score = 0
            
        # Calculate coverage ratio using unique covered frames
        coverage_ratio = len(covered_ref_frames) / len(ref_pitch)
        
        # Apply coverage penalty first - this is most important
        # But be more lenient - good segments should boost score even with lower coverage
        if coverage_ratio < 0.1:  # Very low coverage still gets heavy penalty
            raw_score *= 0.2  # But not as severe as before
        elif coverage_ratio < 0.3:  # Low coverage gets moderate penalty
            raw_score *= 0.6 + (coverage_ratio * 1.2)  # Linear scaling from 0.6 to 0.96
        elif coverage_ratio < 0.7:  # Moderate coverage gets very mild penalty
            raw_score *= 0.92 + (coverage_ratio * 0.08)  # Linear scaling from 0.92 to 0.98
            
        # Apply length ratio penalty second - be more lenient
        # Different lengths are ok as long as the matching segments are good
        if len_ratio < 0.2:  # Very different lengths get mild penalty
            raw_score *= 0.85 + len_ratio  # Linear scaling from 0.85 to 1.05
        elif len_ratio < 0.5:  # Somewhat different lengths get very mild penalty
            raw_score *= 0.96 + (len_ratio * 0.08)  # Linear scaling from 0.96 to 1.0
            
        # Add a boost for high raw scores - if segments match well, be more forgiving
        if raw_score > 65:  # Lower threshold for boost
            boost = min(1.25, 1.0 + (raw_score - 65) / 80)  # Max 25% boost
            raw_score *= boost
            
        # Log final analysis
        logging.info(f"Segment analysis - Length ratio: {len_ratio:.2f}, Coverage: {coverage_ratio:.2f}, Raw score: {raw_score:.2f}, Final: {raw_score:.2f}")
        
        return min(100, raw_score)  # Cap at 100

    def get_detections(self) -> List[Dict[str, float]]:
        """Get list of detected note matches"""
        return self.detections
