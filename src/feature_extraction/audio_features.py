"""
Extract audio features for melody and phonetic matching
"""

import librosa
import numpy as np
from typing import Tuple, Dict, Union, Optional
import logging
import os

from src.feature_extraction.feature_cache import FeatureCache
from src.audio_processing.file_processor import AudioFileProcessor

class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = 128
        self.n_mfcc = 20
        self.feature_cache = FeatureCache()
        self.file_processor = AudioFileProcessor()
        
    def extract_features(self, audio_data: Union[str, np.ndarray], include_waveform: bool = False) -> Optional[Dict[str, np.ndarray]]:
        """Extract features from audio data or file
        
        Args:
            audio_data: Either a path to an audio file or numpy array of audio samples
            include_waveform: If True, include normalized waveform data in output
            
        Returns:
            Dictionary containing extracted features:
                - melody: Pitch contour for melody matching
                - phonetic: MFCC features for phonetic matching
                - waveform: Processed audio waveform (only if include_waveform=True)
        """
        try:
            # If audio_data is a string, check cache first
            if isinstance(audio_data, str) and os.path.isfile(audio_data):
                cached_features = self.feature_cache.get_features(audio_data)
                if cached_features is not None:
                    # Add waveform if needed
                    if include_waveform:
                        # Don't log loading message for cached features
                        audio_data = self.file_processor.load_audio(audio_data, log_info=False)
                        waveform = audio_data.waveform / np.max(np.abs(audio_data.waveform))
                        cached_features['waveform'] = waveform
                    return cached_features
                
            # Load audio if needed
            if isinstance(audio_data, str):
                audio_data = self.file_processor.load_audio(audio_data, log_info=False)
                y = audio_data.waveform
                self.sample_rate = audio_data.sample_rate
            else:
                y = audio_data
                
            # Convert to float32 if needed
            if y.dtype != np.float32:
                y = y.astype(np.float32)
                
            # Ensure audio is mono
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
                
            # Check if we have enough samples
            min_samples = self.hop_length * 4  # Ensure enough samples for features
            if len(y) < min_samples:
                # Pad with zeros if needed
                y = np.pad(y, (0, min_samples - len(y)))
                
            # Extract melody features (pitch)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Clean up pitch contour
            melody = np.where(voiced_flag, f0, 0.0)
            
            # Extract phonetic features (MFCCs)
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length
            )
            
            # Initialize features dictionary
            features = {
                'melody': melody,
                'phonetic': mfccs
            }
            
            # Only include waveform if requested
            if include_waveform:
                # Normalize waveform to -1 to 1 range
                waveform = y / np.max(np.abs(y))
                features['waveform'] = waveform
            
            # Cache features if this was loaded from a file
            if isinstance(audio_data, str) and os.path.isfile(audio_data):
                # Cache without waveform
                cache_features = {k: v for k, v in features.items() if k != 'waveform'}
                self.feature_cache.cache_features(audio_data, cache_features)
            
            # Return features dictionary
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            logging.exception("Full traceback:")
            return None
