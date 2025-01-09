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
            audio_path = None
            # If audio_data is a string, check cache first
            if isinstance(audio_data, str) and os.path.isfile(audio_data):
                audio_path = os.path.abspath(audio_data)  # Ensure we have full path
                logging.info(f"Checking cache for {os.path.basename(audio_path)}")
                cached_features = self.feature_cache.get_features(audio_path)
                if cached_features is not None:
                    logging.info(f"Using cached features for {os.path.basename(audio_path)}")
                    # Add waveform if needed
                    if include_waveform and 'waveform' not in cached_features:
                        # Don't log loading message for cached features
                        audio_data = self.file_processor.load_audio(audio_path, log_info=False)
                        waveform = audio_data.waveform / np.max(np.abs(audio_data.waveform))
                        cached_features['waveform'] = waveform
                    return cached_features
                
                # Load audio data from file
                audio_data = self.file_processor.load_audio(audio_path)
                if audio_data is None:
                    return None
                audio_data = audio_data.waveform
                
            # Ensure audio data is the right shape
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
                
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Extract melody features (pitch contour)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            melody = np.where(voiced_flag, f0, 0)
            
            # Extract phonetic features (MFCC)
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length
            )
            
            # Create features dictionary
            features = {
                'melody': melody.astype(np.float32),
                'phonetic': mfcc.astype(np.float32)
            }
            
            # Add waveform if requested
            if include_waveform:
                features['waveform'] = audio_data.astype(np.float32)
            
            # Cache features if we loaded from file
            if audio_path:
                logging.info(f"Attempting to cache features for {os.path.basename(audio_path)}")
                # Cache without waveform to save space
                cache_features = {k: v for k, v in features.items() if k != 'waveform'}
                success = self.feature_cache.cache_features(audio_path, cache_features)
                if success:
                    logging.info(f"Successfully cached features for {os.path.basename(audio_path)}")
                else:
                    logging.error(f"Failed to cache features for {os.path.basename(audio_path)}")
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            logging.exception("Full traceback:")
            return None
