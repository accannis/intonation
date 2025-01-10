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
        # Create parameter dictionary for cache key
        self.parameters = {
            # Audio parameters
            'sample_rate': sample_rate,
            'hop_length': hop_length,
            
            # Mel spectrogram parameters
            'n_mels': self.n_mels,
            'n_fft': 2048,
            'mel_power': 2.0,
            'top_db': 80,
            
            # MFCC parameters
            'n_mfcc': self.n_mfcc,
            'lifter': 22,
            'delta_width': 9,
            
            # Pitch detection parameters
            'f0_min': librosa.note_to_hz('C2'),
            'f0_max': librosa.note_to_hz('C7'),
            
            # Feature version (increment when making breaking changes)
            'feature_version': 1
        }
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
                cached_features = self.feature_cache.get_features(audio_path, self.parameters)
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
            # First compute mel spectrogram with more frequency resolution
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                n_fft=2048,  # Increased FFT size for better frequency resolution
                power=2.0  # Use power spectrogram
            )
            
            # Convert to log scale with proper scaling
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
            
            # Extract MFCCs from log mel spectrogram with liftering
            mfcc = librosa.feature.mfcc(
                S=mel_spec_db,  # Use pre-computed mel spectrogram
                n_mfcc=self.n_mfcc,
                sr=self.sample_rate,
                lifter=22  # Apply liftering to emphasize important coefficients
            )
            
            # Add delta features with proper width
            mfcc_delta = librosa.feature.delta(mfcc, width=9)  # Wider window for better temporal context
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=9)
            
            # Normalize each feature stream independently with robust scaling
            def robust_normalize(x):
                # Calculate robust statistics
                q1 = np.percentile(x, 25, axis=1, keepdims=True)
                q3 = np.percentile(x, 75, axis=1, keepdims=True)
                median = np.median(x, axis=1, keepdims=True)
                iqr = q3 - q1
                # Scale using IQR and center using median
                return (x - median) / (iqr + 1e-6)
            
            mfcc = robust_normalize(mfcc)
            mfcc_delta = robust_normalize(mfcc_delta)
            mfcc_delta2 = robust_normalize(mfcc_delta2)
            
            # Stack all features with proper weighting
            phonetic_features = np.vstack([
                mfcc,           # Base MFCC features
                0.5 * mfcc_delta,    # Reduced weight for first derivatives
                0.25 * mfcc_delta2   # Further reduced weight for second derivatives
            ])
            
            # Create features dictionary
            features = {
                'melody': melody.astype(np.float32),
                'phonetic': phonetic_features.astype(np.float32)
            }
            
            # Add waveform if requested
            if include_waveform:
                features['waveform'] = audio_data.astype(np.float32)
            
            # Cache features if we loaded from file
            if audio_path:
                logging.info(f"Attempting to cache features for {os.path.basename(audio_path)}")
                # Cache without waveform to save space
                cache_features = {k: v for k, v in features.items() if k != 'waveform'}
                success = self.feature_cache.cache_features(audio_path, self.parameters, cache_features)
                if success:
                    logging.info(f"Successfully cached features for {os.path.basename(audio_path)}")
                else:
                    logging.error(f"Failed to cache features for {os.path.basename(audio_path)}")
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            logging.exception("Full traceback:")
            return None
