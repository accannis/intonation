"""
Extract audio features for melody and phonetic matching
"""

import librosa
import numpy as np
from typing import Tuple, Dict, Union, Optional, Any
import logging
import os

from src.feature_extraction.feature_cache import FeatureCache
from src.audio_processing.file_processor import AudioFileProcessor

class AudioFeatureExtractor:
    """Extract audio features from a file"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the feature extractor with config"""
        self.sample_rate = config["sample_rate"]
        self.hop_length = config["hop_length"]
        self.n_mels = config["n_mels"]
        self.n_mfcc = config["n_mfcc"]
        self.n_fft = config["n_fft"]
        self.mel_power = config["mel_power"]
        self.lifter = config["lifter"]
        self.top_db = config["top_db"]
        self.f0_min = config["f0_min"]
        self.f0_max = config["f0_max"]
        self.delta_width = config["delta_width"]
        self.feature_version = config["feature_version"]
        
        # Initialize cache
        self.cache = FeatureCache()
        self.file_processor = AudioFileProcessor()
        
        # Create parameter dictionary for cache key
        self.parameters = {
            # Audio parameters
            'sample_rate': self.sample_rate,
            'hop_length': self.hop_length,
            
            # Mel spectrogram parameters
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'mel_power': self.mel_power,
            'top_db': self.top_db,
            
            # MFCC parameters
            'n_mfcc': self.n_mfcc,
            'lifter': self.lifter,
            'delta_width': self.delta_width,
            
            # Pitch detection parameters
            'f0_min': self.f0_min,
            'f0_max': self.f0_max,
            
            # Feature version (increment when making breaking changes)
            'feature_version': self.feature_version
        }
        
    def extract_features(self, audio_data: Union[str, np.ndarray], include_waveform: bool = False) -> Optional[Dict[str, np.ndarray]]:
        """Extract features from audio data or file"""
        try:
            # Load audio data if path provided
            if isinstance(audio_data, str):
                if not os.path.isfile(audio_data):
                    raise FileNotFoundError(f"Audio file not found: {audio_data}")
                    
                # Try to get from cache first
                audio_path = os.path.abspath(audio_data)
                logging.info(f"Checking cache for {os.path.basename(audio_path)}")
                cached_features = self.cache.get_features(audio_path, self.parameters)
                if cached_features is not None:
                    logging.info(f"Using cached features for {os.path.basename(audio_path)}")
                    return cached_features
                
                # Load audio file
                logging.info(f"Loading audio file: {audio_data}")
                audio_data, _ = librosa.load(audio_data, sr=self.sample_rate, duration=30.0)  # Only load first 30 seconds for testing
            
            # Extract melody features (pitch contour)
            logging.info("Extracting pitch features...")
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, 
                sr=self.sample_rate,
                fmin=self.f0_min,
                fmax=self.f0_max
            )
            melody = np.where(voiced_flag, f0, 0)
            
            # Extract MFCC features
            logging.info("Extracting MFCC features...")
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                power=self.mel_power
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.top_db)
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(
                S=mel_spec_db,
                n_mfcc=self.n_mfcc,
                lifter=self.lifter
            )
            
            # Add deltas
            mfcc_delta = librosa.feature.delta(mfcc, width=self.delta_width)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=self.delta_width)
            
            # Stack MFCCs and deltas
            mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            
            # Prepare features dict
            features = {
                'melody': np.column_stack([f0, voiced_probs]),
                'phonetic': mfcc_features,
                'duration': float(len(audio_data) / self.sample_rate)
            }
            
            # Cache features if we loaded from file
            if isinstance(audio_data, str):
                logging.info(f"Caching features for {os.path.basename(audio_path)}")
                self.cache.cache_features(audio_path, self.parameters, features)
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            logging.exception("Full traceback:")
            return None

    def extract_features_from_file(self, audio_file: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract all features from an audio file"""
        try:
            logging.info(f"Loading audio file: {audio_file}")
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Extract pitch features
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                sr=sr,
                fmin=self.f0_min,
                fmax=self.f0_max
            )
            
            # Extract MFCC features
            S = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=self.mel_power
            )
            
            # Convert to log scale
            log_S = librosa.power_to_db(S, ref=np.max, top_db=self.top_db)
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(
                S=log_S,
                n_mfcc=self.n_mfcc,
                lifter=self.lifter
            )
            
            # Add deltas
            mfcc_delta = librosa.feature.delta(mfcc, width=self.delta_width)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=self.delta_width)
            
            # Stack MFCCs and deltas
            mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            
            # Prepare features dict
            features = {
                'melody': np.column_stack([f0, voiced_probs]),
                'phonetic': mfcc_features,
                'duration': float(len(y) / sr)
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            logging.exception("Full traceback:")
            return None
