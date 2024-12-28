import librosa
import numpy as np
from typing import Tuple, Dict
from pyaudiosource import AudioSource, AudioMeter

class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.audio_source = None
        self.audio_meter = AudioMeter()
        
    def start_live_input(self, device_id=None, callback=None):
        """Start live audio input"""
        self.audio_source = AudioSource(sample_rate=self.sample_rate)
        if callback:
            self.audio_source.set_callback(callback)
        self.audio_source.start(device_id)
        
    def stop_live_input(self):
        """Stop live audio input"""
        if self.audio_source:
            self.audio_source.stop()
            
    def extract_features(self, audio_path: str = None, audio_data: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Extract relevant features from audio file or data
        
        Args:
            audio_path: Path to audio file (optional)
            audio_data: Audio data as numpy array (optional)
            
        Returns:
            Dictionary containing extracted features:
            - pitch_sequence: Fundamental frequency over time
            - mfcc: Mel-frequency cepstral coefficients
            - onset_env: Onset strength envelope
        """
        # Load audio
        if audio_path:
            y, _ = librosa.load(audio_path, sr=self.sample_rate)
        elif audio_data is not None:
            y = audio_data
        else:
            raise ValueError("Must provide either audio_path or audio_data")
        
        # Extract pitch (f0) using PYIN algorithm
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=13,
            hop_length=self.hop_length
        )
        
        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return {
            'pitch_sequence': f0,
            'voiced_flag': voiced_flag,
            'mfcc': mfcc,
            'onset_env': onset_env
        }
        
    def process_audio_frame(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Process a single frame of audio data"""
        # Get audio level
        level = self.audio_meter.process(data)
        
        # Extract features
        features = self.extract_features(audio_data=data)
        features['level'] = level
        
        return features
