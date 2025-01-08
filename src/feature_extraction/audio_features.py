import librosa
import numpy as np
from typing import Tuple, Dict, Union, Optional
from pyaudiosource import AudioSource, AudioMeter
import logging

class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = 128
        self.n_mfcc = 20
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
            
    def extract_features(self, audio_data: Union[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """Extract features from audio data or file"""
        try:
            # If audio_data is a string, treat it as a file path
            if isinstance(audio_data, str):
                y, _ = librosa.load(audio_data, sr=self.sample_rate)
            else:
                y = audio_data
                
            # Convert to float32 if needed
            if y.dtype != np.float32:
                y = y.astype(np.float32)
                
            # Ensure audio is mono
            if len(y.shape) > 1:
                y = y.mean(axis=1)
                
            # Check if we have enough samples
            min_samples = self.hop_length * 4  # Ensure enough samples for features
            if len(y) < min_samples:
                # Pad with zeros if needed
                y = np.pad(y, (0, min_samples - len(y)))
                
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length
            )
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(
                y=y,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Get the most prominent pitch at each time
            pitch = []
            for time_idx in range(pitches.shape[1]):
                index = magnitudes[:, time_idx].argmax()
                pitch.append(pitches[index, time_idx])
            pitch = np.array(pitch)
            
            # Return features dictionary
            return {
                'mfccs': mfccs,
                'mel_spec': mel_spec,
                'pitch': pitch,
                'audio_data': y,  # Include the original audio data
                'sample_rate': self.sample_rate  # Include the sample rate
            }
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            logging.exception("Full traceback:")
            return None
        
    def process_audio_frame(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Process a single frame of audio data"""
        # Get audio level
        level = self.audio_meter.process(data)
        
        # Extract features
        features = self.extract_features(audio_data=data)
        features['level'] = level
        
        return features
